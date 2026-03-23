/**
 * drilldown.js — Single-ticker deep analysis view
 * TradingView chart with EMAs + Order Blocks
 * Buy/Sell ladder order buttons + auto-backtest
 */

const DrillDown = {
    currentSymbol: null,
    activeOrderId: null,

    async render(symbol, container) {
        this.currentSymbol = symbol;
        container.innerHTML = '<div style="text-align:center; padding:2rem; color:var(--text-dim);">Loading analysis...</div>';

        try {
            const data = await API.scanSymbol(symbol);
            if (data.error) {
                container.innerHTML = `<div style="color:var(--red); padding:1rem;">${data.error}</div>`;
                return;
            }

            const sigCss = this.signalCssClass(data.signal || '');
            const rid = data.regime_id;
            const regCss = rid <= 1 ? 'bull' : (rid >= 5 ? 'bear' : 'neutral');
            const chg = data.change_1d;
            const chgStr = chg != null ? `${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%` : '--';
            const chgCss = chg != null ? (chg >= 0 ? 'bull' : 'bear') : '';

            container.innerHTML = `
                <span class="back-link" onclick="App.showTab('screener')">&larr; Back to Hits</span>
                <h2 style="font-size:1.1rem; color:var(--chrome); margin:0.3rem 0;">${data.symbol}
                    <span style="font-size:0.8rem; font-weight:400; color:var(--text-dim);">$${(data.price || 0).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</span>
                    <span style="font-size:0.8rem; font-weight:400;" class="${chgCss}">${chgStr}</span>
                </h2>

                <div class="signal-banner ${sigCss}">
                    ${data.signal || 'N/A'}
                    <div style="font-size:0.75rem; font-weight:400; margin-top:2px; opacity:0.85;">
                        ${data.action || ''}
                    </div>
                </div>

                <!-- Order Panel -->
                <div class="order-panel">
                    <input type="number" class="order-qty" id="order-qty" value="1" min="1" placeholder="Qty">
                    <button class="btn-buy" onclick="DrillDown.placeLadder('${data.symbol}', 'buy')">Buy</button>
                    <button class="btn-sell" onclick="DrillDown.placeLadder('${data.symbol}', 'sell')">Sell</button>
                    <span style="font-size:0.55rem; color:var(--text-dim); font-family:var(--mono);">Ladder: ±$0.10 × 15</span>
                </div>
                <div class="order-status" id="order-status"></div>

                <!-- Metrics -->
                <div class="metrics-grid">
                    <div class="metric-card"><div class="label">Regime</div><div class="value ${regCss}">${data.regime_label || '?'}</div></div>
                    <div class="metric-card"><div class="label">Confidence</div><div class="value">${data.regime_confidence ? Math.round(data.regime_confidence * 100) + '%' : '?'}</div></div>
                    <div class="metric-card"><div class="label">Confs</div><div class="value">${data.confirmations_met || 0}/${data.confirmations_total || 12}</div></div>
                    <div class="metric-card"><div class="label">Streak</div><div class="value">${data.regime_streak || '?'}b</div></div>
                    <div class="metric-card"><div class="label">RSI</div><div class="value">${data.rsi ? Math.round(data.rsi) : '?'}</div></div>
                    <div class="metric-card"><div class="label">ADX</div><div class="value">${data.adx ? Math.round(data.adx) : '?'}</div></div>
                </div>

                ${this.renderConfirmations(data.confirmation_detail)}

                <!-- Backtest Summary (auto-loaded) -->
                <div id="dd-backtest-summary" style="margin:0.3rem 0;">
                    <div style="color:var(--text-dim); font-size:0.65rem; font-family:var(--mono);">Loading backtest...</div>
                </div>

                <!-- TradingView Chart -->
                <div id="tv-chart-container" style="margin:0.6rem 0; border-radius:6px; overflow:hidden; height:500px;"></div>

                <!-- Regime Chart (Plotly) -->
                <div class="chart-container" id="dd-price-chart"></div>

                <!-- Backtest trade history -->
                <div id="dd-backtest-area"></div>

                <div style="margin-top:0.5rem;">
                    <button class="btn btn-sm" onclick="DrillDown.loadOptions('${data.symbol}')">Options Picks</button>
                </div>
                <div id="dd-options-area"></div>
            `;

            // Embed TradingView
            this.embedTradingView(data.symbol);

            // Regime chart
            if (data.chart_data) {
                Charts.priceWithRegimes('dd-price-chart', data.chart_data, `${data.symbol} Regime Analysis`);
            }

            // Auto-load backtest
            this.loadBacktest(data.symbol);

        } catch (err) {
            container.innerHTML = `<div style="color:var(--red); padding:1rem;">Error: ${err.message}</div>`;
        }
    },

    // ── Ladder Order ──
    async placeLadder(symbol, side) {
        const qty = parseInt(document.getElementById('order-qty').value) || 1;
        const statusEl = document.getElementById('order-status');
        statusEl.className = 'order-status pending';
        statusEl.textContent = `Placing ${side.toUpperCase()} ladder for ${qty} ${symbol}...`;

        try {
            const res = await API.ladderOrder(symbol, side, qty);
            if (res.error) {
                statusEl.className = 'order-status error';
                statusEl.textContent = res.error;
                return;
            }

            this.activeOrderId = res.order_id;
            this.pollLadder(res.order_id, statusEl);

        } catch (err) {
            statusEl.className = 'order-status error';
            statusEl.textContent = `Error: ${err.message}`;
        }
    },

    async pollLadder(orderId, statusEl) {
        const poll = async () => {
            try {
                const s = await API.ladderStatus(orderId);
                if (s.status === 'filled') {
                    statusEl.className = 'order-status success';
                    statusEl.textContent = `Filled @ $${s.fill_price} (attempt ${s.filled_attempt}/${s.max_attempts})`;
                    return;
                } else if (s.status === 'exhausted') {
                    statusEl.className = 'order-status error';
                    statusEl.textContent = `Not filled after ${s.max_attempts} attempts (last: $${s.current_price})`;
                    return;
                } else if (s.status === 'error') {
                    statusEl.className = 'order-status error';
                    statusEl.textContent = `Error: ${s.error || 'Unknown'}`;
                    return;
                } else {
                    statusEl.className = 'order-status pending';
                    statusEl.textContent = `${s.side?.toUpperCase()} attempt ${s.attempt}/${s.max_attempts} @ $${s.current_price || '...'}`;
                    setTimeout(poll, 1500);
                }
            } catch (_) {
                setTimeout(poll, 2000);
            }
        };
        poll();
    },

    // ── TradingView Chart ──
    embedTradingView(symbol) {
        const container = document.getElementById('tv-chart-container');
        if (!container) return;
        container.innerHTML = '';

        const script = document.createElement('script');
        script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
        script.type = 'text/javascript';
        script.async = true;
        script.innerHTML = JSON.stringify({
            "autosize": true,
            "symbol": symbol,
            "interval": "D",
            "timezone": "America/Chicago",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "backgroundColor": "rgba(0, 0, 0, 1)",
            "gridColor": "rgba(30, 30, 30, 1)",
            "hide_top_toolbar": false,
            "hide_legend": false,
            "allow_symbol_change": true,
            "save_image": false,
            "calendar": false,
            "hide_volume": false,
            "support_host": "https://www.tradingview.com",
            "studies": [
                { "id": "MAExp@tv-basicstudies", "inputs": { "length": 10 }, "styles": { "plot": { "color": "#22c55e", "linewidth": 2 } } },
                { "id": "MAExp@tv-basicstudies", "inputs": { "length": 20 }, "styles": { "plot": { "color": "#eab308", "linewidth": 2 } } },
                { "id": "MAExp@tv-basicstudies", "inputs": { "length": 50 }, "styles": { "plot": { "color": "#ef4444", "linewidth": 2 } } },
                { "id": "STD;Order_Block_Breaker_Block" }
            ]
        });

        const wrapper = document.createElement('div');
        wrapper.className = 'tradingview-widget-container';
        wrapper.style.height = '100%';
        wrapper.style.width = '100%';

        const inner = document.createElement('div');
        inner.className = 'tradingview-widget-container__widget';
        inner.style.height = 'calc(100% - 32px)';
        inner.style.width = '100%';

        wrapper.appendChild(inner);
        wrapper.appendChild(script);
        container.appendChild(wrapper);
    },

    signalCssClass(signal) {
        if (signal.includes('ENTER')) return 'signal-long-enter';
        if (signal.includes('CONFIRMING')) return 'signal-long-confirming';
        if (signal.includes('HOLD')) return 'signal-long-hold';
        if (signal.includes('EXIT')) return 'signal-exit';
        if (signal.includes('BEARISH')) return 'signal-bearish';
        return 'signal-cash';
    },

    renderConfirmations(detail) {
        if (!detail || !Object.keys(detail).length) return '';
        let html = '<div class="conf-grid">';
        for (const [name, passed] of Object.entries(detail)) {
            const cls = passed ? 'conf-pass' : 'conf-fail';
            const icon = passed ? '+' : '-';
            html += `<div class="conf-item ${cls}">${icon} ${name}</div>`;
        }
        html += '</div>';
        return html;
    },

    // ── Auto Backtest ──
    async loadBacktest(symbol) {
        const summary = document.getElementById('dd-backtest-summary');
        const area = document.getElementById('dd-backtest-area');

        try {
            const settings = await API.getSettings();
            const bt = await API.backtest(symbol, {
                strategy: settings.strategy || 'v2',
                min_confs: settings.min_confs || 6,
                cooldown: settings.cooldown || 3,
                regime_confirm: settings.regime_confirm || 2,
                capital: settings.initial_capital || 100000,
            });

            if (bt.error) {
                summary.innerHTML = `<div style="color:var(--red); font-size:0.7rem;">${bt.error}</div>`;
                return;
            }

            const m = bt.metrics;
            const retCss = (m.total_return_pct || 0) >= 0 ? 'bull' : 'bear';

            // Summary card
            summary.innerHTML = `
                <div class="bt-summary">
                    <div class="bt-stat"><div class="bt-label">Return</div><div class="bt-val ${retCss}">${(m.total_return_pct||0).toFixed(1)}%</div></div>
                    <div class="bt-stat"><div class="bt-label">Win Rate</div><div class="bt-val">${(m.win_rate||0).toFixed(0)}%</div></div>
                    <div class="bt-stat"><div class="bt-label">Sharpe</div><div class="bt-val">${(m.sharpe_ratio||0).toFixed(2)}</div></div>
                    <div class="bt-stat"><div class="bt-label">Max DD</div><div class="bt-val bear">${(m.max_drawdown_pct||0).toFixed(1)}%</div></div>
                    <div class="bt-stat"><div class="bt-label">Trades</div><div class="bt-val">${m.total_trades || 0}</div></div>
                    <div class="bt-stat"><div class="bt-label">Avg Win</div><div class="bt-val bull">${(m.avg_win_pct||0).toFixed(1)}%</div></div>
                    <div class="bt-stat"><div class="bt-label">Avg Loss</div><div class="bt-val bear">${(m.avg_loss_pct||0).toFixed(1)}%</div></div>
                    <div class="bt-stat"><div class="bt-label">Profit Factor</div><div class="bt-val">${(m.profit_factor||0).toFixed(2)}</div></div>
                </div>
            `;

            // Trade history
            if (area) {
                let tradeHtml = '';
                if (bt.equity_curve) {
                    tradeHtml += '<div class="chart-container" id="dd-equity-chart"></div>';
                }
                tradeHtml += this.renderTradeTable(bt.trades);
                area.innerHTML = tradeHtml;

                if (bt.equity_curve) {
                    Charts.equityCurve('dd-equity-chart', bt.equity_curve);
                }
            }

        } catch (err) {
            summary.innerHTML = `<div style="color:var(--text-dim); font-size:0.65rem;">Backtest unavailable</div>`;
        }
    },

    renderTradeTable(trades) {
        if (!trades || !trades.length) return '';
        let html = `<table class="trade-table"><thead><tr>
            <th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>P&L %</th><th>Reason</th>
        </tr></thead><tbody>`;

        trades.slice(-20).forEach(t => {
            const pnl = t.pnl_pct || 0;
            const pnlCss = pnl >= 0 ? 'color:#34d399' : 'color:#f87171';
            html += `<tr>
                <td>${(t.entry_date || '').substring(0, 10)}</td>
                <td>${(t.exit_date || '').substring(0, 10)}</td>
                <td>$${(t.entry_price || 0).toFixed(2)}</td>
                <td>$${(t.exit_price || 0).toFixed(2)}</td>
                <td style="${pnlCss}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}%</td>
                <td style="color:var(--text-dim); max-width:120px; overflow:hidden; text-overflow:ellipsis;">${t.exit_reason || ''}</td>
            </tr>`;
        });

        html += '</tbody></table>';
        return html;
    },

    async loadOptions(symbol) {
        const area = document.getElementById('dd-options-area');
        area.innerHTML = '<div style="color:var(--text-dim); padding:0.5rem;">Loading options...</div>';

        try {
            const settings = await API.getSettings();
            const opts = await API.getOptions(symbol, settings.min_dte || 21, settings.max_dte || 45, settings.top_n_options || 3);

            if (opts.error) {
                area.innerHTML = `<div style="color:var(--red);">${opts.error}</div>`;
                return;
            }

            if (!opts.recommendations || !opts.recommendations.length) {
                area.innerHTML = '<div style="color:var(--text-dim); font-size:0.7rem;">No options recommendations.</div>';
                return;
            }

            let html = '';
            opts.recommendations.forEach(r => {
                html += `
                <div class="opt-card">
                    <span class="opt-symbol">${r.contractSymbol || '?'}</span>
                    <span class="opt-detail">$${(r.strike || 0).toFixed(0)} strike</span>
                    <span class="opt-detail">${r.dte || '?'}d</span>
                    <span class="opt-detail">$${(r.mid || 0).toFixed(2)} mid</span>
                    <span class="opt-detail" style="color:var(--accent);">d=${(r.delta || 0).toFixed(2)}</span>
                    <span class="opt-detail">IV ${((r.iv || 0) * 100).toFixed(0)}%</span>
                </div>`;
            });

            area.innerHTML = html;
        } catch (err) {
            area.innerHTML = `<div style="color:var(--red);">Options error: ${err.message}</div>`;
        }
    },
};
