/**
 * drilldown.js — Single-ticker deep analysis view
 */

const DrillDown = {
    async render(symbol, container) {
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
                <span class="back-link" onclick="App.showTab('screener')">&larr; Back to Screener</span>
                <h2 style="font-size:1.1rem; color:var(--text-primary); margin:0.3rem 0;">${data.symbol}</h2>

                <div class="signal-banner ${sigCss}">
                    ${data.signal || 'N/A'}
                    <div style="font-size:0.8rem; font-weight:400; margin-top:3px; opacity:0.85;">
                        ${data.action || ''}
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card"><div class="label">Price</div><div class="value">$${(data.price || 0).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</div></div>
                    <div class="metric-card"><div class="label">Regime</div><div class="value ${regCss}">${data.regime_label || '?'}</div></div>
                    <div class="metric-card"><div class="label">Confidence</div><div class="value">${data.regime_confidence ? Math.round(data.regime_confidence * 100) + '%' : '?'}</div></div>
                    <div class="metric-card"><div class="label">Confirmations</div><div class="value">${data.confirmations_met || 0}/${data.confirmations_total || 12}</div></div>
                    <div class="metric-card"><div class="label">Streak</div><div class="value">${data.regime_streak || '?'} bars</div></div>
                    <div class="metric-card"><div class="label">1D Change</div><div class="value ${chgCss}">${chgStr}</div></div>
                </div>

                ${this.renderConfirmations(data.confirmation_detail)}

                <div class="chart-container" id="dd-price-chart"></div>

                <div style="margin-top:0.5rem;">
                    <button class="btn btn-sm" onclick="DrillDown.loadBacktest('${data.symbol}')">Run Backtest</button>
                    <button class="btn btn-sm" onclick="DrillDown.loadOptions('${data.symbol}')">Show Options</button>
                </div>

                <div id="dd-backtest-area"></div>
                <div id="dd-options-area"></div>
            `;

            // Price chart
            if (data.chart_data) {
                Charts.priceWithRegimes('dd-price-chart', data.chart_data, `${data.symbol} Regime Analysis`);
            }

        } catch (err) {
            container.innerHTML = `<div style="color:var(--red); padding:1rem;">Error: ${err.message}</div>`;
        }
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

    async loadBacktest(symbol) {
        const area = document.getElementById('dd-backtest-area');
        area.innerHTML = '<div style="color:var(--text-dim); padding:0.5rem;">Running backtest...</div>';

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
                area.innerHTML = `<div style="color:var(--red);">${bt.error}</div>`;
                return;
            }

            const m = bt.metrics;
            area.innerHTML = `
                <h3 style="font-size:0.9rem; color:var(--text-primary); margin:0.5rem 0 0.3rem;">Backtest Results</h3>
                <div class="metrics-grid" style="grid-template-columns: repeat(4, 1fr);">
                    <div class="metric-card"><div class="label">Total Return</div><div class="value ${(m.total_return_pct||0) >= 0 ? 'bull' : 'bear'}">${(m.total_return_pct||0).toFixed(1)}%</div></div>
                    <div class="metric-card"><div class="label">Win Rate</div><div class="value">${(m.win_rate||0).toFixed(0)}%</div></div>
                    <div class="metric-card"><div class="label">Sharpe</div><div class="value">${(m.sharpe_ratio||0).toFixed(2)}</div></div>
                    <div class="metric-card"><div class="label">Max DD</div><div class="value bear">${(m.max_drawdown_pct||0).toFixed(1)}%</div></div>
                </div>
                <div class="chart-container" id="dd-equity-chart"></div>
                ${this.renderTradeTable(bt.trades)}
            `;

            Charts.equityCurve('dd-equity-chart', bt.equity_curve);

        } catch (err) {
            area.innerHTML = `<div style="color:var(--red);">Backtest error: ${err.message}</div>`;
        }
    },

    renderTradeTable(trades) {
        if (!trades || !trades.length) return '<div style="color:var(--text-dim); font-size:0.8rem;">No trades.</div>';
        let html = `<table class="trade-table"><thead><tr>
            <th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>P&L %</th><th>Reason</th>
        </tr></thead><tbody>`;

        trades.forEach(t => {
            const pnl = t.pnl_pct || 0;
            const pnlCss = pnl >= 0 ? 'color:#34d399' : 'color:#f87171';
            html += `<tr>
                <td>${(t.entry_date || '').substring(0, 10)}</td>
                <td>${(t.exit_date || '').substring(0, 10)}</td>
                <td>$${(t.entry_price || 0).toFixed(2)}</td>
                <td>$${(t.exit_price || 0).toFixed(2)}</td>
                <td style="${pnlCss}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}%</td>
                <td style="color:var(--text-dim); max-width:150px; overflow:hidden; text-overflow:ellipsis;">${t.exit_reason || ''}</td>
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
                area.innerHTML = '<div style="color:var(--text-dim); font-size:0.8rem;">No options recommendations.</div>';
                return;
            }

            let html = '<h3 style="font-size:0.9rem; color:var(--text-primary); margin:0.5rem 0 0.3rem;">Options Picks</h3>';
            opts.recommendations.forEach(r => {
                html += `
                <div class="opt-card">
                    <span class="opt-symbol">${r.contractSymbol || '?'}</span>
                    <span class="opt-detail">$${(r.strike || 0).toFixed(0)} strike</span>
                    <span class="opt-detail">${r.dte || '?'}d</span>
                    <span class="opt-detail">$${(r.mid || 0).toFixed(2)} mid</span>
                    <span class="opt-detail" style="color:var(--accent);">d=${(r.delta || 0).toFixed(2)}</span>
                    <span class="opt-detail">IV ${((r.iv || 0) * 100).toFixed(0)}%</span>
                    <span class="opt-detail">Score: ${(r.score || 0).toFixed(0)}</span>
                </div>`;
            });

            area.innerHTML = html;

        } catch (err) {
            area.innerHTML = `<div style="color:var(--red);">Options error: ${err.message}</div>`;
        }
    },
};
