/**
 * leaps.js — LEAPS Screener Module
 * Renders LEAPS scan hits and loads best LEAPS contracts for drill-down.
 */

const Leaps = {
    results: [],

    render(results, container) {
        this.results = results || [];

        // Filter to LEAPS signals only
        const leapsHits = this.results.filter(r => {
            const sig = r.signal || '';
            return sig.startsWith('LEAPS --') && (sig.includes('BUY') || sig.includes('WATCH') || sig.includes('HOLD'));
        });

        if (!leapsHits.length) {
            container.innerHTML = '<div style="color:var(--text-dim); padding:1rem; text-align:center; font-size:0.75rem;">No LEAPS signals found. Run a scan with LEAPS strategy selected.</div>';
            return;
        }

        // Sort by confirmations descending
        leapsHits.sort((a, b) => (b.confirmations_met || 0) - (a.confirmations_met || 0));

        let html = `
            <div style="font-size:0.6rem; font-family:var(--mono); color:var(--chrome); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem; padding:0 0.3rem;">
                LEAPS Candidates — ${leapsHits.length} hits (click for contracts)
            </div>
            <table class="screener-table" style="width:100%; border-collapse:collapse; font-size:0.7rem;">
            <thead><tr style="color:var(--text-dim); font-size:0.6rem; text-transform:uppercase;">
                <th style="text-align:left; padding:4px 6px;">Ticker</th>
                <th style="text-align:left;">Price</th>
                <th style="text-align:left;">52W</th>
                <th style="text-align:left;">Regime</th>
                <th style="text-align:left;">IV Rank</th>
                <th style="text-align:left;">Confs</th>
                <th style="text-align:left;">Signal</th>
            </tr></thead><tbody>`;

        leapsHits.forEach(r => {
            const sym = r.symbol || '?';
            const price = r.price || 0;
            const regime = r.regime_label || '?';
            const conf = r.regime_confidence || 0;
            const confs = r.confirmations_met || 0;
            const total = r.confirmations_total || 10;
            const signal = (r.signal || '').replace('LEAPS -- ', '');
            const hvRank = r.hv_rank != null ? Math.round(r.hv_rank * 100) + '%' : '?';
            const pct52w = r.pct_52w != null ? (r.pct_52w >= 0 ? '+' : '') + r.pct_52w.toFixed(1) + '%' : '--';
            const pct52wColor = r.pct_52w >= 0 ? '#2dd4bf' : '#f87171';
            const name = r.name || '';
            const sector = r.sector || '';

            const sigColor = signal === 'BUY' ? '#22c55e' : (signal === 'WATCH' ? '#eab308' : '#60a5fa');
            const confColor = confs >= 8 ? '#22c55e' : (confs >= 6 ? '#eab308' : '#f87171');
            const regCss = (r.regime_id || 3) <= 1 ? 'color:#2dd4bf' : ((r.regime_id || 3) >= 5 ? 'color:#f87171' : 'color:#9ca3af');

            html += `
            <tr style="border-bottom:1px solid #1e2028; cursor:pointer;" onclick="Leaps.drillDown('${sym}')">
                <td style="padding:5px 6px;">
                    <b>${sym}</b>
                    <div style="font-size:0.55rem; color:#6b7280;">${name}${sector ? ' · ' + sector : ''}</div>
                </td>
                <td>$${price.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}</td>
                <td style="color:${pct52wColor}">${pct52w}</td>
                <td style="${regCss}">${regime}<br><span style="font-size:0.55rem; opacity:0.7;">${Math.round(conf * 100)}%</span></td>
                <td>${hvRank}</td>
                <td style="color:${confColor}">${confs}/${total}</td>
                <td style="color:${sigColor}; font-weight:600;">${signal}</td>
            </tr>`;
        });

        html += '</tbody></table>';
        container.innerHTML = html;
    },

    async drillDown(symbol) {
        App.showTab('leaps');
        const container = document.getElementById('leaps-content');
        container.innerHTML = '<div style="text-align:center; padding:2rem; color:var(--text-dim);">Loading LEAPS contracts...</div>';

        try {
            const [leapsData, settings] = await Promise.all([
                API.getLeaps(symbol),
                API.getSettings(),
            ]);

            if (leapsData.error) {
                container.innerHTML = `<div style="color:var(--red); padding:1rem;">${leapsData.error}</div>`;
                return;
            }

            const spot = leapsData.spot_price || 0;
            const recs = leapsData.recommendations || [];

            let html = `
                <span class="back-link" onclick="Leaps.render(Leaps.results, document.getElementById('leaps-content'))">&larr; Back to LEAPS Hits</span>
                <h2 style="font-size:1.1rem; color:var(--chrome); margin:0.3rem 0;">${symbol}
                    <span style="font-size:0.8rem; font-weight:400; color:var(--text-dim);">$${spot.toFixed(2)} — LEAPS Analysis</span>
                </h2>

                <div style="font-size:0.6rem; font-family:var(--mono); color:var(--chrome); text-transform:uppercase; letter-spacing:1px; margin:0.5rem 0 0.3rem;">
                    Best LEAPS Contracts (${recs.length} of ${leapsData.total_evaluated || 0} evaluated)
                </div>
                <div style="font-size:0.55rem; color:var(--text-dim); margin-bottom:0.4rem;">
                    Scoring: Delta 0.65-0.80 (stock replacement) · DTE 270-540 · Low IV · Tight spreads · High OI
                </div>
            `;

            if (!recs.length) {
                html += '<div style="color:var(--text-dim); padding:0.5rem;">No qualifying LEAPS contracts found (check if options with 180+ DTE exist).</div>';
            } else {
                recs.forEach((r, i) => {
                    const isBest = i === 0;
                    const border = isBest ? 'border-left:3px solid var(--green);' : 'border-left:3px solid transparent;';
                    const label = isBest ? '<span style="color:var(--green); font-size:0.55rem; font-weight:700;">BEST LEAPS</span> ' : '';
                    const itm = r.inTheMoney ? '<span style="color:#60a5fa; font-size:0.5rem;">ITM</span>' : '<span style="color:#9ca3af; font-size:0.5rem;">OTM</span>';
                    const moneyStr = r.moneyness > 0 ? `${r.moneyness.toFixed(1)}% ITM` : `${Math.abs(r.moneyness).toFixed(1)}% OTM`;

                    html += `
                    <div style="padding:0.5rem; margin-bottom:0.3rem; ${border} background:rgba(255,255,255,0.02); border-radius:3px;">
                        <div style="display:flex; flex-wrap:wrap; gap:0.3rem 0.8rem; align-items:center; font-size:0.75rem;">
                            ${label}
                            <span style="color:var(--chrome); font-family:var(--mono); font-weight:600;">$${r.strike} C</span>
                            ${itm}
                            <span style="color:var(--text-dim);">${r.expiration} (${r.dte}d)</span>
                            <span style="color:var(--text-dim);">$${r.mid.toFixed(2)} mid</span>
                            <span style="color:#22c55e;">d=${r.delta.toFixed(2)}</span>
                            <span style="color:#eab308;">t=${r.theta.toFixed(3)}</span>
                            <span style="color:var(--text-dim);">IV ${r.iv_pct}%</span>
                        </div>
                        <div style="display:flex; flex-wrap:wrap; gap:0.3rem 0.8rem; font-size:0.6rem; color:var(--text-dim); margin-top:0.2rem; font-family:var(--mono);">
                            <span>${moneyStr}</span>
                            <span>Vol: ${(r.volume || 0).toLocaleString()}</span>
                            <span>OI: ${(r.openInterest || 0).toLocaleString()}</span>
                            <span>Bid: $${(r.bid || 0).toFixed(2)}</span>
                            <span>Ask: $${(r.ask || 0).toFixed(2)}</span>
                            <span>Spread: ${r.bid > 0 && r.ask > 0 ? ((r.ask - r.bid) / r.mid * 100).toFixed(1) + '%' : '?'}</span>
                            <span style="color:var(--chrome);">Score: ${r.score}</span>
                        </div>`;

                    if (r.contracts) {
                        html += `
                        <div style="font-size:0.6rem; color:#22c55e; margin-top:0.2rem; font-family:var(--mono);">
                            <b>${r.contracts} contracts</b> = $${(r.total_cost || 0).toLocaleString()} (${(r.pct_of_capital || 0).toFixed(1)}% of capital)
                        </div>`;
                    }

                    html += '</div>';
                });
            }

            container.innerHTML = html;

        } catch (err) {
            container.innerHTML = `<div style="color:var(--red); padding:1rem;">Error: ${err.message}</div>`;
        }
    },
};
