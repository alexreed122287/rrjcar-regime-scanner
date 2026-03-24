/**
 * screener.js — Screener table rendering
 * Clean column layout with headers, no RSI/ADX
 */

const Screener = {
    currentFilter: 'All',
    currentSort: 'signal',
    minConfidence: 0,
    results: [],

    render(results, container) {
        this.results = results;
        container.innerHTML = '';

        let filtered = this.applyFilter(results);
        filtered = this.applyConfidenceFilter(filtered);
        filtered = this.applySort(filtered);

        const errored = filtered.filter(r => r.error && !r.price);
        const valid = filtered.filter(r => !(r.error && !r.price));

        if (!valid.length) {
            container.innerHTML = '<div style="color: var(--text-dim); padding: 1rem; text-align: center;">No tickers match the current filter.</div>';
            return;
        }

        const table = document.createElement('div');
        table.className = 'screener-table';

        // Column headers
        const header = document.createElement('div');
        header.className = 'screener-header';
        header.innerHTML = `
            <span class="sh-symbol">Symbol</span>
            <span class="sh-price">Price</span>
            <span class="sh-change">Chg</span>
            <span class="sh-regime">Regime</span>
            <span class="sh-signal">Signal</span>
            <span class="sh-conf">Confs</span>
        `;
        table.appendChild(header);

        const list = document.createElement('div');
        list.className = 'screener-list';

        valid.forEach(r => {
            const row = document.createElement('div');
            row.className = 'screener-row';
            row.onclick = () => App.drillDown(r.symbol);

            const priceStr = r.price ? `$${r.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '--';
            const chg = r.change_1d;
            const chgHex = chg != null && chg !== 0 ? (chg >= 0 ? '#34d399' : '#f87171') : '#6b7280';
            const chgStr = chg != null && chg !== 0 ? `${chg >= 0 ? '+' : ''}${chg.toFixed(1)}%` : '--';

            const regimeHtml = r.regime_id != null ? this.regimeBadge(r.regime_id, r.regime_label, r.regime_confidence) : '<span class="sr-regime">--</span>';

            const { shortSig, sigHex } = this.signalInfo(r);
            const flash = (r.signal || '').includes('ENTER') || (r.signal || '').includes('EXIT') ? 'alert-flash' : '';

            const cmet = r.confirmations_met || 0;
            const cTotal = r.confirmations_total || 12;
            const ctRatio = cmet / Math.max(cTotal, 1);
            const ctHex = ctRatio >= 0.6 ? '#34d399' : (ctRatio >= 0.4 ? '#5eead4' : '#f87171');

            row.innerHTML = `
                <span class="sr-symbol">${r.symbol}</span>
                <span class="sr-price">${priceStr}</span>
                <span class="sr-change" style="color:${chgHex}">${chgStr}</span>
                ${regimeHtml}
                <span class="sr-sig ${flash}" style="color:${sigHex}">${shortSig}</span>
                <span class="sr-conf" style="color:${ctHex}">${cmet}/${cTotal}</span>
            `;
            list.appendChild(row);
        });

        table.appendChild(list);
        container.appendChild(table);

        // Errors
        if (errored.length) {
            const details = document.createElement('details');
            details.className = 'errors-panel';
            details.innerHTML = `<summary>${errored.length} tickers failed to scan</summary>` +
                errored.map(r => `<div class="error-item">${r.symbol}: ${(r.error || 'unknown').substring(0, 60)}</div>`).join('');
            container.appendChild(details);
        }
    },

    regimeBadge(rid, label, confidence) {
        const color = REGIME_COLORS[rid] || '#666';
        const bg = REGIME_BG_COLORS[rid] || '#111';
        const conf = confidence ? ` (${Math.round(confidence * 100)}%)` : '';
        return `<span class="regime-badge" style="background:${bg}; color:${color}; border:1px solid ${color};">${label}${conf}</span>`;
    },

    signalInfo(r) {
        const sig = r.signal || '';
        const shortSig = sig.replace('LONG -- ', '').replace('CASH -- ', '').replace('EXIT -- ', 'EXIT: ');
        const colors = { ENTER: '#34d399', CONFIRMING: '#5eead4', HOLD: '#2dd4bf', EXIT: '#f87171', BEARISH: '#f87171' };
        let sigHex = '#6b7280';
        for (const [k, v] of Object.entries(colors)) {
            if (sig.includes(k)) { sigHex = v; break; }
        }
        return { shortSig, sigHex };
    },

    applyFilter(results) {
        if (this.currentFilter === 'All') return results;
        return results.filter(r => (r.signal || '') === this.currentFilter);
    },

    applyConfidenceFilter(results) {
        if (this.minConfidence <= 0) return results;
        return results.filter(r => (r.regime_confidence || 0) >= this.minConfidence);
    },

    applySort(results) {
        const PRIORITY = {
            'LONG -- ENTER': 0, 'EXIT -- REGIME FLIP': 1, 'LONG -- CONFIRMING': 2,
            'LONG -- HOLD': 3, 'CASH -- NEUTRAL': 4, 'CASH -- BEARISH': 5, 'ERROR': 99,
        };

        switch (this.currentSort) {
            case 'signal':
                return [...results].sort((a, b) => {
                    const pa = PRIORITY[a.signal] ?? 50;
                    const pb = PRIORITY[b.signal] ?? 50;
                    return pa - pb || (b.confirmations_met || 0) - (a.confirmations_met || 0);
                });
            case 'confidence':
                return [...results].sort((a, b) => (b.regime_confidence || 0) - (a.regime_confidence || 0));
            case 'confirmations':
                return [...results].sort((a, b) => (b.confirmations_met || 0) - (a.confirmations_met || 0));
            case 'change':
                return [...results].sort((a, b) => (b.change_1d || 0) - (a.change_1d || 0));
            default:
                return results;
        }
    },
};
