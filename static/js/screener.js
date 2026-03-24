/**
 * screener.js — Screener table rendering
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

        const list = document.createElement('div');
        list.className = 'screener-list';

        valid.forEach(r => {
            const row = document.createElement('div');
            row.className = 'screener-row';
            row.onclick = () => App.drillDown(r.symbol);

            const priceStr = r.price ? `$${r.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : '--';
            const chg = r.change_1d;
            const chgHex = chg != null ? (chg >= 0 ? '#34d399' : '#f87171') : '#6b7280';
            const chgStr = chg != null ? `<span style="color:${chgHex}">${chg >= 0 ? '+' : ''}${chg.toFixed(1)}%</span>` : '';

            const regimeHtml = r.regime_id != null ? this.regimeBadge(r.regime_id, r.regime_label, r.regime_confidence) : '';

            const { shortSig, sigHex } = this.signalInfo(r);
            const flash = (r.signal || '').includes('ENTER') || (r.signal || '').includes('EXIT') ? 'alert-flash' : '';

            const cmet = r.confirmations_met || 0;
            const cTotal = r.confirmations_total || 12;
            const ctRatio = cmet / Math.max(cTotal, 1);
            const ctHex = ctRatio >= 0.6 ? '#34d399' : (ctRatio >= 0.4 ? '#5eead4' : '#f87171');

            let rsiHtml = '';
            if (r.rsi != null) {
                const rsiHex = r.rsi > 70 ? '#f87171' : (r.rsi < 30 ? '#34d399' : '#9ca3af');
                rsiHtml = `<span style="color:${rsiHex}; font-size:0.8rem">RSI ${Math.round(r.rsi)}</span>`;
            }

            let adxHtml = '';
            if (r.adx != null) {
                const adxHex = r.adx > 25 ? '#34d399' : '#6b7280';
                adxHtml = `<span style="color:${adxHex}; font-size:0.8rem">ADX ${Math.round(r.adx)}</span>`;
            }

            row.innerHTML = `
                <span class="sr-symbol">${r.symbol}</span>
                <span class="sr-price">${priceStr}</span>
                ${chgStr}
                ${regimeHtml}
                <span class="sr-sig ${flash}" style="color:${sigHex}">${shortSig}</span>
                <span class="sr-conf" style="color:${ctHex}">${cmet}/${cTotal}</span>
                ${rsiHtml}
                ${adxHtml}
            `;
            list.appendChild(row);
        });

        container.appendChild(list);

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
            case 'rsi':
                return [...results].sort((a, b) => (a.rsi || 100) - (b.rsi || 100));
            case 'change':
                return [...results].sort((a, b) => (b.change_1d || 0) - (a.change_1d || 0));
            default:
                return results;
        }
    },
};
