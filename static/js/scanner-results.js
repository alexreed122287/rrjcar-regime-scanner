/**
 * scanner-results.js — Scanner Results tab
 * Shows signal hit counters, confirmation hit rates, and VIX level.
 */

const ScannerResults = {
    signalCounts: {},
    confirmationCounts: {},
    vix: null,
    totalScanned: 0,
    elapsed: 0,

    async fetchVix() {
        try {
            const res = await fetch('/api/vix');
            const data = await res.json();
            this.vix = data;
        } catch (_) {
            this.vix = { vix: null };
        }
    },

    update(summary) {
        this.signalCounts = summary.signal_counts || {};
        this.confirmationCounts = summary.confirmation_counts || {};
        this.totalScanned = summary.total || 0;
        this.elapsed = summary.elapsed || 0;
        this.render();
    },

    render() {
        const container = document.getElementById('results-content');
        if (!container) return;

        let html = '';

        // VIX Card
        html += this.renderVix();

        // Signal counts
        html += this.renderSignalCounts();

        // Confirmation hit rates
        html += this.renderConfirmationCounts();

        container.innerHTML = html;
    },

    renderVix() {
        const v = this.vix;
        if (!v || v.vix == null) {
            return `<div class="results-card">
                <div class="results-card-title">VIX</div>
                <div class="results-vix-value" style="color:var(--text-dim);">--</div>
            </div>`;
        }
        const level = v.vix;
        let color = '#34d399'; // green = low fear
        let label = 'LOW FEAR';
        if (level >= 35) { color = '#f87171'; label = 'HALT ZONE'; }
        else if (level >= 25) { color = '#fbbf24'; label = 'CAUTION'; }
        else if (level >= 20) { color = '#eab308'; label = 'ELEVATED'; }

        const chgColor = v.change >= 0 ? '#f87171' : '#34d399';
        const chgSign = v.change >= 0 ? '+' : '';

        return `<div class="results-card">
            <div class="results-card-title">VIX <span style="color:${color}; font-size:0.55rem; margin-left:0.3rem;">${label}</span></div>
            <div class="results-vix-row">
                <span class="results-vix-value" style="color:${color}">${level.toFixed(2)}</span>
                <span class="results-vix-change" style="color:${chgColor}">${chgSign}${v.change} (${chgSign}${v.change_pct}%)</span>
            </div>
            ${level >= 35 ? '<div class="results-vix-warn">VIX > 35 — All entries halted</div>' : ''}
            ${level >= 25 && level < 35 ? '<div class="results-vix-caution">VIX > 25 — Requires 8+ confirmations</div>' : ''}
        </div>`;
    },

    renderSignalCounts() {
        const counts = this.signalCounts;
        if (!Object.keys(counts).length) return '';

        const order = [
            'LONG -- ENTER', 'LONG -- CONFIRMING', 'LONG -- HOLD',
            'EXIT -- REGIME FLIP', 'CASH -- NEUTRAL', 'CASH -- BEARISH', 'ERROR'
        ];
        const colors = {
            'LONG -- ENTER': '#34d399',
            'LONG -- CONFIRMING': '#5eead4',
            'LONG -- HOLD': '#2dd4bf',
            'EXIT -- REGIME FLIP': '#f87171',
            'CASH -- NEUTRAL': '#94a3b8',
            'CASH -- BEARISH': '#f87171',
            'ERROR': '#666',
        };
        const labels = {
            'LONG -- ENTER': 'ENTER',
            'LONG -- CONFIRMING': 'CONFIRMING',
            'LONG -- HOLD': 'HOLD',
            'EXIT -- REGIME FLIP': 'EXIT',
            'CASH -- NEUTRAL': 'NEUTRAL',
            'CASH -- BEARISH': 'BEARISH',
            'ERROR': 'ERROR',
        };

        let rows = '';
        for (const sig of order) {
            const count = counts[sig] || 0;
            if (count === 0) continue;
            const pct = this.totalScanned > 0 ? (count / this.totalScanned * 100).toFixed(1) : '0';
            const color = colors[sig] || '#666';
            const label = labels[sig] || sig;
            const barWidth = this.totalScanned > 0 ? Math.max(2, count / this.totalScanned * 100) : 0;
            rows += `<div class="results-row">
                <span class="results-row-label" style="color:${color}">${label}</span>
                <div class="results-row-bar-bg"><div class="results-row-bar" style="width:${barWidth}%; background:${color};"></div></div>
                <span class="results-row-count" style="color:${color}">${count}</span>
                <span class="results-row-pct">${pct}%</span>
            </div>`;
        }

        return `<div class="results-card">
            <div class="results-card-title">Signal Hits <span class="results-card-sub">${this.totalScanned} scanned | ${this.elapsed}s</span></div>
            ${rows}
        </div>`;
    },

    renderConfirmationCounts() {
        const counts = this.confirmationCounts;
        if (!Object.keys(counts).length) return '';

        // Sort by pass rate descending
        const entries = Object.entries(counts).sort((a, b) => {
            const rateA = a[1].pass / Math.max(a[1].pass + a[1].fail, 1);
            const rateB = b[1].pass / Math.max(b[1].pass + b[1].fail, 1);
            return rateB - rateA;
        });

        let rows = '';
        for (const [name, data] of entries) {
            const total = data.pass + data.fail;
            const rate = total > 0 ? (data.pass / total * 100).toFixed(1) : '0';
            const rateNum = parseFloat(rate);
            let color = '#34d399';
            if (rateNum < 30) color = '#f87171';
            else if (rateNum < 50) color = '#fbbf24';
            else if (rateNum < 70) color = '#eab308';

            rows += `<div class="results-row">
                <span class="results-row-label">${name}</span>
                <div class="results-row-bar-bg"><div class="results-row-bar" style="width:${rate}%; background:${color};"></div></div>
                <span class="results-row-count" style="color:${color}">${data.pass}/${total}</span>
                <span class="results-row-pct" style="color:${color}">${rate}%</span>
            </div>`;
        }

        return `<div class="results-card">
            <div class="results-card-title">Confirmation Hit Rates</div>
            ${rows}
        </div>`;
    },
};
