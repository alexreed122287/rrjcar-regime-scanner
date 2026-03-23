/**
 * app.js — Main application controller
 * Clean homepage -> $ dropdown -> scan -> bullish hits only
 */

const App = {
    scanResults: [],    // Only bullish ENTER hits
    allScanned: 0,      // Total tickers scanned
    scanning: false,

    async init() {
        await Settings.init();
        this.bindEvents();
    },

    bindEvents() {
        // $ toggle opens settings drawer
        const dollar = document.getElementById('hero-dollar');
        const drawer = document.getElementById('settings-drawer');
        dollar.onclick = () => {
            dollar.classList.toggle('open');
            drawer.classList.toggle('open');
        };

        // Scan button
        document.getElementById('btn-scan').onclick = () => this.runScan();

        // Tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.onclick = () => this.showTab(btn.dataset.tab);
        });

        // Back to home
        document.getElementById('back-home').onclick = () => this.goHome();
    },

    goHome() {
        document.getElementById('scanner-results').classList.add('hidden');
        document.getElementById('hero-section').classList.remove('compact');
        document.getElementById('settings-drawer').classList.remove('open');
        document.getElementById('hero-dollar').classList.remove('open');
    },

    showTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === `tab-${tabName}`));
    },

    async runScan() {
        if (this.scanning) return;
        this.scanning = true;
        this.scanResults = [];
        this.allScanned = 0;

        // Collapse hero, show results area
        document.getElementById('hero-section').classList.add('compact');
        document.getElementById('settings-drawer').classList.remove('open');
        document.getElementById('hero-dollar').classList.remove('open');

        const resultsArea = document.getElementById('scanner-results');
        resultsArea.classList.remove('hidden');

        const progress = document.getElementById('batch-progress');
        const progressFill = document.getElementById('batch-fill');
        const progressText = document.getElementById('batch-text');
        progress.classList.remove('hidden');

        document.getElementById('btn-scan').disabled = true;
        this.showTab('screener');

        // Reset metrics
        this.setMetric('metric-scanned', '0');
        this.setMetric('metric-bullish', '0', 'bull');
        this.setMetric('metric-entries', '0', 'bull');
        document.getElementById('scan-time').textContent = '--';

        try {
            await Settings.save();
            const params = Settings.gather();

            const body = {
                watchlist: params.watchlist,
                custom_tickers: params.custom_tickers,
                strategy: params.strategy,
                min_confs: params.min_confs,
                regime_confirm: params.regime_confirm,
                max_workers: params.max_workers,
                bullish_only: false,  // we filter client-side for running list
            };

            // Use streaming endpoint
            const res = await fetch('/api/scan/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const msg = JSON.parse(line.slice(6));
                        if (msg.type === 'result') {
                            this.allScanned = msg.progress.done;
                            const r = msg.data;

                            // Only keep bullish ENTER hits
                            const sig = r.signal || '';
                            if (sig === 'LONG -- ENTER') {
                                this.scanResults.push(r);
                            }

                            // Update progress
                            const pct = (msg.progress.done / msg.progress.total * 100).toFixed(0);
                            progressFill.style.width = `${pct}%`;
                            progressText.textContent = `${msg.progress.done} / ${msg.progress.total} scanned | ${this.scanResults.length} bullish hits`;

                            // Live update metrics
                            this.setMetric('metric-scanned', msg.progress.done);
                            this.setMetric('metric-bullish', this.scanResults.length, 'bull');
                            this.setMetric('metric-entries', this.scanResults.length, 'bull');

                            // Re-render hits table progressively
                            Screener.render(this.scanResults, document.getElementById('screener-content'));

                        } else if (msg.type === 'done') {
                            const s = msg.summary;
                            document.getElementById('scan-time').textContent = `${s.elapsed}s`;
                            progressText.textContent = `Done! ${this.allScanned} scanned | ${this.scanResults.length} bullish entries found | ${s.elapsed}s`;
                        }
                    } catch (_) {}
                }
            }

            // Final render
            Screener.render(this.scanResults, document.getElementById('screener-content'));
            if (this.scanResults.length) {
                Charts.regimeHeatmap('heatmap-chart', this.scanResults);
                Charts.signalDistribution('signal-dist-chart', this.scanResults);
            }

        } catch (err) {
            console.error('Scan error:', err);
            document.getElementById('screener-content').innerHTML =
                `<div style="color:var(--red); padding:1rem;">${err.message}</div>`;
        } finally {
            document.getElementById('btn-scan').disabled = false;
            this.scanning = false;
        }
    },

    setMetric(id, value, cls = '') {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value;
            el.className = 'value ' + cls;
        }
    },

    async drillDown(symbol) {
        this.showTab('drilldown');
        const container = document.getElementById('drilldown-content');
        await DrillDown.render(symbol, container);
    },
};

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
