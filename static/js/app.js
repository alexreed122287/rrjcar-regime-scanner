/**
 * app.js — Main application controller
 */

const App = {
    scanResults: [],
    scanning: false,

    async init() {
        await Settings.init();
        this.bindEvents();
        this.showTab('screener');

        // Try loading cached results
        try {
            const cached = await API.getCached();
            if (cached.results && cached.results.length) {
                this.scanResults = cached.results;
                this.renderResults(cached);
            }
        } catch (_) {}
    },

    bindEvents() {
        // Settings toggle
        const toggle = document.getElementById('settings-toggle');
        const body = document.getElementById('settings-body');
        if (toggle && body) {
            toggle.onclick = () => body.classList.toggle('open');
        }

        // Scan button
        document.getElementById('btn-scan').onclick = () => this.runScan();

        // Filter & sort
        document.getElementById('filter-signal').onchange = (e) => {
            Screener.currentFilter = e.target.value;
            Screener.render(this.scanResults, document.getElementById('screener-content'));
        };
        document.getElementById('sort-by').onchange = (e) => {
            Screener.currentSort = e.target.value;
            Screener.render(this.scanResults, document.getElementById('screener-content'));
        };

        // Tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.onclick = () => this.showTab(btn.dataset.tab);
        });
    },

    showTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === `tab-${tabName}`));
    },

    async runScan() {
        if (this.scanning) return;
        this.scanning = true;
        this.scanResults = [];

        const loadingText = document.getElementById('loading-text');
        const overlay = document.getElementById('loading-overlay');
        overlay.classList.remove('hidden');
        loadingText.textContent = 'Scanning...';
        document.getElementById('btn-scan').disabled = true;

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
                bullish_only: false,
            };

            // Use streaming endpoint — results appear one by one
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
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const msg = JSON.parse(line.slice(6));
                        if (msg.type === 'result') {
                            this.scanResults.push(msg.data);
                            // Live update
                            loadingText.textContent = `Scanning... ${msg.progress.done}/${msg.progress.total}`;
                            this.setMetric('metric-scanned', msg.progress.done);
                            Screener.render(this.scanResults, document.getElementById('screener-content'));
                        } else if (msg.type === 'done') {
                            this.renderResults({ summary: msg.summary });
                        }
                    } catch (_) {}
                }
            }

        } catch (err) {
            console.error('Scan error:', err);
            document.getElementById('screener-content').innerHTML =
                `<div style="color:var(--red); padding:1rem;">${err.message}</div>`;
        } finally {
            overlay.classList.add('hidden');
            document.getElementById('btn-scan').disabled = false;
            this.scanning = false;
        }
    },

    renderResults(result) {
        const summary = result.summary || {};

        // Update metrics
        this.setMetric('metric-scanned', summary.total || this.scanResults.length);
        this.setMetric('metric-bullish', summary.bullish || 0, 'bull');
        this.setMetric('metric-bearish', summary.bearish || 0, 'bear');
        this.setMetric('metric-neutral', summary.neutral || 0, 'neutral');
        this.setMetric('metric-entries', summary.entries || 0, 'bull');
        this.setMetric('metric-exits', summary.exits || 0, 'bear');

        if (summary.elapsed) {
            document.getElementById('scan-time').textContent = `${summary.elapsed}s`;
        }

        // Render screener
        Screener.render(this.scanResults, document.getElementById('screener-content'));

        // Render charts
        if (this.scanResults.length) {
            Charts.regimeHeatmap('heatmap-chart', this.scanResults);
            Charts.signalDistribution('signal-dist-chart', this.scanResults);
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
