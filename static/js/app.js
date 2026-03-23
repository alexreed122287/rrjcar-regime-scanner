/**
 * app.js — Main controller
 * Homepage: Logo + $ → clicking $ opens sidebar with scanner
 */

const App = {
    scanResults: [],
    allScanned: 0,
    scanning: false,

    async init() {
        await Settings.init();
        this.bindEvents();
    },

    bindEvents() {
        // $ opens sidebar
        document.getElementById('home-dollar').onclick = () => this.openSidebar();
        document.getElementById('sidebar-close').onclick = () => this.closeSidebar();
        document.getElementById('sidebar-overlay').onclick = () => this.closeSidebar();

        // Scan
        document.getElementById('btn-scan').onclick = () => this.runScan();

        // Tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.onclick = () => this.showTab(btn.dataset.tab);
        });
    },

    openSidebar() {
        document.getElementById('sidebar').classList.add('open');
        document.getElementById('sidebar-overlay').classList.remove('hidden');
    },

    closeSidebar() {
        document.getElementById('sidebar').classList.remove('open');
        document.getElementById('sidebar-overlay').classList.add('hidden');
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

        const progress = document.getElementById('sidebar-progress');
        const progressFill = document.getElementById('batch-fill');
        const progressText = document.getElementById('batch-text');
        const metrics = document.getElementById('sidebar-metrics');
        const tabs = document.getElementById('sidebar-tabs');

        progress.classList.remove('hidden');
        metrics.classList.remove('hidden');
        tabs.classList.remove('hidden');
        progressFill.style.width = '0%';

        document.getElementById('btn-scan').disabled = true;
        document.getElementById('metric-scanned').textContent = '0';
        document.getElementById('metric-entries').textContent = '0';
        document.getElementById('scan-time').textContent = '--';
        this.showTab('screener');

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

                            // Only keep bullish ENTER
                            if ((r.signal || '') === 'LONG -- ENTER') {
                                this.scanResults.push(r);
                            }

                            const pct = (msg.progress.done / msg.progress.total * 100).toFixed(0);
                            progressFill.style.width = `${pct}%`;
                            progressText.textContent = `${msg.progress.done}/${msg.progress.total} | ${this.scanResults.length} hits`;
                            document.getElementById('metric-scanned').textContent = msg.progress.done;
                            document.getElementById('metric-entries').textContent = this.scanResults.length;
                            Screener.render(this.scanResults, document.getElementById('screener-content'));

                        } else if (msg.type === 'done') {
                            document.getElementById('scan-time').textContent = `${msg.summary.elapsed}s`;
                            progressText.textContent = `Done | ${this.allScanned} scanned | ${this.scanResults.length} entries | ${msg.summary.elapsed}s`;
                        }
                    } catch (_) {}
                }
            }

            Screener.render(this.scanResults, document.getElementById('screener-content'));
            if (this.scanResults.length) {
                Charts.regimeHeatmap('heatmap-chart', this.scanResults);
            }

        } catch (err) {
            document.getElementById('screener-content').innerHTML =
                `<div style="color:var(--red); padding:0.5rem;">${err.message}</div>`;
        } finally {
            document.getElementById('btn-scan').disabled = false;
            this.scanning = false;
        }
    },

    async drillDown(symbol) {
        this.showTab('drilldown');
        await DrillDown.render(symbol, document.getElementById('drilldown-content'));
    },
};

document.addEventListener('DOMContentLoaded', () => App.init());
