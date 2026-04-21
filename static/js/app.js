/**
 * app.js — Main controller
 * Homepage: Logo + $ → clicking $ opens sidebar with scanner
 * Includes ENTER + CONFIRMING hits (bullish signals)
 */

const BULLISH_SIGNALS = ['LONG -- ENTER', 'LONG -- CONFIRMING', 'LONG -- HOLD', 'LEAPS -- BUY', 'LEAPS -- WATCH', 'LEAPS -- HOLD', 'BOTTOM -- BUY', 'BOTTOM -- WATCH'];

const App = {
    scanResults: [],
    allScanned: 0,
    scanning: false,
    abortController: null,

    async init() {
        await Settings.init();
        Orders.init();
        this.bindEvents();
        this.loadConfig();
    },

    bindEvents() {
        document.getElementById('home-dollar').onclick = () => this.openSidebar();
        document.getElementById('sidebar-close').onclick = () => this.closeSidebar();
        document.getElementById('sidebar-overlay').onclick = () => this.closeSidebar();
        document.getElementById('btn-scan').onclick = () => this.runScan();
        document.getElementById('btn-stop').onclick = () => this.stopScan();
        document.getElementById('btn-save-config').onclick = () => this.saveConfig();
        document.getElementById('btn-connect-tradier').onclick = () => this.connectTradier();

        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.onclick = () => this.showTab(btn.dataset.tab);
        });

        // Live feed toggle
        document.getElementById('hit-tape-toggle').onclick = () => {
            const list = document.getElementById('hit-tape-list');
            const arrow = document.getElementById('feed-arrow');
            list.classList.toggle('hidden');
            arrow.innerHTML = list.classList.contains('hidden') ? '&#9654;' : '&#9660;';
        };

        // Section toggles
        document.querySelectorAll('.section-toggle').forEach(t => {
            t.onclick = () => {
                const body = t.nextElementSibling;
                body.classList.toggle('collapsed');
                t.classList.toggle('open');
            };
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

    // ── Config: Tradier, email, schedule, API keys ──
    loadConfig() {
        try {
            const cfg = JSON.parse(localStorage.getItem('rrjcar_config') || '{}');
            if (cfg.alert_email) document.getElementById('cfg-email').value = cfg.alert_email;
            if (cfg.scan_time) document.getElementById('cfg-scan-time').value = cfg.scan_time;
            if (cfg.scan_watchlist) document.getElementById('cfg-scan-watchlist').value = cfg.scan_watchlist;
            if (cfg.alpha_vantage_key) document.getElementById('cfg-av-key').value = cfg.alpha_vantage_key;
            if (cfg.fmp_key) document.getElementById('cfg-fmp-key').value = cfg.fmp_key;
            if (cfg.twelve_data_key) document.getElementById('cfg-td-key').value = cfg.twelve_data_key;
        } catch (_) {}

        // Load Tradier status from backend
        this.loadTradierStatus();
    },

    async loadTradierStatus() {
        const statusEl = document.getElementById('tradier-status');
        try {
            const res = await API.brokerStatus();
            if (res.configured && res.account_info && !res.account_info.error) {
                const info = res.account_info;
                const mode = info.sandbox ? 'SANDBOX' : 'PRODUCTION';
                statusEl.innerHTML = `<span style="color:var(--green);">Connected</span> | ${mode} | Acct: ${info.account_id} | Equity: $${(info.total_equity || 0).toLocaleString()}`;
                // Pre-fill fields
                document.getElementById('cfg-tradier-account').value = info.account_id || '';
                document.getElementById('cfg-tradier-mode').value = info.sandbox ? 'sandbox' : 'production';
                document.getElementById('cfg-tradier-token').placeholder = 'Token saved (enter new to change)';
            } else if (res.configured) {
                statusEl.innerHTML = `<span style="color:#eab308;">Configured but connection failed</span> — check token`;
            } else {
                statusEl.innerHTML = `<span style="color:var(--text-dim);">Not connected</span> — enter credentials below`;
            }
        } catch (_) {
            statusEl.textContent = 'Could not check status';
        }
    },

    async connectTradier() {
        const token = document.getElementById('cfg-tradier-token').value.trim();
        const account = document.getElementById('cfg-tradier-account').value.trim();
        const mode = document.getElementById('cfg-tradier-mode').value;

        if (!token || !account) {
            document.getElementById('tradier-status').innerHTML = '<span style="color:var(--red);">Enter both token and account ID</span>';
            return;
        }

        const btn = document.getElementById('btn-connect-tradier');
        btn.disabled = true;
        btn.textContent = 'Connecting...';

        try {
            const res = await API.brokerConnect({
                access_token: token,
                account_id: account,
                sandbox: mode === 'sandbox',
            });

            if (res.success) {
                btn.textContent = 'Connected';
                btn.style.background = '#22c55e';
                this.loadTradierStatus();
            } else {
                document.getElementById('tradier-status').innerHTML = `<span style="color:var(--red);">Failed: ${res.error || 'Unknown error'}</span>`;
                btn.textContent = 'Connect Tradier';
            }
        } catch (err) {
            document.getElementById('tradier-status').innerHTML = `<span style="color:var(--red);">Error: ${err.message}</span>`;
            btn.textContent = 'Connect Tradier';
        } finally {
            btn.disabled = false;
            setTimeout(() => { btn.textContent = 'Connect Tradier'; btn.style.background = ''; }, 2000);
        }
    },

    async saveConfig() {
        const cfg = {
            alert_email: document.getElementById('cfg-email').value.trim(),
            scan_time: document.getElementById('cfg-scan-time').value,
            scan_watchlist: document.getElementById('cfg-scan-watchlist').value,
            alpha_vantage_key: document.getElementById('cfg-av-key').value.trim(),
            fmp_key: document.getElementById('cfg-fmp-key').value.trim(),
            twelve_data_key: document.getElementById('cfg-td-key').value.trim(),
        };
        localStorage.setItem('rrjcar_config', JSON.stringify(cfg));

        // Save email + schedule to backend
        try {
            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    alert_email: cfg.alert_email,
                    alerts_enabled: !!cfg.alert_email,
                    alert_on_bull_entry: true,
                }),
            });
        } catch (_) {}

        // Flash save confirmation
        const btn = document.getElementById('btn-save-config');
        btn.textContent = 'Saved';
        btn.style.background = '#22c55e';
        setTimeout(() => { btn.textContent = 'Save'; btn.style.background = ''; }, 1500);
    },

    // ── Scanner ──
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
        const hitTape = document.getElementById('hit-tape');
        const hitTapeList = document.getElementById('hit-tape-list');

        progress.classList.remove('hidden');
        metrics.classList.remove('hidden');
        tabs.classList.remove('hidden');
        hitTape.classList.remove('hidden');
        hitTapeList.innerHTML = '';
        progressFill.style.width = '0%';

        document.getElementById('btn-scan').disabled = true;
        document.getElementById('btn-stop').classList.remove('hidden');
        document.getElementById('metric-scanned').textContent = '0';
        document.getElementById('metric-entries').textContent = '0';
        document.getElementById('scan-time').textContent = '--';

        // Reset filters so all hits are visible immediately
        Screener.currentFilter = 'All';
        Screener.minConfidence = 0;
        document.getElementById('filter-signal').value = 'All';
        document.getElementById('filter-confidence').value = '0';

        this.showTab('screener');

        // Reset and fetch VIX in background while scan runs
        ScannerResults.reset();
        ScannerResults.fetchVix();

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

            this.abortController = new AbortController();
            const res = await fetch('/api/scan/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: this.abortController.signal,
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
                            ScannerResults.addResult(r);

                            const isHit = BULLISH_SIGNALS.includes(r.signal || '');
                            if (isHit) {
                                this.scanResults.push(r);
                                Screener.render(this.scanResults, document.getElementById('screener-content'));
                            }
                            this.addFeedLine(r, isHit, hitTapeList);

                            const pct = (msg.progress.done / msg.progress.total * 100).toFixed(0);
                            progressFill.style.width = `${pct}%`;
                            progressText.textContent = `${msg.progress.done}/${msg.progress.total} | ${this.scanResults.length} hits`;
                            document.getElementById('metric-scanned').textContent = msg.progress.done;
                            document.getElementById('metric-entries').textContent = this.scanResults.length;

                        } else if (msg.type === 'progress') {
                            // Progress-only update (failed/skipped ticker)
                            this.allScanned = msg.progress.done;
                            const pct = (msg.progress.done / msg.progress.total * 100).toFixed(0);
                            progressFill.style.width = `${pct}%`;
                            progressText.textContent = `${msg.progress.done}/${msg.progress.total} | ${this.scanResults.length} hits`;
                            document.getElementById('metric-scanned').textContent = msg.progress.done;

                        } else if (msg.type === 'done') {
                            document.getElementById('scan-time').textContent = `${msg.summary.elapsed}s`;
                            progressText.textContent = `Done | ${this.allScanned} scanned | ${this.scanResults.length} hits | ${msg.summary.elapsed}s`;
                            ScannerResults.update(msg.summary);
                        }
                    } catch (_) {}
                }
            }

            Screener.render(this.scanResults, document.getElementById('screener-content'));
            if (this.scanResults.length) {
                Charts.regimeHeatmap('heatmap-chart', this.scanResults);
            }

            // Render LEAPS tab if LEAPS strategy was used
            const strategy = Settings.getVal('setting-strategy');
            if (strategy === 'leaps' && this.scanResults.length) {
                Leaps.render(this.scanResults, document.getElementById('leaps-content'));
            }

        } catch (err) {
            if (err.name !== 'AbortError') {
                document.getElementById('screener-content').innerHTML =
                    `<div style="color:var(--red); padding:0.5rem;">${err.message}</div>`;
            } else {
                progressText.textContent = `Stopped | ${this.allScanned} scanned | ${this.scanResults.length} hits`;
            }
        } finally {
            document.getElementById('btn-scan').disabled = false;
            document.getElementById('btn-stop').classList.add('hidden');
            this.abortController = null;
            this.scanning = false;
            // Collapse settings to give more room for results
            const sp = document.getElementById('settings-panel');
            if (sp) sp.removeAttribute('open');
        }
    },

    addFeedLine(r, isHit, container) {
        const line = document.createElement('div');
        if (isHit) {
            const sig = r.signal || '';
            const isEnter = sig.includes('ENTER') || sig.includes('BUY');
            const sigLabel = sig.replace('LONG -- ', '').replace('LEAPS -- ', 'LEAPS: ');
            const conf = r.regime_confidence ? ` ${Math.round(r.regime_confidence * 100)}%` : '';
            const chg = r.change_1d != null ? ` ${r.change_1d >= 0 ? '+' : ''}${r.change_1d.toFixed(1)}%` : '';
            const cmet = r.confirmations_met || 0;
            const ctot = r.confirmations_total || 12;
            line.className = `feed-line hit ${isEnter ? 'hit-enter' : 'hit-conf'}`;
            line.textContent = `>>> ${r.symbol} — ${sigLabel} | ${cmet}/${ctot} confs | ${r.regime_label || ''}${conf}${chg}`;
            line.onclick = () => this.drillDown(r.symbol);
        } else {
            line.className = 'feed-line scan';
            line.textContent = `    ${r.symbol} — ${r.regime_label || r.signal || 'skip'}`;
        }
        container.appendChild(line);
        // Keep feed scrolled to bottom, trim old scan lines to save memory
        if (container.children.length > 200) {
            const old = container.querySelector('.feed-line.scan');
            if (old) old.remove();
        }
        container.scrollTop = container.scrollHeight;
    },

    stopScan() {
        if (this.abortController) {
            this.abortController.abort();
        }
    },

    async drillDown(symbol) {
        const strategy = Settings.getVal('setting-strategy');
        if (strategy === 'leaps') {
            this.showTab('leaps');
            await Leaps.drillDown(symbol);
        } else {
            this.showTab('drilldown');
            await DrillDown.render(symbol, document.getElementById('drilldown-content'));
        }
    },
};

document.addEventListener('DOMContentLoaded', () => App.init());
