/**
 * settings.js — Settings panel logic
 * Uses /api/watchlists/all to include the 10K+ universe lists
 */

const DTE_PRESETS = {
    '0dte':    { min: 0, max: 0 },
    'weekly':  { min: 0, max: 7 },
    'swing':   { min: 14, max: 45 },
    'monthly': { min: 30, max: 60 },
    'leaps':   { min: 180, max: 730 },
    'all':     { min: 0, max: 365 },
};

const Settings = {
    watchlists: {},

    async init() {
        try {
            const [wl, settings] = await Promise.all([
                API.getAllWatchlists(),   // includes ALL TICKERS, All Stocks, All ETFs
                API.getSettings(),
            ]);
            this.watchlists = wl;
            this.populate(settings);
        } catch (err) {
            console.error('Settings init error:', err);
        }
    },

    populate(settings) {
        const wlSelect = document.getElementById('setting-watchlist');
        if (wlSelect) {
            wlSelect.innerHTML = '';

            // Sort: curated first, then big universes
            const curated = [];
            const universes = [];
            Object.keys(this.watchlists).forEach(name => {
                const count = this.watchlists[name].length;
                if (count > 500) {
                    universes.push(name);
                } else {
                    curated.push(name);
                }
            });

            // Add curated
            curated.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = `${name} (${this.watchlists[name].length})`;
                if (name === settings.watchlist) opt.selected = true;
                wlSelect.appendChild(opt);
            });

            // Separator
            if (universes.length) {
                const sep = document.createElement('option');
                sep.disabled = true;
                sep.textContent = '────── Full Universe ──────';
                wlSelect.appendChild(sep);

                universes.forEach(name => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = `${name} (${this.watchlists[name].length})`;
                    if (name === settings.watchlist) opt.selected = true;
                    wlSelect.appendChild(opt);
                });
            }
        }

        this.setVal('setting-custom-tickers', settings.custom_tickers || '');
        this.setVal('setting-strategy', settings.strategy || 'v2');
        this.setVal('setting-min-confs', settings.min_confs || 6);
        this.setVal('setting-regime-confirm', settings.regime_confirm || 2);
        this.setVal('setting-cooldown', settings.cooldown || 3);
        this.setVal('setting-capital', settings.initial_capital || 100000);
        this.setVal('setting-max-workers', settings.max_workers || 6);
        this.setVal('setting-min-dte', settings.min_dte ?? 0);
        this.setVal('setting-max-dte', settings.max_dte ?? 365);
    },

    setVal(id, val) {
        const el = document.getElementById(id);
        if (el) el.value = val;
    },

    getVal(id) {
        const el = document.getElementById(id);
        return el ? el.value : '';
    },

    gather() {
        return {
            watchlist: this.getVal('setting-watchlist'),
            custom_tickers: this.getVal('setting-custom-tickers'),
            strategy: this.getVal('setting-strategy'),
            min_confs: parseInt(this.getVal('setting-min-confs')) || 6,
            regime_confirm: parseInt(this.getVal('setting-regime-confirm')) || 2,
            cooldown: parseInt(this.getVal('setting-cooldown')) || 3,
            max_workers: parseInt(this.getVal('setting-max-workers')) || 6,
            min_dte: parseInt(this.getVal('setting-min-dte')) || 0,
            max_dte: parseInt(this.getVal('setting-max-dte')) || 365,
        };
    },

    applyDtePreset(preset) {
        if (!preset || !DTE_PRESETS[preset]) return;
        const p = DTE_PRESETS[preset];
        this.setVal('setting-min-dte', p.min);
        this.setVal('setting-max-dte', p.max);
    },

    async save() {
        try {
            const data = this.gather();
            data.initial_capital = parseFloat(this.getVal('setting-capital')) || 100000;
            await API.saveSettings(data);
        } catch (err) {
            console.error('Save settings error:', err);
        }
    },

    async saveAndConfirm(btn) {
        await this.save();
        const orig = btn.textContent;
        btn.textContent = 'Saved';
        btn.style.background = '#22c55e';
        btn.style.borderColor = '#22c55e';
        btn.style.color = '#000';
        setTimeout(() => {
            btn.textContent = orig;
            btn.style.background = '#1e2028';
            btn.style.borderColor = '#2dd4bf';
            btn.style.color = '#2dd4bf';
        }, 1500);
    },
};
