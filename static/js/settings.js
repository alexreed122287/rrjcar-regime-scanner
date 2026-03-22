/**
 * settings.js — Settings panel logic
 */

const Settings = {
    watchlists: {},

    async init() {
        try {
            const [wl, settings] = await Promise.all([
                API.getWatchlists(),
                API.getSettings(),
            ]);
            this.watchlists = wl;
            this.populate(settings);
        } catch (err) {
            console.error('Settings init error:', err);
        }
    },

    populate(settings) {
        // Watchlist dropdown
        const wlSelect = document.getElementById('setting-watchlist');
        if (wlSelect) {
            wlSelect.innerHTML = '';
            Object.keys(this.watchlists).forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = `${name} (${this.watchlists[name].length})`;
                if (name === settings.watchlist) opt.selected = true;
                wlSelect.appendChild(opt);
            });
        }

        // Other fields
        this.setVal('setting-custom-tickers', settings.custom_tickers || '');
        this.setVal('setting-strategy', settings.strategy || 'v2');
        this.setVal('setting-min-confs', settings.min_confs || 6);
        this.setVal('setting-regime-confirm', settings.regime_confirm || 2);
        this.setVal('setting-cooldown', settings.cooldown || 3);
        this.setVal('setting-capital', settings.initial_capital || 100000);
        this.setVal('setting-max-workers', settings.max_workers || 6);
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
        };
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
};
