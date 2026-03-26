/**
 * api.js — Fetch wrappers for all API endpoints
 */

const API = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
        return res.json();
    },

    async post(url, data) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error(`POST ${url} failed: ${res.status}`);
        return res.json();
    },

    // Scan
    getWatchlists() { return this.get('/api/watchlists'); },
    getAllWatchlists() { return this.get('/api/watchlists/all'); },
    runScan(params) { return this.post('/api/scan', params); },
    scanStatus() { return this.get('/api/scan/status'); },
    getCached() { return this.get('/api/scan/cached'); },
    scanSymbol(symbol, strategy = 'v2') {
        return this.get(`/api/scan/${symbol}?strategy=${strategy}`);
    },

    // Backtest
    backtest(symbol, params = {}) {
        const qs = new URLSearchParams(params).toString();
        return this.get(`/api/backtest/${symbol}?${qs}`);
    },

    // Options
    getOptions(symbol, minDte = 0, maxDte = 365, topN = 5) {
        return this.get(`/api/options/${symbol}?min_dte=${minDte}&max_dte=${maxDte}&top_n=${topN}&include_gex=true`);
    },

    // GEX
    getGex(symbol, minDte = 0, maxDte = 365) {
        return this.get(`/api/gex/${symbol}?min_dte=${minDte}&max_dte=${maxDte}`);
    },

    // Settings
    getSettings() { return this.get('/api/settings'); },
    saveSettings(data) { return this.post('/api/settings', data); },

    // APIs status
    getApis() { return this.get('/api/apis'); },

    // Broker
    brokerStatus() { return this.get('/api/broker/status'); },
    brokerConnect(data) { return this.post('/api/broker/connect', data); },
    brokerPositions() { return this.get('/api/broker/positions'); },
    brokerOrders(status = 'all') { return this.get(`/api/broker/orders?status=${status}`); },

    // Ladder orders (incremental pricing)
    ladderOrder(symbol, side, quantity = 1) {
        return this.post('/api/broker/ladder', { symbol, side, quantity });
    },
    ladderStatus(orderId) { return this.get(`/api/broker/ladder/${orderId}`); },
};
