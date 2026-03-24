/**
 * orders.js &mdash; Orders & Positions management
 * Orders tab: top 3 contracts with BUY buttons
 * Positions tab: filled orders with live P&L
 */

const Orders = {
    pending: [],    // { symbol, strike, dte, mid, delta, contracts, totalCost, status: 'ready'|'buying'|'filled' }
    positions: [],  // { symbol, strike, dte, contracts, fillPrice, fillTime, currentPrice, pnlDollars, pnlPct }

    init() {
        // Load positions from localStorage
        try {
            this.positions = JSON.parse(localStorage.getItem('rrjcar_positions') || '[]');
        } catch (_) { this.positions = []; }
        this.renderPositions();
    },

    /** Called from DrillDown when position sizing loads - populates Orders tab */
    setContracts(symbol, recommendations) {
        this.pending = recommendations.slice(0, 3).map((r, i) => ({
            symbol,
            strike: r.strike,
            dte: r.dte,
            mid: r.mid,
            delta: r.delta,
            iv_pct: r.iv_pct,
            volume: r.volume,
            openInterest: r.openInterest,
            contracts: r.contracts || 1,
            totalCost: r.total_cost || 0,
            score: r.score || 0,
            rank: i,
            status: 'ready',
        }));
        this.renderOrders();
    },

    renderOrders() {
        const container = document.getElementById('orders-content');
        if (!this.pending.length) {
            container.innerHTML = '<div style="color:var(--text-dim); padding:1rem; text-align:center; font-size:0.75rem;">Select a ticker from Hits to see order recommendations.</div>';
            return;
        }

        const sym = this.pending[0].symbol;
        let html = `<div class="orders-header">${sym} &mdash; Top Contracts</div>`;
        html += '<div class="orders-list">';

        this.pending.forEach((o, i) => {
            const isBest = i === 0;
            const bestTag = isBest ? '<span class="order-best">BEST</span>' : '';
            const btnClass = o.status === 'filled' ? 'order-btn filled' : (o.status === 'buying' ? 'order-btn buying' : 'order-btn ready');
            const btnText = o.status === 'filled' ? 'FILLED' : (o.status === 'buying' ? 'BUYING...' : 'BUY');

            html += `
            <div class="order-card ${isBest ? 'best' : ''}">
                <div class="order-info">
                    ${bestTag}
                    <span class="order-strike">$${o.strike} C</span>
                    <span class="order-detail">${o.dte}d</span>
                    <span class="order-detail">$${o.mid.toFixed(2)} mid</span>
                    <span class="order-delta">d=${o.delta.toFixed(2)}</span>
                </div>
                <div class="order-sizing">
                    <span class="order-contracts">${o.contracts} contracts</span>
                    <span class="order-cost">= $${o.totalCost.toLocaleString()}</span>
                    <span class="order-score">Score: ${o.score.toFixed(0)}</span>
                </div>
                <div class="order-actions">
                    <button class="${btnClass}" id="order-btn-${i}" onclick="Orders.buyContract(${i})" ${o.status !== 'ready' ? 'disabled' : ''}>
                        ${btnText}
                    </button>
                </div>
            </div>`;
        });

        html += '</div>';
        container.innerHTML = html;
    },

    async buyContract(index) {
        const order = this.pending[index];
        if (!order || order.status !== 'ready') return;

        order.status = 'buying';
        this.renderOrders();

        try {
            // Use ladder order for the underlying symbol
            const res = await API.ladderOrder(order.symbol, 'buy', order.contracts);

            if (res.error) {
                order.status = 'ready';
                this.renderOrders();
                return;
            }

            // Poll for fill
            this.pollOrder(index, res.order_id);

        } catch (err) {
            order.status = 'ready';
            this.renderOrders();
        }
    },

    async pollOrder(index, orderId) {
        const poll = async () => {
            try {
                const s = await API.ladderStatus(orderId);
                if (s.status === 'filled') {
                    this.pending[index].status = 'filled';
                    this.renderOrders();
                    // Move to positions
                    this.addPosition(this.pending[index], s.fill_price);
                } else if (s.status === 'exhausted' || s.status === 'error') {
                    this.pending[index].status = 'ready';
                    this.renderOrders();
                } else {
                    setTimeout(poll, 1500);
                }
            } catch (_) {
                setTimeout(poll, 2000);
            }
        };
        poll();
    },

    addPosition(order, fillPrice) {
        const pos = {
            symbol: order.symbol,
            strike: order.strike,
            dte: order.dte,
            contracts: order.contracts,
            fillPrice: fillPrice || order.mid,
            fillTime: new Date().toISOString(),
            currentPrice: fillPrice || order.mid,
            pnlDollars: 0,
            pnlPct: 0,
        };
        this.positions.push(pos);
        localStorage.setItem('rrjcar_positions', JSON.stringify(this.positions));
        this.renderPositions();
    },

    renderPositions() {
        const container = document.getElementById('positions-content');
        if (!container) return;

        if (!this.positions.length) {
            container.innerHTML = '<div style="color:var(--text-dim); padding:1rem; text-align:center; font-size:0.75rem;">No open positions.</div>';
            return;
        }

        let html = '<div class="positions-header">Open Positions</div>';
        html += '<div class="positions-list">';

        this.positions.forEach((p, i) => {
            const pnlColor = p.pnlDollars >= 0 ? 'bull' : 'bear';
            const pnlSign = p.pnlDollars >= 0 ? '+' : '';
            const time = new Date(p.fillTime).toLocaleDateString();

            html += `
            <div class="position-card">
                <div class="position-info">
                    <span class="position-symbol">${p.symbol}</span>
                    <span class="position-detail">$${p.strike} C</span>
                    <span class="position-detail">${p.contracts} contracts</span>
                    <span class="position-detail">Filled ${time}</span>
                </div>
                <div class="position-pnl">
                    <span class="position-fill">Fill: $${p.fillPrice.toFixed(2)}</span>
                    <span class="position-current">Now: $${p.currentPrice.toFixed(2)}</span>
                    <span class="position-gain ${pnlColor}">${pnlSign}$${p.pnlDollars.toFixed(2)}</span>
                    <span class="position-gain-pct ${pnlColor}">${pnlSign}${p.pnlPct.toFixed(1)}%</span>
                </div>
                <div class="position-actions">
                    <button class="order-btn close-btn" onclick="Orders.closePosition(${i})">CLOSE</button>
                </div>
            </div>`;
        });

        html += '</div>';
        container.innerHTML = html;
    },

    async closePosition(index) {
        const pos = this.positions[index];
        if (!pos) return;

        try {
            await API.ladderOrder(pos.symbol, 'sell', pos.contracts);
        } catch (_) {}

        // Remove from positions
        this.positions.splice(index, 1);
        localStorage.setItem('rrjcar_positions', JSON.stringify(this.positions));
        this.renderPositions();
    },

    /** Refresh position prices from broker */
    async refreshPositions() {
        if (!this.positions.length) return;
        try {
            const bp = await API.brokerPositions();
            if (!bp || !bp.positions) return;

            this.positions.forEach(pos => {
                const match = bp.positions.find(p =>
                    p.symbol === pos.symbol || (p.symbol && p.symbol.startsWith(pos.symbol))
                );
                if (match && match.cost_basis) {
                    pos.currentPrice = match.cost_basis / (match.quantity || 1);
                    pos.pnlDollars = (pos.currentPrice - pos.fillPrice) * pos.contracts * 100;
                    pos.pnlPct = pos.fillPrice > 0 ? ((pos.currentPrice - pos.fillPrice) / pos.fillPrice) * 100 : 0;
                }
            });

            localStorage.setItem('rrjcar_positions', JSON.stringify(this.positions));
            this.renderPositions();
        } catch (_) {}
    },
};
