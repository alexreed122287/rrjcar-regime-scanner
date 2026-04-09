/**
 * charts.js — Plotly chart builders (dark terminal theme)
 */

const REGIME_COLORS = {
    0: '#2dd4bf', 1: '#14b8a6', 2: '#5eead4',
    3: '#6b7280', 4: '#9ca3af', 5: '#f87171', 6: '#ef4444',
};
const REGIME_BG_COLORS = {
    0: '#0d3d38', 1: '#0d3330', 2: '#1a3a35',
    3: '#1f2937', 4: '#27272a', 5: '#3b1818', 6: '#451a1a',
};
const REGIME_LABELS = [
    'Bull Run', 'Bull Trend', 'Mild Bull', 'Neutral / Chop',
    'Mild Bear', 'Bear Trend', 'Crash / Capitulation',
];

const DARK_LAYOUT = {
    template: 'plotly_dark',
    paper_bgcolor: '#101114',
    plot_bgcolor: '#101114',
    font: { family: 'Inter, sans-serif', color: '#e5e7eb' },
    xaxis: { gridcolor: '#1f2937' },
    yaxis: { gridcolor: '#1f2937' },
    margin: { l: 55, r: 15, t: 45, b: 35 },
    legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, font: { size: 10 } },
};

const Charts = {
    priceWithRegimes(containerId, data, title = '') {
        if (!data || !data.dates) return;
        const traces = [];
        const uniqueRegimes = [...new Set(data.regime_ids)].sort((a, b) => a - b);

        uniqueRegimes.forEach(rid => {
            const mask = data.regime_ids.map((r, i) => r === rid ? i : -1).filter(i => i >= 0);
            traces.push({
                x: mask.map(i => data.dates[i]),
                y: mask.map(i => data.close[i]),
                mode: 'markers',
                marker: { size: 3, color: REGIME_COLORS[rid] || '#666' },
                name: REGIME_LABELS[rid] || `State ${rid}`,
                hovertemplate: `<b>${REGIME_LABELS[rid] || rid}</b><br>$%{y:,.2f}<br>%{x}<extra></extra>`,
            });
        });

        Plotly.newPlot(containerId, traces, {
            ...DARK_LAYOUT, title: title, height: 280,
            yaxis: { ...DARK_LAYOUT.yaxis, title: 'Price' },
        }, { responsive: true, displayModeBar: false });
    },

    equityCurve(containerId, eqData) {
        if (!eqData) return;
        const traces = [
            {
                x: eqData.dates, y: eqData.equity,
                mode: 'lines', name: 'HMM Strategy',
                line: { color: '#2dd4bf', width: 2 },
            },
            {
                x: eqData.bh_dates, y: eqData.bh_equity,
                mode: 'lines', name: 'Buy & Hold',
                line: { color: '#666', width: 1, dash: 'dash' },
            },
        ];

        Plotly.newPlot(containerId, traces, {
            ...DARK_LAYOUT,
            title: 'Strategy vs Buy & Hold',
            height: 260,
            yaxis: { ...DARK_LAYOUT.yaxis, title: 'Equity ($)' },
        }, { responsive: true, displayModeBar: false });
    },

    regimeHeatmap(containerId, results) {
        const valid = results.filter(r => r.regime_id != null && r.price != null);
        if (!valid.length) return;

        const symbols = valid.map(r => r.symbol);
        const regimeIds = valid.map(r => r.regime_id);
        const invOrder = [6, 5, 4, 3, 2, 1, 0];
        const invLabels = invOrder.map(i => REGIME_LABELS[i]);

        const matrix = symbols.map((_, i) => {
            const row = new Array(7).fill(0);
            const colIdx = invOrder.indexOf(regimeIds[i]);
            if (colIdx >= 0) row[colIdx] = 1;
            return row;
        });

        const traces = [{
            z: matrix, x: invLabels, y: symbols, type: 'heatmap',
            colorscale: [[0, '#101114'], [0.5, '#1a1f2e'], [1, '#2dd4bf']],
            showscale: false,
            hovertemplate: '<b>%{y}</b><br>%{x}<extra></extra>',
        }];

        // Marker dots
        valid.forEach((r, i) => {
            const colIdx = invOrder.indexOf(r.regime_id);
            if (colIdx >= 0) {
                traces.push({
                    x: [invLabels[colIdx]], y: [r.symbol], mode: 'markers',
                    marker: { size: 14, color: REGIME_COLORS[r.regime_id] || '#666', symbol: 'square' },
                    showlegend: false,
                    hovertemplate: `<b>${r.symbol}</b><br>${REGIME_LABELS[r.regime_id]}<extra></extra>`,
                });
            }
        });

        Plotly.newPlot(containerId, traces, {
            ...DARK_LAYOUT,
            height: Math.min(450, Math.max(200, symbols.length * 22 + 60)),
            xaxis: { ...DARK_LAYOUT.xaxis, side: 'top', tickfont: { size: 9 } },
            yaxis: { ...DARK_LAYOUT.yaxis, tickfont: { size: 9 } },
            legend: { ...DARK_LAYOUT.legend },
            margin: { l: 60, r: 10, t: 30, b: 20 },
        }, { responsive: true, displayModeBar: false });
    },

    signalDistribution(containerId, results) {
        const signals = results.map(r => r.signal).filter(Boolean);
        if (!signals.length) return;

        const counts = {};
        signals.forEach(s => { counts[s] = (counts[s] || 0) + 1; });

        const colorMap = {
            'LONG -- ENTER': '#2dd4bf', 'LONG -- HOLD': '#22c55e',
            'LONG -- CONFIRMING': '#88cc44', 'EXIT -- REGIME FLIP': '#ff4444',
            'CASH -- NEUTRAL': '#ffaa00', 'CASH -- BEARISH': '#ff6666',
        };

        const labels = Object.keys(counts);
        const values = Object.values(counts);
        const colors = labels.map(l => colorMap[l] || '#666');

        Plotly.newPlot(containerId, [{
            labels, values, type: 'pie', hole: 0.4,
            marker: { colors },
            textinfo: 'label+value',
            textfont: { size: 10 },
        }], {
            ...DARK_LAYOUT, title: 'Signal Distribution', height: 260,
            showlegend: false,
            margin: { l: 20, r: 20, t: 50, b: 20 },
        }, { responsive: true, displayModeBar: false });
    },
};
