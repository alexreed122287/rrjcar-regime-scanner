"""API routes for backtesting."""

from fastapi import APIRouter
from data_loader import fetch_data, engineer_features
from hmm_engine import RegimeDetector
from backtester import run_backtest
from strategy_v2 import run_backtest_v2

router = APIRouter()


@router.get("/backtest/{symbol}")
async def backtest_symbol(
    symbol: str,
    strategy: str = "v2",
    min_confs: int = 6,
    cooldown: int = 3,
    regime_confirm: int = 2,
    capital: float = 100000,
    n_regimes: int = 7,
):
    try:
        raw = fetch_data(symbol=symbol, period_days=730, interval="1d")
        feat = engineer_features(raw)
        detector = RegimeDetector(n_regimes=n_regimes)
        regime_df = detector.train(feat)

        if strategy == "v2":
            bt = run_backtest_v2(
                regime_df,
                min_confirmations=min_confs,
                cooldown_bars=cooldown,
                regime_confirm_bars=regime_confirm,
                initial_capital=capital,
            )
        else:
            bt = run_backtest(
                regime_df,
                min_confirmations=min_confs,
                cooldown_bars=cooldown,
                regime_confirm_bars=regime_confirm,
                initial_capital=capital,
            )

        metrics = bt["metrics"]
        # Serialize metrics (convert numpy types)
        clean_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                clean_metrics[k] = v.item()
            elif isinstance(v, float) and v != v:  # NaN check
                clean_metrics[k] = None
            else:
                clean_metrics[k] = v

        # Equity curve data
        eq = bt["equity_curve"]
        eq_data = {
            "dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in eq.index],
            "equity": [round(float(v), 2) for v in eq.values],
        }

        # Buy & hold for comparison
        bh_start = regime_df["Close"].iloc[0]
        bh_equity = (regime_df["Close"] / bh_start) * capital
        eq_data["bh_dates"] = [str(d.date()) if hasattr(d, "date") else str(d) for d in regime_df.index]
        eq_data["bh_equity"] = [round(float(v), 2) for v in bh_equity.values]

        # Trades
        trades = bt.get("trades", [])
        clean_trades = []
        for t in trades[:50]:  # limit to 50 trades
            ct = {}
            for k, v in t.items():
                if hasattr(v, "item"):
                    ct[k] = v.item()
                elif isinstance(v, float) and v != v:
                    ct[k] = None
                else:
                    ct[k] = v
            clean_trades.append(ct)

        return {
            "symbol": symbol.upper(),
            "strategy": strategy,
            "metrics": clean_metrics,
            "equity_curve": eq_data,
            "trades": clean_trades,
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}
