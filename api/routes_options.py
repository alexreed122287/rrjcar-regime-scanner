"""API routes for options recommendations."""

from fastapi import APIRouter

router = APIRouter()

# Import lazily to avoid slow startup
_options_picker = None


def _get_picker():
    global _options_picker
    if _options_picker is None:
        import options_picker as op
        _options_picker = op
    return _options_picker


@router.get("/options/{symbol}")
async def get_options(
    symbol: str,
    min_dte: int = 21,
    max_dte: int = 45,
    top_n: int = 3,
):
    try:
        from api.routes_scan import _scan_cache
        picker = _get_picker()

        # Try to get regime info from cache
        full_results = _scan_cache.get("results_full", [])
        cached = next((r for r in full_results if r.get("symbol", "").upper() == symbol.upper()), None)

        if cached and cached.get("price"):
            recs = picker.get_options_recommendations(
                symbol=symbol,
                current_price=cached["price"],
                regime_id=cached.get("regime_id", 3),
                regime_label=cached.get("regime_label", "Unknown"),
                confirmations=cached.get("confirmations_met", 0),
                signal=cached.get("signal", ""),
                min_dte=min_dte,
                max_dte=max_dte,
                top_n=top_n,
            )
        else:
            # Fetch price and use defaults
            from data_loader import fetch_data
            df = fetch_data(symbol=symbol, period_days=30, interval="1d")
            price = float(df["Close"].iloc[-1])
            recs = picker.get_options_recommendations(
                symbol=symbol,
                current_price=price,
                regime_id=3,
                regime_label="Unknown",
                confirmations=0,
                signal="",
                min_dte=min_dte,
                max_dte=max_dte,
                top_n=top_n,
            )

        # Serialize recommendations
        clean_recs = []
        for r in recs.get("recommendations", []):
            cr = {}
            for k, v in r.items():
                if hasattr(v, "item"):
                    cr[k] = v.item()
                elif isinstance(v, float) and v != v:
                    cr[k] = None
                else:
                    cr[k] = v
            clean_recs.append(cr)

        # Position sizing
        from settings_manager import load_settings
        settings = load_settings()
        capital = settings.get("initial_capital", 100000)
        risk_pct = settings.get("risk_pct", 2)  # default 2% risk per trade
        risk_amount = capital * (risk_pct / 100)

        stock_price = recs.get("price", 0) or 0
        shares_sized = int(risk_amount / stock_price) if stock_price > 0 else 0

        for cr in clean_recs:
            mid = cr.get("mid", 0) or 0
            if mid > 0:
                # Max contracts = risk_amount / (contract cost * 100)
                contract_cost = mid * 100
                cr["contracts"] = max(1, int(risk_amount / contract_cost))
                cr["total_cost"] = round(cr["contracts"] * contract_cost, 2)
                cr["pct_of_capital"] = round(cr["total_cost"] / capital * 100, 2)
            else:
                cr["contracts"] = 0
                cr["total_cost"] = 0
                cr["pct_of_capital"] = 0

        return {
            "symbol": recs.get("symbol"),
            "price": recs.get("price"),
            "regime_label": recs.get("regime_label"),
            "signal": recs.get("signal"),
            "recommendations": clean_recs,
            "error": recs.get("error"),
            "position_sizing": {
                "capital": capital,
                "risk_pct": risk_pct,
                "risk_amount": round(risk_amount, 2),
                "shares_equity": shares_sized,
                "shares_cost": round(shares_sized * stock_price, 2),
            },
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}
