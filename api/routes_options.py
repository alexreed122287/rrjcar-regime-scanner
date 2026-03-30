"""API routes for options recommendations and GEX analysis."""

from fastapi import APIRouter

router = APIRouter()

# Import lazily to avoid slow startup
_options_picker = None
_gex_engine = None
_leaps_strategy = None


def _get_picker():
    global _options_picker
    if _options_picker is None:
        import options_picker as op
        _options_picker = op
    return _options_picker


def _get_gex():
    global _gex_engine
    if _gex_engine is None:
        import gex_engine as ge
        _gex_engine = ge
    return _gex_engine


def _get_leaps():
    global _leaps_strategy
    if _leaps_strategy is None:
        import strategy_leaps as sl
        _leaps_strategy = sl
    return _leaps_strategy


@router.get("/leaps/{symbol}")
async def get_leaps(symbol: str, top_n: int = 5, min_dte: int = 180, max_dte: int = 730):
    """Get best LEAPS contracts for a symbol with scoring."""
    try:
        leaps = _get_leaps()

        # Get spot price and HV rank from scan cache or fetch
        spot = 0
        hv_rank = 0.5
        try:
            from api.routes_scan import _scan_cache
            full_results = _scan_cache.get("results_full", [])
            cached = next((r for r in full_results if r.get("symbol", "").upper() == symbol.upper()), None)
            if cached and cached.get("price"):
                spot = cached["price"]
                hv_rank = cached.get("hv_rank", 0.5) or 0.5
        except Exception:
            pass

        if not spot:
            from data_loader import fetch_data
            df = fetch_data(symbol=symbol, period_days=30, interval="1d")
            spot = float(df["Close"].iloc[-1])

        result = leaps.find_best_leaps(
            symbol=symbol,
            spot_price=spot,
            hv_rank=hv_rank,
            min_dte=min_dte,
            max_dte=max_dte,
            top_n=top_n,
        )

        # Serialize numpy types
        clean_recs = []
        for r in result.get("recommendations", []):
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
        risk_pct = settings.get("risk_pct", 10)
        risk_amount = capital * (risk_pct / 100)

        for cr in clean_recs:
            mid = cr.get("mid", 0) or 0
            if mid > 0:
                contract_cost = mid * 100
                cr["contracts"] = max(1, int(risk_amount / contract_cost))
                cr["total_cost"] = round(cr["contracts"] * contract_cost, 2)
                cr["pct_of_capital"] = round(cr["total_cost"] / capital * 100, 2)

        return {
            "symbol": result.get("symbol"),
            "spot_price": result.get("spot_price"),
            "recommendations": clean_recs,
            "total_evaluated": result.get("total_evaluated", 0),
            "error": result.get("error"),
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}


@router.get("/gex/{symbol}")
async def get_gex(symbol: str, min_dte: int = 0, max_dte: int = 365):
    """Get full GEX profile for a symbol."""
    try:
        gex = _get_gex()
        profile = gex.compute_gex_profile(symbol, min_dte=min_dte, max_dte=max_dte)

        if profile.get("error"):
            return {"error": profile["error"], "symbol": symbol.upper()}

        # Get regime info from scan cache if available
        regime_id = 3
        regime_label = "Unknown"
        try:
            from api.routes_scan import _scan_cache
            full_results = _scan_cache.get("results_full", [])
            cached = next((r for r in full_results if r.get("symbol", "").upper() == symbol.upper()), None)
            if cached:
                regime_id = cached.get("regime_id", 3)
                regime_label = cached.get("regime_label", "Unknown")
        except Exception:
            pass

        # Get GEX-informed strategy
        strategy = gex.gex_contract_strategy(profile, regime_id, regime_label)

        return {
            **profile,
            "strategy": strategy,
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}


@router.get("/options/{symbol}")
async def get_options(
    symbol: str,
    min_dte: int = 0,
    max_dte: int = 365,
    top_n: int = 5,
    include_gex: bool = True,
):
    try:
        from api.routes_scan import _scan_cache
        picker = _get_picker()

        # Try to get regime info from cache
        full_results = _scan_cache.get("results_full", [])
        cached = next((r for r in full_results if r.get("symbol", "").upper() == symbol.upper()), None)

        regime_id = 3
        regime_label = "Unknown"
        confirmations = 0
        signal = ""
        price = 0

        if cached and cached.get("price"):
            regime_id = cached.get("regime_id", 3)
            regime_label = cached.get("regime_label", "Unknown")
            confirmations = cached.get("confirmations_met", 0)
            signal = cached.get("signal", "")
            price = cached["price"]
        else:
            from data_loader import fetch_data
            df = fetch_data(symbol=symbol, period_days=30, interval="1d")
            price = float(df["Close"].iloc[-1])

        # Get GEX strategy if requested
        gex_strategy = None
        gex_profile = None
        if include_gex:
            try:
                gex = _get_gex()
                gex_profile = gex.compute_gex_profile(symbol, min_dte=min_dte, max_dte=max_dte)
                if not gex_profile.get("error"):
                    gex_strategy = gex.gex_contract_strategy(gex_profile, regime_id, regime_label)
            except Exception:
                pass

        recs = picker.get_options_recommendations(
            symbol=symbol,
            current_price=price,
            regime_id=regime_id,
            regime_label=regime_label,
            confirmations=confirmations,
            signal=signal,
            min_dte=min_dte,
            max_dte=max_dte,
            top_n=top_n,
            gex_strategy=gex_strategy,
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
        risk_pct = settings.get("risk_pct", 2)
        risk_amount = capital * (risk_pct / 100)

        stock_price = recs.get("price", 0) or 0
        shares_sized = int(risk_amount / stock_price) if stock_price > 0 else 0

        for cr in clean_recs:
            mid = cr.get("mid", 0) or 0
            if mid > 0:
                contract_cost = mid * 100
                cr["contracts"] = max(1, int(risk_amount / contract_cost))
                cr["total_cost"] = round(cr["contracts"] * contract_cost, 2)
                cr["pct_of_capital"] = round(cr["total_cost"] / capital * 100, 2)
            else:
                cr["contracts"] = 0
                cr["total_cost"] = 0
                cr["pct_of_capital"] = 0

        result = {
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

        # Include GEX data
        if gex_strategy:
            result["gex_strategy"] = gex_strategy
        if gex_profile and not gex_profile.get("error"):
            result["gex"] = {
                "call_wall": gex_profile.get("call_wall"),
                "put_wall": gex_profile.get("put_wall"),
                "gex_flip": gex_profile.get("gex_flip"),
                "max_gamma_strike": gex_profile.get("max_gamma_strike"),
                "gex_bias": gex_profile.get("gex_bias"),
                "total_gex": gex_profile.get("total_gex"),
            }

        return result

    except Exception as e:
        return {"error": str(e), "symbol": symbol.upper()}
