"""API routes for Tradier broker integration."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class BrokerConnect(BaseModel):
    access_token: str
    account_id: str
    sandbox: bool = True


@router.get("/broker/status")
async def broker_status():
    try:
        from tradier_broker import is_configured, get_account_info
        configured = is_configured()
        info = None
        if configured:
            try:
                info = get_account_info()
            except Exception as e:
                info = {"error": str(e)}
        return {"configured": configured, "account_info": info}
    except Exception as e:
        return {"configured": False, "error": str(e)}


@router.post("/broker/connect")
async def broker_connect(req: BrokerConnect):
    try:
        from tradier_broker import save_config, is_configured, get_account_info
        save_config(req.access_token, req.account_id, req.sandbox)
        # Clear data_loader cache so it picks up new config
        import data_loader
        data_loader._tradier_config_cache = None
        data_loader._tradier_session = None

        configured = is_configured()
        info = None
        if configured:
            try:
                info = get_account_info()
            except Exception:
                pass
        return {"success": True, "configured": configured, "account_info": info}
    except Exception as e:
        return {"success": False, "error": str(e)}
