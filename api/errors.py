"""Error responses that are honest about their HTTP status.

The route handlers used to swallow exceptions and ``return {"error": str(e)}``, which FastAPI
serializes as **HTTP 200**. A failed backtest was therefore indistinguishable from a
successful one to any client that checks ``response.ok`` / ``resp.status_code == 200`` -- the
failure only showed up if the caller happened to look for an ``"error"`` key in the body.

These helpers keep the body shape identical (so existing front-end checks for ``data.error``
keep working) while setting a truthful status code.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


def error_response(message: str, status_code: int = 500, **extra: Any) -> JSONResponse:
    """Return an error body with a real status code.

    Parameters
    ----------
    message : str
        Human-readable error, placed under the ``"error"`` key as before.
    status_code : int
        502 for upstream/data-provider failures, 404 for a missing resource, 400 for bad
        input, 500 for an unexpected internal error. Defaults to 500.
    **extra
        Additional keys merged into the body (e.g. ``symbol=...``, ``results=[]``) so
        response shapes stay backward compatible.
    """
    body: dict[str, Any] = {"error": str(message)}
    body.update(extra)
    return JSONResponse(status_code=int(status_code), content=body)
