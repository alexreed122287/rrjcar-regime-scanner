"""
app.py — FastAPI entry point for HMM Regime Terminal
Serves the REST API and static frontend files.
"""

# Thread caps first, before anything drags numpy in. OpenBLAS reads these variables
# when its shared library loads, so a later call is a silent no-op. See thread_limits
# for the measurements (3.5-4.4x on the process-pool scan path).
from thread_limits import apply_thread_caps

apply_thread_caps()

import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes_scan import router as scan_router
from api.routes_backtest import router as backtest_router
from api.routes_options import router as options_router
from api.routes_settings import router as settings_router
from api.routes_broker import router as broker_router

app = FastAPI(title="HMM Regime Terminal", version="2.0.0")

# Register API routes
app.include_router(scan_router, prefix="/api")
app.include_router(backtest_router, prefix="/api")
app.include_router(options_router, prefix="/api")
app.include_router(settings_router, prefix="/api")
app.include_router(broker_router, prefix="/api")

# Serve static frontend files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))
