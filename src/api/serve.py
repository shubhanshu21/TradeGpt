"""
KAT Inference API
=================
FastAPI server exposing:
  POST /predict/kat13        → single next-candle price
  POST /predict/kat14        → single next-candle price (high-dim)
  POST /predict/kat2/{variant} → full trajectory (N steps)
  GET  /health               → health check + loaded models

Designed to be production-ready:
  - Loads model + scaler once at startup
  - Input validation via Pydantic
  - Async prediction (non-blocking)
  - CORS enabled for web clients

Run:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload
"""

import os, sys
import numpy as np
from pathlib import Path
from typing import List, Optional
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path setup ───────────────────────────────────────────────────────────────
SRC_ROOT = Path(__file__).parent.parent
PROJ_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT / "architectures"))

from preprocess import KATScaler, add_derived_features, build_feature_cols
import pandas as pd
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

CKPT_DIR = PROJ_ROOT / "models"

_models  = {}
_scalers = {}

def load_model(name: str):
    """Lazy-load a model by name. Cached after first load."""
    if name in _models:
        return _models[name]

    path = CKPT_DIR / f"{name}_final.keras"
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    print(f"Loading {name} from {path}...")
    _models[name] = tf.keras.models.load_model(str(path))
    return _models[name]


def load_scaler(variant: str = "base") -> KATScaler:
    if variant in _scalers:
        return _scalers[variant]
    path = CKPT_DIR / f"scaler_{variant}.pkl"
    if not path.exists():
        path = CKPT_DIR / "scaler_base.pkl"
    if not path.exists():
        # Fallback to current project root scaler location if any
        path = PROJ_ROOT / "models" / "scaler_base.pkl"
        
    scaler = KATScaler.load(str(path))
    _scalers[variant] = scaler
    return scaler


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def get_latest_history():
    """Scans logs/ for the newest training history."""
    try:
        log_root = PROJ_ROOT / "logs" / "kat3"
        if not log_root.exists(): return []
        
        # Get newest folder
        sessions = sorted([d for d in log_root.iterdir() if d.is_dir()])
        if not sessions: return []
        
        hist_file = sessions[-1] / "history.csv"
        if not hist_file.exists(): return []
        
        df = pd.read_csv(hist_file)
        # Format for JS: [{epoch: 0, mae: 1.2}, ...]
        return df[['epoch', 'mae', 'loss']].to_dict(orient='records')
    except:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

class CandleRow(BaseModel):
    """Single 1-minute candle + order book snapshot."""
    open:    float
    high:    float
    low:     float
    close:   float
    volume:  float
    bid1: float; bid2: float; bid3: float; bid4: float; bid5: float
    ask1: float; ask2: float; ask3: float; ask4: float; ask5: float
    bid_vol1: float; bid_vol2: float; bid_vol3: float; bid_vol4: float; bid_vol5: float
    ask_vol1: float; ask_vol2: float; ask_vol3: float; ask_vol4: float; ask_vol5: float


class PredictRequest(BaseModel):
    """
    Send the last `context_window` candles as an ordered list.
    Minimum 150 rows for base models; 1440 for Tiger.
    """
    candles: List[CandleRow] = Field(
        ..., min_length=150,
        description="Ordered list of 1-min candles, oldest first"
    )
    steps: int = Field(default=60, ge=1, le=1440,
                       description="Steps to forecast (KAT 2 only)")


class PredictResponse(BaseModel):
    model:       str
    next_close:  Optional[float]   = None   # KAT 1.x
    trajectory:  Optional[List[float]] = None  # KAT 2
    direction:   str                   # "UP" | "DOWN" | "FLAT"
    confidence:  float                 # placeholder


# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="KAT Predictive Engine",
    description="Real-time BTC price trajectory forecasting API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def candles_to_array(candles: List[CandleRow], scaler: KATScaler) -> np.ndarray:
    """Convert candle list → scaled feature array (N, n_features)."""
    rows = [c.dict() for c in candles]
    df   = pd.DataFrame(rows)
    df   = add_derived_features(df)
    feature_cols = [c for c in build_feature_cols() if c in df.columns]
    X = df[feature_cols].values
    return scaler.transform_X(X), feature_cols


# ── Dashboard (The Visual HUD) ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    dash_path = Path(__file__).parent / "dashboard.html"
    if dash_path.exists():
        return dash_path.read_text()
    return "<h1>KAT 3 — Dashboard Missing! Check src/api/dashboard.html</h1>"


@app.get("/api/stats")
async def get_stats():
    """Neural Pulse: Feeds the dashboard with live data."""
    history = get_latest_history()
    latest_loss = history[-1]['mae'] if history else 0.0
    status = "MODEL TRAINING" if history else "ARCHIVE READY"
    
    return {
        "status": status,
        "latest_loss": latest_loss,
        "history": history[-200:] # Last 200 epochs only for performance
    }


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_models": list(_models.keys()),
        "gpu": bool(tf.config.list_physical_devices("GPU")),
    }


@app.post("/predict/kat13", response_model=PredictResponse)
async def predict_kat13(req: PredictRequest):
    try:
        model  = load_model("kat13")
        scaler = load_scaler("base")

        X_scaled, _ = candles_to_array(req.candles[-150:], scaler)
        inp = X_scaled[np.newaxis]  # (1, 150, F)

        pred_scaled = float(model.predict(inp, verbose=0)[0])
        pred_price  = float(scaler.inverse_y(np.array([pred_scaled]))[0])
        last_close  = req.candles[-1].close

        direction = "UP" if pred_price > last_close else ("DOWN" if pred_price < last_close else "FLAT")

        return PredictResponse(
            model="KAT 1.3",
            next_close=round(pred_price, 2),
            direction=direction,
            confidence=0.0,  # implement calibrated probability here
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/kat14", response_model=PredictResponse)
async def predict_kat14(req: PredictRequest):
    try:
        model  = load_model("kat14")
        scaler = load_scaler("base")

        X_scaled, _ = candles_to_array(req.candles[-150:], scaler)
        inp = X_scaled[np.newaxis]

        pred_scaled = float(model.predict(inp, verbose=0)[0])
        pred_price  = float(scaler.inverse_y(np.array([pred_scaled]))[0])
        last_close  = req.candles[-1].close

        direction = "UP" if pred_price > last_close else ("DOWN" if pred_price < last_close else "FLAT")

        return PredictResponse(
            model="KAT 1.4",
            next_close=round(pred_price, 2),
            direction=direction,
            confidence=0.0,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/kat2/{variant}", response_model=PredictResponse)
async def predict_kat2(variant: str, req: PredictRequest):
    """
    variant: base | lion | tiger
    Returns a full price trajectory of `steps` candles.
    """
    if variant not in ("base", "lion", "tiger"):
        raise HTTPException(status_code=400, detail="variant must be base|lion|tiger")

    ctx_map = {"base": 150, "lion": 480, "tiger": 1440}
    ctx = ctx_map[variant]

    try:
        from kat2 import KAT2   # needed for generate()
        model  = load_model(f"kat2_{variant}")
        scaler = load_scaler(variant)

        if len(req.candles) < ctx:
            raise HTTPException(
                status_code=422,
                detail=f"KAT 2 {variant} requires at least {ctx} candles"
            )

        X_scaled, _ = candles_to_array(req.candles[-ctx:], scaler)
        seed = X_scaled   # (ctx, F)

        # Autoregressive generation
        traj = model.generate(seed, steps=req.steps, scaler=scaler)
        last_close = req.candles[-1].close
        direction  = "UP" if traj[-1] > last_close else ("DOWN" if traj[-1] < last_close else "FLAT")

        return PredictResponse(
            model=f"KAT 2 {variant.capitalize()}",
            trajectory=[round(float(p), 2) for p in traj],
            direction=direction,
            confidence=0.0,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)