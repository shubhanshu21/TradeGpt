"""API + dashboard server for the equity swing trading pipeline — neural
network only (ml_swing), no rule-based strategies.

    uvicorn api.swing_dashboard.main:app --reload --port 9000

Then open http://localhost:9000 for the dashboard.

Scope, deliberately:
  - Backtest: fully controllable via API (safe - no money, no live orders).
  - Paper trading: fully controllable via API (safe - virtual orders only).
  - Live trading: READ-ONLY status here. There is intentionally no
    "start live trading" endpoint - placing real orders from a one-click
    web button defeats the whole point of the typed-confirmation safety
    gate in live_trading_swing/executor.py. Start it from a terminal you
    are actively watching: `python3 auto_run.py live`.
"""
import csv
import sys
import threading
from pathlib import Path

# kiteconnect (Zerodha) must be imported before tensorflow ever loads in
# this process - importing them in the opposite order segfaults (verified:
# a native-library conflict, not a Python-level error). This process is
# long-lived and can serve a backtest request (which lazy-loads tensorflow
# via MLSwingStrategy) and a paper/live-trading request (which can lazy-load
# kiteconnect via get_broker, if broker.name=zerodha) in either order across
# its lifetime, so the only safe fix is forcing kiteconnect first, always -
# harmless even when broker.name isn't zerodha.
import kiteconnect  # noqa: F401,E402

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # kat/
sys.path.insert(0, str(ROOT / "src"))

from backtest.metrics import compute_metrics  # noqa: E402
from exchange.brokers.factory import get_broker  # noqa: E402
from data.fetch_historical import fetch_universe_swing  # noqa: E402
from data.preprocess import load_universe_from_cache  # noqa: E402
from swing_backtest.engine import SwingBacktestEngine  # noqa: E402
from swing_paper_trading.engine import SwingPaperTradingEngine  # noqa: E402
from strategies.swing.ml_strategy import MLSwingStrategy  # noqa: E402

REPORTS_DIR = ROOT / "reports" / "swing"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"


def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


app = FastAPI(title="Equity Swing Trading Pipeline API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- in-memory job/session state (single-user, single-process by design) ----
_lock = threading.Lock()
_backtest_job = {"running": False, "error": None, "completed_at": None}
_paper_session = {"engine": None, "thread": None, "stop_event": None}


# ============================== General ==============================

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    return load_config()


# ============================== Today's Market View ==============================
# Reuses the SAME trained checkpoints backtest/paper/live trade on - this is
# read-only "what does the model think right now", not a trading action, so
# it's safe to expose without any confirmation gate (unlike live trading).
_snapshot_strategy = MLSwingStrategy()


@app.get("/api/predict/snapshot")
def predict_snapshot(symbol: str):
    cfg = load_config()
    if symbol not in cfg["universe"]["symbols"]:
        raise HTTPException(404, f"{symbol} is not in the configured universe.")

    data = load_universe_from_cache([symbol])
    if symbol not in data:
        return {"available": False, "symbol": symbol,
                "reason": "No cached price history for this stock yet - run a backtest "
                          "once (it fetches and caches historical data) or wait for the next data refresh."}

    return _snapshot_strategy.explain_latest(data[symbol], symbol)


# ============================== Backtest ==============================

def _run_backtest_job(refresh: bool):
    cfg = load_config()
    try:
        data = fetch_universe_swing(force_refresh=refresh)
        if not data:
            raise RuntimeError("No historical data returned - check broker credentials/config.")

        initial_capital = cfg["swing"]["initial_capital"]
        strategy = MLSwingStrategy()
        engine = SwingBacktestEngine(cfg, strategy)
        trades = engine.run(data)
        trades.to_csv(REPORTS_DIR / f"trades_{strategy.name}.csv", index=False)
        metrics = compute_metrics(trades, initial_capital)
        summary = pd.DataFrame([metrics]).set_index(pd.Index([strategy.name], name="strategy"))
        summary.to_csv(REPORTS_DIR / "strategy_comparison.csv")

        with _lock:
            _backtest_job["running"] = False
            _backtest_job["error"] = None
            _backtest_job["completed_at"] = pd.Timestamp.now().isoformat()
    except Exception as exc:
        with _lock:
            _backtest_job["running"] = False
            _backtest_job["error"] = str(exc)


@app.post("/api/backtest/run")
def run_backtest(refresh: bool = False):
    with _lock:
        if _backtest_job["running"]:
            raise HTTPException(409, "A backtest is already running.")
        _backtest_job["running"] = True
        _backtest_job["error"] = None

    thread = threading.Thread(target=_run_backtest_job, args=(refresh,), daemon=True)
    thread.start()
    return {"started": True}


@app.get("/api/backtest/status")
def backtest_status():
    with _lock:
        return dict(_backtest_job)


@app.get("/api/backtest/results")
def backtest_results():
    path = REPORTS_DIR / "strategy_comparison.csv"
    if not path.exists():
        raise HTTPException(404, "No backtest results yet - run a backtest first.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/api/backtest/trades")
def backtest_trades():
    path = REPORTS_DIR / "trades_ml_swing.csv"
    if not path.exists():
        raise HTTPException(404, "No trade log yet - run a backtest first.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


# ============================== Training progress ==============================

@app.get("/api/training/symbols")
def training_symbols():
    """Which symbols have per-epoch training history logged so far (see
    train.py's EdgeTracker callback) - one edge_tracker_<SYMBOL>.csv per
    symbol, written as that symbol's training run progresses."""
    return sorted(
        p.stem.removeprefix("edge_tracker_")
        for p in LOG_DIR.glob("edge_tracker_*.csv")
    )


@app.get("/api/training/progress")
def training_progress(symbol: str):
    path = LOG_DIR / f"edge_tracker_{symbol}.csv"
    if not path.exists():
        raise HTTPException(404, f"No training history yet for {symbol}.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/api/training/status")
def training_status():
    """Real per-symbol training status across the WHOLE tradeable universe
    (not just symbols with some history) - this is what backtest/paper
    trading actually need to know before running, since a symbol with no
    checkpoint yet is silently skipped rather than erroring. Without this,
    there's no way to tell "0 trades" (no signal) apart from "37 of 38
    symbols were never trained" from the UI alone."""
    cfg = load_config()
    universe = cfg["universe"]["symbols"]
    training_cfg = (cfg.get("training") or {}).get("symbols") or universe

    rows = []
    for symbol in universe:
        has_checkpoint = (MODELS_DIR / symbol / "hydra_best.keras").exists()
        edge_path = LOG_DIR / f"edge_tracker_{symbol}.csv"
        epochs_logged, best_val_acc, best_epoch, significant_edge, last_val_acc = 0, None, None, False, None
        if edge_path.exists():
            df = pd.read_csv(edge_path)
            if len(df):
                epochs_logged = len(df)
                best_row = df.loc[df["val_dir_acc"].idxmax()]
                best_val_acc = float(best_row["val_dir_acc"])
                best_epoch = int(best_row["epoch"])
                last_row = df.iloc[-1]
                last_val_acc = float(last_row["val_dir_acc"])
                significant_edge = bool(last_row["significant_edge"])
        rows.append({
            "symbol": symbol,
            "in_training_config": symbol in training_cfg,
            "has_checkpoint": has_checkpoint,
            "epochs_logged": epochs_logged,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "last_val_acc": last_val_acc,
            "significant_edge": significant_edge,
        })

    trained_count = sum(1 for r in rows if r["has_checkpoint"])
    return {"symbols": rows, "trained_count": trained_count, "universe_count": len(universe)}


# ============================== Paper trading ==============================

@app.post("/api/paper/start")
def paper_start(symbols: list[str] | None = None):
    cfg = load_config()
    if cfg["broker"]["name"] == "csv":
        raise HTTPException(
            400, "broker.name is 'csv' (backtest-only, static data) in config/settings.yaml. "
                 "Paper trading needs live prices - set broker.name to zerodha, upstox, or dhan first.")

    with _lock:
        if _paper_session["engine"] is not None and _paper_session["thread"].is_alive():
            raise HTTPException(409, "A paper trading session is already running. Stop it first.")

        broker = get_broker(cfg)
        strategies = [MLSwingStrategy()]
        target_symbols = symbols or cfg["universe"]["symbols"]

        stop_event = threading.Event()
        engine = SwingPaperTradingEngine(cfg, strategies, broker, target_symbols, stop_event=stop_event)
        thread = threading.Thread(target=engine.run_forever, daemon=True)

        _paper_session["engine"] = engine
        _paper_session["thread"] = thread
        _paper_session["stop_event"] = stop_event
        thread.start()

    return {"started": True, "symbols": target_symbols}


@app.post("/api/paper/stop")
def paper_stop():
    with _lock:
        if _paper_session["engine"] is None:
            raise HTTPException(400, "No paper trading session is running.")
        _paper_session["stop_event"].set()
    return {"stopping": True}


@app.get("/api/paper/status")
def paper_status():
    with _lock:
        engine = _paper_session["engine"]
        if engine is None:
            return {"running": False}
        return engine.get_status()


@app.get("/api/paper/trades")
def paper_trades():
    path = Path(load_config()["paper_trading"]["swing_log_file"])
    full_path = path if path.is_absolute() else ROOT / path
    if not full_path.exists():
        return []
    with open(full_path) as f:
        return list(csv.DictReader(f))


# ============================== Live trading ==============================

@app.get("/api/live/status")
def live_status():
    """Read-only. Live trading is started/stopped exclusively from the
    terminal (python3 auto_run.py live) - no start/stop endpoint here, on
    purpose (see this file's module docstring).
    """
    cfg = load_config()
    path = Path(cfg["live_trading"]["log_file"])
    full_path = path if path.is_absolute() else ROOT / path

    trades = []
    if full_path.exists():
        with open(full_path) as f:
            trades = list(csv.DictReader(f))

    return {
        "enabled_in_config": cfg["live_trading"]["enabled"],
        "note": "Live trading has no API start/stop by design - use "
                "`python3 auto_run.py live` from a terminal you're actively watching.",
        "recent_trades": trades[-50:],
        "total_trades_logged": len(trades),
    }


# ============================== Static frontend ==============================

frontend_dir = Path(__file__).resolve().parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
