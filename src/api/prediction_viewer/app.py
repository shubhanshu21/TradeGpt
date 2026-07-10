"""
IRON ORACLE PREDICTION VIEWER (adapted from Kronos's webui pattern)
=====================================================================
Interactive candlestick chart: real history, multiple sampled future paths
(genuine uncertainty from generate_with_confidence), and — if a holdout
window is picked — the real actual outcome for visual comparison.

Unlike the original Kronos webui (which shows one deterministic OHLC
forecast), this shows several independently-sampled future paths at once,
since our generate_with_confidence() gives real path-to-path agreement
as a confidence signal, not just a single guess.
"""
import os
import sys
import json
import pickle
import warnings
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.hydra import build_kraken, init_kraken_hardware, generate_with_confidence
from data.preprocess import compute_indicators, build_feature_cols, apply_dls, tokenize_returns

app = Flask(__name__)
# Flask disables Jinja2 template auto-reload whenever debug=False, which means
# template edits (templates/index.html) silently don't take effect without a
# server restart. Force it on explicitly so template changes are picked up live.
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app)

DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

# Global state — loaded once via /api/load-model. context_window/forecast_steps
# are read from the checkpoint's own saved metadata (models/return_vocab.pkl),
# NOT hardcoded — different runs (e.g. --timeframe 1h) use different shapes,
# and a mismatch here breaks weight-loading (e.g. MLALayer's RoPE buffers are
# built for a specific sequence length).
_state = {"model": None, "bin_centers": None, "bin_edges": None, "loaded": False,
          "error": None, "context_window": 120, "forecast_steps": 15, "timeframe": "15m"}


def try_load_model():
    """Load the trained checkpoint + return-token vocabulary, if they exist."""
    ckpt_path = os.path.join(MODEL_DIR, "hydra_best.keras")
    vocab_path = os.path.join(MODEL_DIR, "return_vocab.pkl")

    if not os.path.exists(ckpt_path):
        _state["error"] = f"No trained checkpoint found at {ckpt_path}. Train the model first."
        return False
    if not os.path.exists(vocab_path):
        _state["error"] = f"No return-token vocabulary found at {vocab_path}. Train the model first."
        return False

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    ctx_win  = vocab.get("context_window", 120)
    forecast = vocab.get("forecast_steps", 15)

    init_kraken_hardware()
    n_feat = vocab.get("n_features", len(build_feature_cols()))
    model = build_kraken(n_features=n_feat, context_window=ctx_win, forecast_steps=forecast,
                          vocab_size=vocab["vocab_size"])
    model.load_weights(ckpt_path)
    _state["context_window"] = ctx_win
    _state["forecast_steps"] = forecast
    _state["timeframe"] = vocab.get("timeframe", "15m")

    _state["model"] = model
    _state["bin_centers"] = vocab["bin_centers"]
    _state["bin_edges"] = vocab["bin_edges"]
    _state["loaded"] = True
    _state["error"] = None
    return True


def load_master_parquet():
    """Load our real Delta Exchange BTCUSD candle cache, matching the loaded
    checkpoint's timeframe (15m/1h) — not hardcoded, since different runs use
    different cache files."""
    timeframe = _state.get("timeframe", "15m")
    path = os.path.join(DATA_DIR, f"BTCUSD_{timeframe}_history_master.parquet")
    df = pd.read_parquet(path)
    df = df.reset_index().rename(columns={"index": "timestamps", "timestamp": "timestamps"})
    if "timestamps" not in df.columns:
        df["timestamps"] = df.iloc[:, 0]
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill().bfill()
    return df


def prepare_context(df, end_idx, ctx_win=None):
    """Compute indicators + DLS scaling + tokens for a real window ending at end_idx."""
    ctx_win = ctx_win or _state["context_window"]
    # compute_indicators expects a DatetimeIndex (matching train.py's pipeline,
    # which reads parquet with its native index) — set it explicitly here since
    # this dataframe was reset_index()'d to expose a "timestamps" column for
    # charting/date-filtering elsewhere. Leaving "timestamps" as a plain column
    # breaks compute_indicators' final numeric .clip() over all columns.
    window_df = df.iloc[max(0, end_idx - ctx_win - 150):end_idx].copy()
    window_df = window_df.set_index("timestamps")
    df_feat = compute_indicators(window_df)
    features = build_feature_cols()
    data = df_feat[features].values.astype("float32")
    x_raw = data[-ctx_win:]
    x_scaled, local_mean, local_std = apply_dls(x_raw)

    close_prices = data[:, features.index("close")]
    raw_returns = np.diff(close_prices) / (close_prices[:-1] + 1e-9)
    raw_returns = np.concatenate([[0.0], raw_returns]).astype("float64")
    token_ids = tokenize_returns(raw_returns[-ctx_win:], _state["bin_edges"])

    return x_scaled, token_ids, local_mean, local_std, features.index("close")


def create_chart(hist_df, entry_ts, paths, mean_close_path, actual_df=None, timeframe_minutes=15):
    """
    Deliberately simple: one blue line for real recent price (easy to read,
    no candlestick literacy required), a green/red dashed line for the AI's
    guess, and (if known) a purple line for what actually happened. No
    per-sample spaghetti lines — confidence is shown separately as a plain
    percentage meter instead of asking the viewer to read line spread.
    """
    fig = go.Figure()

    # Force plain Python lists (not pandas Series/numpy arrays) for every
    # x/y — newer plotly (server-side, installed as 6.x) serializes numpy
    # arrays as compact {"dtype","bdata"} binary blobs by default, which an
    # older/mismatched plotly.js build on the page can't decode, silently
    # rendering nothing. Plain lists always serialize as ordinary JSON arrays
    # regardless of plotly.js version, so this can't break again on a CDN update.
    hist_x = [t.isoformat() for t in hist_df["timestamps"]]
    hist_y = [float(v) for v in hist_df["close"]]

    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, mode="lines",
        line=dict(color="#3b82f6", width=2), name="Real price (recent)",
        hovertemplate="%{x|%b %d, %I:%M %p}<br>$%{y:,.0f}<extra></extra>"))

    future_ts = pd.date_range(start=entry_ts + pd.Timedelta(minutes=timeframe_minutes),
                               periods=len(mean_close_path), freq=f"{timeframe_minutes}min")

    is_up = mean_close_path[-1] >= hist_df["close"].iloc[-1]
    pred_color = "#16a34a" if is_up else "#dc2626"

    # Connect the historical line directly into the prediction so it reads as
    # one continuous story, not two disconnected charts.
    bridge_x = [hist_x[-1]] + [t.isoformat() for t in future_ts]
    bridge_y = [hist_y[-1]] + [float(v) for v in mean_close_path]
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y, mode="lines+markers",
        line=dict(color=pred_color, width=3, dash="dash"), name="AI's guess",
        hovertemplate="%{x|%b %d, %I:%M %p}<br>$%{y:,.0f}<extra>AI's guess</extra>"))

    if actual_df is not None and len(actual_df) > 0:
        bridge_actual_x = [hist_x[-1]] + [t.isoformat() for t in actual_df["timestamps"]]
        bridge_actual_y = [hist_y[-1]] + [float(v) for v in actual_df["close"]]
        fig.add_trace(go.Scatter(
            x=bridge_actual_x, y=bridge_actual_y, mode="lines+markers",
            line=dict(color="#7c3aed", width=2), name="What really happened",
            hovertemplate="%{x|%b %d, %I:%M %p}<br>$%{y:,.0f}<extra>Actual</extra>"))

    # "NOW" marker — the single most important landmark on this chart: it's
    # the line between "this already happened" and "this is a guess." Without
    # it, a first-time viewer has no way to tell where history ends.
    fig.add_vline(x=hist_x[-1], line_width=1.5, line_dash="dot", line_color="#64748b",
                  annotation_text="NOW", annotation_position="top",
                  annotation_font=dict(color="#64748b", size=12))

    # Zoom the y-axis to the actual price movement, not down to $0 — BTC
    # moves by a few thousand dollars around an ~$80k base, so an axis
    # starting at zero squeezes all the real detail into a sliver at the top.
    all_prices = hist_y + bridge_y + (bridge_actual_y if actual_df is not None and len(actual_df) > 0 else [])
    pad = (max(all_prices) - min(all_prices)) * 0.15 or max(all_prices) * 0.01
    fig.update_yaxes(range=[min(all_prices) - pad, max(all_prices) + pad])

    fig.update_layout(title="Recent Price → AI's Guess vs Reality",
                       xaxis_title="Time", yaxis_title="Price (USD)",
                       template="plotly_white", height=560, showlegend=True,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=False, type="date", showspikes=True,
                      spikemode="across", spikecolor="#94a3b8", spikethickness=1)
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-status")
def model_status():
    if not _state["loaded"]:
        try_load_model()
    if _state["loaded"]:
        return jsonify({"available": True, "loaded": True, "message": "Iron Oracle model loaded.",
                         "context_length": _state["context_window"],
                         "forecast_steps": _state["forecast_steps"],
                         "timeframe": _state["timeframe"]})
    return jsonify({"available": False, "loaded": False, "message": _state["error"]})


@app.route("/api/load-model", methods=["POST"])
def load_model_route():
    ok = try_load_model()
    if not ok:
        return jsonify({"error": _state["error"]}), 400
    return jsonify({"success": True, "message": "Iron Oracle checkpoint loaded.",
                     "model_info": {"name": "Iron Oracle (KAT)",
                                    "context_length": _state["context_window"],
                                    "forecast_steps": _state["forecast_steps"],
                                    "timeframe": _state["timeframe"]}})


@app.route("/api/data-files")
def data_files():
    timeframe = _state.get("timeframe", "15m")
    name = f"BTCUSD_{timeframe}_history_master.parquet"
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return jsonify([])
    size = os.path.getsize(path)
    return jsonify([{"name": name, "path": path, "size": f"{size/1024/1024:.1f} MB"}])


@app.route("/api/load-data", methods=["POST"])
def load_data_route():
    try:
        df = load_master_parquet()
        price_cols = ["open", "high", "low", "close"]
        return jsonify({
            "success": True,
            "data_info": {
                "rows": len(df),
                "columns": list(df.columns),
                "start_date": df["timestamps"].min().isoformat(),
                "end_date": df["timestamps"].max().isoformat(),
                "price_range": {
                    "min": float(df[price_cols].min().min()),
                    "max": float(df[price_cols].max().max()),
                },
                "prediction_columns": price_cols + (["volume"] if "volume" in df.columns else []),
                "timeframe": _state.get("timeframe", "15m"),
            },
            "message": f"Loaded {len(df):,} real Delta Exchange BTCUSD candles."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_route():
    if not _state["loaded"] and not try_load_model():
        return jsonify({"error": _state["error"]}), 400

    try:
        data = request.get_json()
        pred_len = int(data.get("pred_len") or _state["forecast_steps"])
        n_samples = int(data.get("sample_count", 5))
        temperature = float(data.get("temperature", 1.0))
        top_p = float(data.get("top_p", 0.9))
        start_date = data.get("start_date")

        df = load_master_parquet()

        if start_date:
            start_dt = pd.to_datetime(start_date, utc=True)
            end_idx = int((df["timestamps"] >= start_dt).idxmax())
        else:
            end_idx = len(df) - pred_len  # leave room for an actual-outcome holdout by default

        if end_idx < _state["context_window"] + 150:
            return jsonify({"error": "Not enough history before this point."}), 400

        x_scaled, tok_ids, local_mean, local_std, t_close = prepare_context(df, end_idx)

        conf = generate_with_confidence(
            _state["model"], x_scaled, tok_ids, pred_len, _state["bin_centers"],
            local_mean, local_std, t_close,
            n_samples=n_samples, temperature=temperature, top_p=top_p)

        mean_path = []
        raw_close = float(x_scaled[-1, t_close] * local_std[t_close] + local_mean[t_close])
        for step in range(pred_len):
            step_closes = [p[step]["close"] for p in conf["paths"]]
            mean_path.append(float(np.mean(step_closes)))

        hist_df = df.iloc[end_idx - 120:end_idx]
        entry_ts = df["timestamps"].iloc[end_idx - 1]

        actual_df = None
        has_comparison = False
        if end_idx + pred_len <= len(df):
            actual_df = df.iloc[end_idx:end_idx + pred_len]
            has_comparison = True

        tf_minutes = 60 if _state["timeframe"] == "1h" else 15
        chart_json = create_chart(hist_df, entry_ts, conf["paths"], mean_path, actual_df,
                                   timeframe_minutes=tf_minutes)

        return jsonify({
            "success": True,
            "chart": chart_json,
            "confidence": {
                "up_fraction": conf["up_fraction"],
                "down_fraction": conf["down_fraction"],
                "agreement": conf["agreement"],
                "majority_direction": conf["majority_direction"],
            },
            "has_comparison": has_comparison,
            "message": f"Generated {n_samples} sampled paths, {pred_len} steps ahead. "
                       f"Majority direction: {conf['majority_direction']} "
                       f"({conf['agreement']*100:.0f}% agreement)."
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("Starting Iron Oracle Prediction Viewer...")
    try_load_model()
    app.run(debug=False, host="0.0.0.0", port=5000)
