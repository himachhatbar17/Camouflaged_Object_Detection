"""
FastAPI backend for the ΔFusion (CamoNet) camouflaged-object-detection demo.

Endpoints
---------
GET  /api/health              -> liveness + which models are loaded
GET  /api/models               -> list of available Stage-4 checkpoints + their metrics
POST /api/predict?model=...    -> run inference on an uploaded image
GET  /api/results/summary      -> published benchmark tables (baselines, ours, robustness)
GET  /api/results/per-image    -> per-image metric rows for a given variant/seed/dataset

All 6 Stage-4 checkpoints (init/scratch x seed 42/123/2024) are loaded once
at startup and kept in memory; /api/predict picks one by name.
"""
import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from .model import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, attn_to_heatmap, load_camonet

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"
TABLES_DIR = Path(__file__).resolve().parent / "tables"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="ΔFusion CamoNet API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Model registry -- loaded once at startup
# --------------------------------------------------------------------------
MODEL_SPECS = [
    {"id": "scratch_seed42", "file": "stage4_scratch_seed42.pth", "variant": "scratch", "seed": 42},
    {"id": "scratch_seed123", "file": "stage4_scratch_seed123.pth", "variant": "scratch", "seed": 123},
    {"id": "scratch_seed2024", "file": "stage4_scratch_seed2024.pth", "variant": "scratch", "seed": 2024},
    {"id": "init_seed42", "file": "stage4_init_seed42.pth", "variant": "init", "seed": 42},
    {"id": "init_seed123", "file": "stage4_init_seed123.pth", "variant": "init", "seed": 123},
    {"id": "init_seed2024", "file": "stage4_init_seed2024.pth", "variant": "init", "seed": 2024},
]
DEFAULT_MODEL_ID = "scratch_seed2024"  # best Sm/MAE across all 6 checkpoints

MODELS = {}          # id -> torch model (eval mode)
MODEL_META = {}       # id -> {variant, seed, best_Sm, metrics: {...}}


def _load_metrics(variant: str, seed: int):
    p = TABLES_DIR / f"metrics_stage4_{variant}_seed{seed}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@app.on_event("startup")
def load_all_models():
    for spec in MODEL_SPECS:
        ckpt_path = CKPT_DIR / spec["file"]
        if not ckpt_path.exists():
            print(f"[startup] WARNING missing checkpoint: {ckpt_path}")
            continue
        model, meta = load_camonet(str(ckpt_path), device=DEVICE, strict=True)
        MODELS[spec["id"]] = model
        MODEL_META[spec["id"]] = {
            "variant": spec["variant"],
            "seed": spec["seed"],
            "best_Sm": meta.get("best_Sm"),
            "metrics": _load_metrics(spec["variant"], spec["seed"]),
        }
        print(f"[startup] loaded {spec['id']} <- {ckpt_path.name}")
    print(f"[startup] {len(MODELS)}/{len(MODEL_SPECS)} models ready on {DEVICE}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return t


def _png_b64(arr_uint8: np.ndarray) -> str:
    """arr_uint8: HxW (grayscale) or HxWx3/4 (RGB/RGBA) uint8 array -> data URI."""
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _jet_rgb(h: np.ndarray) -> np.ndarray:
    """h: HxW float [0,1] -> HxWx3 float [0,1] jet colormap (blue->cyan->yellow->red).
    Standard 'hot region = high value' convention, used for the confidence
    heatmap and the attention overlay below."""
    h = np.clip(h, 0, 1)
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _colorize_heatmap(heat: np.ndarray) -> np.ndarray:
    """Prediction-confidence heatmap: mask probability -> HxWx3 uint8 jet image.
    Blue = low foreground probability, red = high -- directly readable as
    'how sure is the model this pixel is the camouflaged object'."""
    rgb = _jet_rgb(heat)
    return (rgb * 255).astype(np.uint8)


def _overlay_attention(orig_resized: np.ndarray, attn_heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """ΔT-guided self-attention heatmap, jet-colored and alpha-blended over the
    RGB input (matches the notebook's fig5 'Attn. Heatmap (overlay)' panel) --
    shows *where* the attention module is looking when it guides the fusion,
    distinct from the plain confidence heatmap above."""
    rgb = _jet_rgb(attn_heat)
    base = orig_resized.astype(np.float32) / 255.0
    a = np.clip(attn_heat, 0, 1)[..., None] * alpha
    blended = base * (1 - a) + rgb * a
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def _contrast_stretch(x: np.ndarray, floor: float = 0.02) -> np.ndarray:
    """Per-image min-max contrast stretch for DISPLAY ONLY. Some gate maps
    saturate close to a near-constant value across the whole image (a real,
    legitimate model behavior -- e.g. an RGB-only deployment model learning
    to consistently favor the RGB path); a raw [0,1] colorize of that looks
    like a flat, uninformative block even though there's real (small)
    spatial structure underneath. This stretches whatever range IS present
    to the full display range so that structure becomes visible, without
    touching the actual reported statistics (e.g. 'Gate mean') anywhere
    else in the response."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < floor:
        # Genuinely (near-)constant map: stretching noise wouldn't be honest.
        # Center it at mid-gray-equivalent instead of amplifying float noise.
        return np.full_like(x, 0.5)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _colorize_gate(gate: np.ndarray) -> np.ndarray:
    """Illumination gate -> HxWx3 uint8, red-yellow-green diverging colormap.
    Green = gate near 1 (model is relying on the RGB path), red = gate near 0
    (model is falling back on the pseudo-thermal / ΔT path) -- matches the
    'green=RGB, red=Thermal' convention from the training notebook's Stage-2
    sanity figure, so it reads the same way here."""
    g = np.clip(gate, 0, 1)
    r = np.clip(2 * (1 - g), 0, 1)
    gr = np.clip(2 * g, 0, 1)
    b = np.zeros_like(g) + 0.12 * (1 - np.abs(2 * g - 1))
    rgb = np.stack([r, gr, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _colorize_diverging_rdbu(x: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Delta-T (pseudo-thermal residual) map -> HxWx3 uint8, red-blue diverging
    colormap centered at 0 (matches the notebook's RdBu_r ΔT figures).
    Blue = cooler-than-local-mean, white = ~no residual, red = warmer -- the
    same convention used for both the real and pseudo ΔT maps in training.
    If the residual magnitude is genuinely tiny everywhere, dividing by its
    max would blow up float noise into a fake-looking pattern -- guard
    against that and show a flat neutral map instead, honestly."""
    m = float(np.max(np.abs(x)))
    if m < floor:
        return np.full(x.shape + (3,), 255, dtype=np.uint8)
    xn = np.clip(x / (m + 1e-8), -1, 1)  # [-1, 1]
    pos = np.clip(xn, 0, 1)
    neg = np.clip(-xn, 0, 1)
    r = 1 - neg * 0.55
    g = 1 - pos * 0.75 - neg * 0.75
    b = 1 - pos * 0.55
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _colorize_inferno(v: np.ndarray) -> np.ndarray:
    """Grayscale [0,1] -> HxWx3 uint8 approximate 'inferno' colormap
    (black -> purple -> orange -> pale yellow), used for the pseudo-thermal
    proxy image since a real thermal sensor isn't available at RGB-only
    deployment time."""
    v = np.clip(v, 0, 1)
    r = np.clip(v * 2.2, 0, 1)
    g = np.clip(v * 1.7 - 0.35, 0, 1) ** 1.1
    b = np.clip(0.6 - v * 1.3, 0, 1) * (1 - v) + np.clip((v - 0.75) * 4, 0, 1) * 0.35
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _mask_overlay(orig_resized: np.ndarray, prob: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """orig_resized: HxWx3 uint8, prob: HxW float [0,1] -> HxWx3 uint8 overlay."""
    binary = (prob > thresh).astype(np.float32)
    overlay = orig_resized.astype(np.float32).copy()
    tint = np.array([56, 214, 172], dtype=np.float32)  # accent-mint-ish
    alpha = 0.45 * binary[..., None]
    overlay = overlay * (1 - alpha) + tint * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "models_loaded": list(MODELS.keys())}


@app.get("/api/models")
def list_models():
    out = []
    for spec in MODEL_SPECS:
        meta = MODEL_META.get(spec["id"], {})
        out.append({
            "id": spec["id"],
            "variant": spec["variant"],
            "seed": spec["seed"],
            "label": f"{spec['variant'].capitalize()} · seed {spec['seed']}",
            "loaded": spec["id"] in MODELS,
            "best_Sm": meta.get("best_Sm"),
            "metrics": meta.get("metrics"),
        })
    return {"default": DEFAULT_MODEL_ID, "models": out}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_ID, description="Model id, see /api/models"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    if not model or not model.strip():
        model = DEFAULT_MODEL_ID
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown or unloaded model '{model}'. See /api/models.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    orig_resized = np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)

    x = _preprocess(img).to(DEVICE)
    net = MODELS[model]

    t0 = time.time()
    with torch.no_grad():
        mask_log, edge_log, pred_dt, gate, attn_weights = net.forward_s4(x)
        prob = torch.sigmoid(mask_log)[0, 0]  # (H, W)
        edge_prob = torch.sigmoid(edge_log)[0, 0]
        gate_map = gate[0, 0]

        # Attention heatmap: blend scales 2 and 3 (the two coarsest / most
        # semantically-concentrated scales -- empirically the ones that
        # actually carry per-position variance once the mean-over-M bug
        # above is fixed; the finer scales 0/1 are comparatively flat).
        downsamples = [4, 8, 16, 32]
        heat_scales = []
        for si in (2, 3):
            hw = IMG_SIZE // downsamples[si]
            heat_scales.append(attn_to_heatmap(attn_weights[si], (hw, hw), out_size=IMG_SIZE)[0])
        heat = torch.stack(heat_scales, dim=0).mean(dim=0)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)  # re-normalize the blend

        # Pseudo-ΔT map, calculated from where it's actually CONSUMED: pred_dt
        # is never used raw anywhere in the network -- it only ever passes
        # through DeltaTGuidedAttention's `bp` projection (384ch -> per-head
        # bias) before being added into the attention logits. A raw
        # channel-mean of the 384-ch pseudo_gen output (the old approach)
        # averages across whatever the network actually learned to read,
        # which is why it looked like noise. This replicates the exact
        # internal bp-projection step (avg_pool by the scale's sr_ratio,
        # then the same bp conv the attention module itself calls) at the
        # same two scales used for the Attention Map above, so both figures
        # tell a consistent story about the same underlying signal.
        deltat_scales = []
        for si in (2, 3):
            attn_mod = net.attn[si]
            bias_src = pred_dt[si]
            bs_r = F.avg_pool2d(bias_src, attn_mod.sr, attn_mod.sr, ceil_mode=True) if attn_mod.sr > 1 else bias_src
            bm = attn_mod.bp(bs_r).mean(dim=1)  # (B, Hr, Wr) -- mean over heads
            bm = F.interpolate(bm[:, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
            deltat_scales.append(bm)
        deltat = torch.stack(deltat_scales, dim=0).mean(dim=0)

    elapsed_ms = (time.time() - t0) * 1000

    prob_np = prob.cpu().numpy()
    edge_np = edge_prob.cpu().numpy()
    gate_np = gate_map.cpu().numpy()
    heat_np = heat.cpu().numpy()
    deltat_np = deltat.cpu().numpy()

    # Confidence: mean probability mass assigned to the predicted foreground region,
    # i.e. how "sure" the model is about the pixels it thinks belong to the object.
    fg = prob_np > threshold
    if fg.sum() > 0:
        confidence = float(prob_np[fg].mean())
    else:
        confidence = float(1.0 - prob_np.mean())  # confidently predicting "nothing here"
    coverage = float(fg.mean())  # fraction of image predicted as foreground

    # Pseudo-thermal proxy: RGB-only deployment has no real sensor, so this is
    # an inferno-colorized luminance proxy, clearly labeled as such in the UI.
    luminance = (orig_resized.astype(np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    thermal_png = _png_b64(_colorize_inferno(luminance))

    mask_png = _png_b64((prob_np * 255).astype(np.uint8))
    overlay_png = _png_b64(_mask_overlay(orig_resized, prob_np, threshold))
    heatmap_png = _png_b64(_colorize_heatmap(prob_np))
    edge_png = _png_b64((edge_np * 255).astype(np.uint8))

    gate_resized_f = np.array(
        Image.fromarray((gate_np * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    gate_png = _png_b64(_colorize_gate(_contrast_stretch(gate_resized_f)))

    attn_png = _png_b64(_overlay_attention(orig_resized, heat_np))
    deltat_png = _png_b64(_colorize_diverging_rdbu(deltat_np))

    meta = MODEL_META.get(model, {})
    return JSONResponse({
        "model": model,
        "variant": meta.get("variant"),
        "seed": meta.get("seed"),
        "inference_ms": round(elapsed_ms, 1),
        "confidence": round(confidence, 4),
        "coverage": round(coverage, 4),
        "gate_mean": round(float(gate_np.mean()), 4),
        "threshold": threshold,
        "thermal_png": thermal_png,
        "mask_png": mask_png,
        "overlay_png": overlay_png,
        "heatmap_png": heatmap_png,
        "edge_png": edge_png,
        "gate_png": gate_png,
        "attn_png": attn_png,
        "deltat_png": deltat_png,
        "reported_metrics": meta.get("metrics"),
    })


@app.get("/api/results/summary")
def results_summary():
    import pandas as pd

    def read_csv(name):
        p = TABLES_DIR / name
        if not p.exists():
            return []
        df = pd.read_csv(p)
        return json.loads(df.to_json(orient="records"))

    return {
        "baselines": read_csv("main_results_baselines.csv"),
        "ours_mean_std": read_csv("main_results_ours_mean_std.csv"),
        "stage4_mean_std": read_csv("stage4_results_mean_std.csv"),
        "illumination_robustness": read_csv("illumination_robustness.csv"),
        "pseudo_dt_quality": read_csv("pseudo_dt_quality.csv"),
    }


@app.get("/api/results/per-image")
def results_per_image(
    variant: str = Query(..., pattern="^(init|scratch)$"),
    seed: int = Query(...),
    dataset: str = Query(..., pattern="^(COD10K|NC4K)$"),
    limit: int = Query(200, ge=1, le=5000),
):
    import pandas as pd

    fname = f"perimage_stage4_{variant}_seed{seed}_{dataset}.csv"
    p = TABLES_DIR / fname
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No per-image table for {variant}/{seed}/{dataset}")
    df = pd.read_csv(p).head(limit)
    return json.loads(df.to_json(orient="records"))
