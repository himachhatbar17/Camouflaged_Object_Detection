# ΔFusion / CamoNet — FastAPI Backend

## Setup
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn app.main:app --reload --port 8000
```

Checkpoints live in `checkpoints/` (already included) and are loaded once at
startup — all 6 Stage-4 variants (init/scratch × seed 42/123/2024).

## Endpoints
- `GET  /api/health` — liveness + loaded models
- `GET  /api/models` — list of models with their reported benchmark metrics
- `POST /api/predict?model=<id>&threshold=0.5` — multipart file upload, returns
  mask/overlay/heatmap/edge/gate PNGs (base64) + confidence + timing
- `GET  /api/results/summary` — baseline comparison tables, robustness, pseudo-ΔT quality
- `GET  /api/results/per-image?variant=&seed=&dataset=` — per-image metric rows

Interactive docs at `http://localhost:8000/docs`.
