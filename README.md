# ΔFusion — CamoNet Camouflaged Object Detection

Full-stack demo: Next.js frontend → FastAPI backend → CamoNet (PyTorch).

```
Frontend (Next.js)  --upload image-->  FastAPI Backend  --load CamoNet-->  Prediction
                                                                        --> Mask + Confidence + Heatmap
```

## What's included

- **`backend/`** — FastAPI server. Loads all 6 Stage-4 checkpoints (init/scratch ×
  seed 42/123/2024) at startup, serves `/api/predict`, `/api/models`, and
  `/api/results/*` (your published benchmark tables).
- **`frontend/`** — Next.js app styled after your reference dashboard design
  (dark glass cards, ΔFusion branding). Upload an image, pick a model, run
  inference, see the mask/overlay/heatmap/edge/gate outputs plus your
  benchmark tables.

## Run it — 2 terminals

**Terminal 1 — backend**
```bash
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Wait for `[startup] 6/6 models ready on cpu` (or `cuda` if you have a GPU).

**Terminal 2 — frontend**
```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:3000**

That's it — upload an image, pick a model from the dropdown, hit **Run Inference**.

## Latest UI revision

- Navigation moved into the top bar next to the ΔFusion logo; the old
  square nav panel is gone.
- Header decluttered — removed the Task / Device stat-chip row.
- Model Info panel cleaned up into a single-column label/value list and is
  now the first sidebar panel.
- Upload & Predict panel enlarged to fill the freed-up space.
- Prediction Output now shows 9 tiles in this order: **Input → Thermal →
  Overlay → Mask → Heatmap → Edge Map → Gate Map → Attention Map →
  Delta-T Map.**
  - **Thermal** is a clearly-labeled inferno-colorized RGB-derived proxy
    (Stage 4 is RGB-only deployment, so there's no real sensor feed).
  - **Heatmap** now colorizes the mask *probability* map in jet
    (blue→red = low→high foreground confidence) instead of raw attention
    noise, so it reads as a proper confidence heatmap.
  - **Gate Map** now uses a red→green diverging colormap (green = RGB
    reliance, red = thermal-path reliance), matching the convention from
    your own Stage-2 sanity figure.
  - **Attention Map** is the ΔT-guided self-attention, jet-overlaid on the
    RGB input (matches your notebook's fig5 "Attn. Heatmap (overlay)").
  - **Delta-T Map** is the pseudo-ΔT residual (scale 2), in a red/blue
    diverging colormap centered at zero, matching your fig3/fig6 style.
- Color palette replaced throughout (CSS variables in
  `frontend/app/globals.css`) with Midnight Orchid / Dusky Lilac / Iris
  Mist / Plum Blossom / Silver Wisteria.

## Notes

- `frontend/.env.local` points at `http://localhost:8000` by default. Change
  `NEXT_PUBLIC_API_BASE` there if you deploy the backend elsewhere.
- The backend runs on CPU by default; if you have a CUDA GPU with PyTorch
  installed for it, it's picked up automatically (`torch.cuda.is_available()`).
- All 6 checkpoints are included in `backend/checkpoints/` — no external
  downloads needed.
- `backend/app/model.py` is a faithful reconstruction of `CamoNet` from your
  training notebook — every Stage-4 checkpoint loads with `strict=True`
  (verified: zero missing/unexpected keys across all 6 files).
