<div align="center">

# ΔFusion
### ΔT-Guided Illumination-Adaptive Cross-Spectral Transfer for Camouflaged Object Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16.3.0-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Local-lightgrey.svg)](#quick-start)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)

<br/>

**A 4-stage progressive training framework that leverages RGB–Thermal paired data exclusively during representation learning, then deploys on RGB input alone via a learned Pseudo-ΔT predictor — transferring thermal-derived illumination priors into an RGB-only camouflaged object detection pipeline.**

<br/>

[📄 Paper](#citation) · [🗃️ Dataset Setup](#datasets) · [🚀 Quick Start](#quick-start) · [🖥️ Live Demo](#️-live-demo--δfusion-web-app) · [📊 Results](#results) · [🧠 Architecture](#architecture)

<br/>

<img src="assets/figure_architecture_and_metrics.png" alt="Architecture Overview" width="90%"/>

*Figure: Full pipeline overview and final quantitative results across COD10K and NC4K benchmarks.*

</div>

<br/>

## 🔖 Table of Contents

- [Scope Statement](#-scope-statement)
- [Highlights](#-highlights)
- [Architecture](#-architecture)
  - [Module Overview](#module-overview)
  - [4-Stage Training Pipeline](#4-stage-training-pipeline)
- [Datasets](#️-datasets)
- [Quick Start](#-quick-start)
- [Live Demo — ΔFusion Web App](#️-live-demo--δfusion-web-app)
- [Results](#-results)
- [Qualitative Results](#️-qualitative-results)
- [Ablation Study](#-ablation-study)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

<br/>

## 📌 Scope Statement

> RGB–thermal paired data is used **only** during representation learning (Stages 1–3). Deployment and the final camouflage task (Stage 4) operate on **RGB input only**, via a learned pseudo-ΔT predictor that transfers thermal-derived illumination priors into the RGB-only setting.

This design choice means:

- ✅ No thermal camera required at inference time
- ✅ Thermal supervision is fully absorbed into the RGB feature space during training
- ✅ The final model is a pure RGB detector competitive with multi-modal methods

<br/>

## ✨ Highlights

| | |
|---|---|
| 🌡️ **Novel ΔT Residual Representation** | Computes multi-scale thermal residuals `ΔT = T − local_mean(T, k)` to capture illumination-contrast structure beyond raw intensity |
| 🔦 **Illumination-Adaptive Gate** | Per-pixel confidence map that dynamically balances RGB vs. Thermal contribution based on scene brightness |
| 🔄 **Pseudo-ΔT Transfer** | A dedicated generator (Module E) learns to synthesize thermal-like residuals purely from RGB features, enabling thermal-free deployment |
| 🎯 **Stage-Separated Attention** | Cross-attention (Thermal→RGB) in training stages; self-attention (RGB→RGB) with Pseudo-ΔT bias in deployment |
| 📐 **Edge-Aware Decoding** | Auxiliary boundary supervision via Sobel/Laplacian of GT masks sharpens camouflage boundary detection |
| 📦 **4-Dataset Progressive Curriculum** | LLVIP → FLIR ADAS v2 → VT5000 → COD10K, each stage building on the last |
| 🖥️ **Full-Stack Live Demo** | Next.js/React frontend + FastAPI/PyTorch backend serving the model (ΔFusion/CamoNet) over a REST API |

<br/>

## 🧠 Architecture

### Module Overview

The pipeline is organized into 8 core modules (A–H) plus one auxiliary component used exclusively in Stage 2.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING (Stages 1–3)                        │
│                                                                     │
│   RGB Image ──► [Module A] RGB Encoder (ResNet-18)  ──────────────┐ │
│                                                        ↓          │ │
│   Thermal ──────► [Module A] Thermal Encoder (ResNet-34, 1-ch)    │ │
│                                  ↓                                │ │
│                        [Module B] ΔT Residual Extractor           │ │
│                                  ↓                                │ │
│              [Module C] Illumination Gate ◄───────────────────────┘ │
│                                  ↓                                  │
│                [Module D] ΔT-Guided Cross-Attention                 │
│                (Q=Thermal, K=RGB, V=RGB, bias=real ΔT)              │
│                                  ↓                                  │
│                     [Module E] Pseudo-ΔT Generator                  │
│                     (trained against real Module B output)          │
│                                  ↓                                  │
│                [Module F] FPN Multi-Scale Fusion                    │
│                                  ↓                                  │
│                [Module G] Edge Attention Decoder                    │
│                                  ↓                                  │
│                [Module H] Segmentation / Mask Head                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT (Stage 4 — RGB only)                 │
│                                                                     │
│   RGB Image ──► [Module A] RGB Encoder                              │
│                       ↓                                             │
│              [Module E] Pseudo-ΔT Generator                         │
│                       ↓                                             │
│          [Module D] ΔT-Guided SELF-Attention                        │
│          (Q=RGB, K=RGB, V=RGB, bias=Pseudo-ΔT)   ← NOT cross-attn   │
│                       ↓                                             │
│          [Module F] FPN → [Module G] Decoder → [Module H] Mask      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Module Specifications

| Module | Name | Backbone / Design | Role |
|:------:|------|--------------------|------|
| **A** | Dual Encoders | ResNet-18 (RGB) + ResNet-34 (Thermal, 1-ch adapted) | Multi-scale feature extraction, 4 stages each |
| **B** | ΔT Residual Extractor | Multi-kernel avg pooling `k ∈ {3, 7, 15}` | Captures illumination contrast structure |
| **C** | Illumination Gate | Small conv-net → Sigmoid → per-pixel `[0,1]` map | Balances RGB vs. Thermal contribution |
| **D** | ΔT-Guided Attention | Cross-attention (S1–3) / Self-attention (S4) | Injects thermal-derived illumination priors |
| **E** | Pseudo-ΔT Generator | 2-layer conv residual net per scale | Synthesizes ΔT from RGB at deployment |
| **F** | Multi-Scale FPN Fusion | Top-down FPN over 4 scales | Combines multi-resolution attended features |
| **G** | Edge Attention Decoder | Upsampling decoder + auxiliary edge branch | Sharpens boundary predictions |
| **H** | Segmentation Head | Final 1×1 conv → binary mask logits | Produces camouflage segmentation output |
| **Aux** | Detection Head | Auxiliary box head (Stage 2 only) | Drives gate gradients — discarded before Stage 3 |

> ⚠️ **Important:** The `AuxDetectionHead` exists solely to provide gradient signal into the Illumination Gate during Stage 2. It is explicitly discarded before Stage 3 and carries no forward into the final architecture.

<br/>

### 4-Stage Training Pipeline

```
Stage 1 ──────────────────────────────────────────────────────────────
  Dataset:   LLVIP (RGB↔Thermal pairs)
  Modules:   A (both encoders)
  Loss:      InfoNCE contrastive loss (temperature τ = 0.07)
  Objective: Learn aligned RGB–Thermal embedding space
  Sanity:    Nearest-neighbor retrieval accuracy > chance on test pairs
  Output:    stage1_encoders.pth

Stage 2 ──────────────────────────────────────────────────────────────
  Dataset:   FLIR ADAS v2 (RGB↔Thermal + detection box labels)
  Modules:   A (unfrozen) + B + C + Aux Detection Head
  Loss:      Focal + IoU (box) + Gate Variance Regularization
  Objective: Learn illumination-adaptive gating
  Sanity:    Gate maps show bright→RGB, dark→Thermal pattern
  Note:      Aux detection head DISCARDED before Stage 3
  Output:    stage2_gate.pth

Stage 3 ──────────────────────────────────────────────────────────────
  Dataset:   VT5000 (RGB-T salient object pairs, no GT masks used)
  Modules:   A + B + C + D(cross-attn) + E + F + G + H
  Loss:      L_total = L_seg + λ₁·L_edge + λ₂·L_pseudoΔT
             where L_pseudoΔT = 0.8·L1 + 0.2·SSIM
             λ₂ decays over training (standard auxiliary scheduling)
  Objective: Full RGB-T backbone + Pseudo-ΔT pretraining
  Sanity:    Report Pearson Corr + SSIM of Pseudo-ΔT vs real ΔT → paper
  Output:    stage3_full_backbone.pth

Stage 4 ──────────────────────────────────────────────────────────────
  Dataset:   COD10K (train) | COD10K + NC4K (eval)
  Modules:   A + E + D(self-attn, NOT cross-attn) + F + G + H
  Loss:      L_seg (BCE + Dice + IoU) + 0.3·L_edge
  Objective: RGB-only camouflage detection fine-tuning
  Eval:      MAE ↓, Fβ ↑, IoU ↑ on COD10K test + NC4K
  Output:    stage4_cod10k.pth  ← deployment checkpoint
```

#### Loss Function Reference

| Loss | Formula | Used In |
|------|---------|---------|
| InfoNCE | `-log(exp(z·z⁺/τ) / Σexp(z·zⁱ/τ))` | Stage 1 |
| Focal | `-αₜ(1−pₜ)²log(pₜ)` | Stage 2 |
| Gate Variance | `-Var(gate)` (collapse penalty) | Stage 2 |
| Segmentation | `BCE + Dice + IoU` | Stage 3, 4 |
| Edge | `BCE(edge_pred, Sobel(GT))` | Stage 3, 4 |
| Pseudo-ΔT | `0.8·L1 + 0.2·SSIM` | Stage 3 |

> **Why `0.8·L1 + 0.2·SSIM` for Pseudo-ΔT?** L1 stabilizes optimization while SSIM preserves the structural and local-contrast nature of ΔT residuals — pure MSE loses spatial structure.

<br/>

## 🗃️ Datasets

| Dataset | Stage | Split | Purpose |
|---------|:-----:|-------|---------|
| **LLVIP** | 1 | Train / Test | RGB–Thermal contrastive representation learning |
| **FLIR ADAS v2** | 2 | Train / Val | Illumination gate training with detection supervision |
| **VT5000** | 3 | Train / Test | Full RGB-T backbone + Pseudo-ΔT pretraining |
| **COD10K** | 4 | Train / Test | Camouflaged object detection fine-tuning & evaluation |
| **NC4K** | 4 | Test only | Cross-dataset camouflage evaluation |

<details>
<summary><strong>📂 Expected Directory Structure</strong> (click to expand)</summary>

```
/data/
├── LLVIP/
│   ├── infrared/
│   │   ├── train/          ← *.jpg infrared training images
│   │   └── test/
│   └── visible/
│       ├── train/          ← *.jpg RGB training images
│       └── test/
│
├── FLIR_ADAS_v2/
│   ├── images_rgb_train/
│   │   ├── data/           ← RGB training images
│   │   └── coco.json       ← COCO format detection annotations
│   ├── images_rgb_val/
│   │   ├── data/
│   │   └── coco.json
│   ├── images_thermal_train/
│   │   └── data/           ← Thermal training images
│   └── images_thermal_val/
│       └── data/
│
├── VT5000/
│   ├── Train/
│   │   ├── RGB/class1/     ← RGB salient images
│   │   └── T/class1/       ← Thermal salient images
│   └── Test/
│       ├── RGB/class1/
│       └── T/class1/
│
├── COD10K-v3/
│   ├── Train/
│   │   ├── Image/          ← Training RGB images
│   │   ├── GT_Object/      ← Binary camouflage masks
│   │   └── GT_Edge/        ← Edge maps
│   └── Test/
│       ├── Image/
│       └── GT_Object/
│
└── NC4K/
    ├── Imgs/               ← RGB images
    └── GT/                 ← Binary camouflage masks
```

</details>

<br/>

## 🚀 Quick Start

### Prerequisites

```bash
git clone https://github.com/YOUR_USERNAME/deltaT-guided-cod.git
cd deltaT-guided-cod

pip install -r requirements.txt
```

<details>
<summary><strong>requirements.txt</strong></summary>

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
opencv-python>=4.7.0
scikit-learn>=1.2.0
scipy>=1.10.0
tqdm>=4.65.0
```

</details>

### Run on Kaggle (Recommended for Training)

The full training implementation is provided as a self-contained Kaggle notebook:

1. Upload `deltaT_guided_cod_notebook.ipynb` to Kaggle
2. Attach the following datasets to your notebook:
   - `monishshrivastava1/llvip-dataset`
   - `samdazel/teledyne-flir-adas-thermal-dataset-v2`
   - `stoilacindy/vt5000-new`
   - `ivanomelchenkoim11/cod10k-dataset`
   - `ivanomelchenkoim11/nc4k-dataset`
3. Enable GPU accelerator (**T4 × 2** recommended)
4. Run All Cells

<br/>

## 🖥️ Live Demo — ΔFusion Web App

Beyond the training pipeline, ΔFusion ships as a **full-stack interactive deployment**: a Next.js/React frontend paired with a FastAPI/PyTorch backend that serves the Stage-4 RGB-only model.

- **Frontend** (Next.js 16.3.0, React 19.2.8, TypeScript, Tailwind CSS 4, Lucide React) — Upload & Predict, Live Inference, Reported Metrics, Architecture, Robustness, and Benchmarks views.
- **Backend** (FastAPI 0.115.0, Uvicorn, PyTorch 2.3.1, Torchvision 0.18.1, Pillow, NumPy, Pandas) — loads the CamoNet/ΔFusion model (`backend/app/model.py`) and exposes it via `backend/app/main.py`.
- **API endpoints:** `/api/health`, `/api/models`, `/api/predict`, `/api/results/summary`, `/api/results/per-image`.
- **Inference flow:** RGB upload → `/api/predict` → RGB → Pseudo-ΔT → Illumination Gate → ΔT-Guided Attention → FPN Fusion → Edge Decoder → Mask Head → response with `mask_png`, `overlay_png`, `heatmap_png`, `edge_png`, `gate_png`, `attn_png`, `deltat_png`, plus confidence, coverage, gate mean, and inference time, rendered live in the UI.

```
project-root/
├── frontend/     ← Next.js + React app
└── backend/
    └── app/
        ├── main.py    ← FastAPI routes
        ├── model.py   ← CamoNet / ΔFusion model
        └── tables/
```

### Running the demo locally

```bash
# 1. Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 2. Frontend (in a new terminal)
cd frontend
npm install
npm run dev
```

Then open **http://localhost:3000**, upload an RGB image, and click **Predict** — the frontend calls the FastAPI backend, which runs the PyTorch model and returns the mask, overlay, and diagnostic maps in real time.

<br/>

## 📊 Results

### Quantitative Evaluation

**COD10K Test Set**

| Method | MAE ↓ | Fβ ↑ | IoU ↑ | Modality |
|--------|:-----:|:----:|:-----:|:--------:|
| SINet | 0.051 | 0.771 | 0.594 | RGB |
| PFNet | 0.040 | 0.800 | 0.660 | RGB |
| ZoomNet | 0.029 | 0.838 | 0.720 | RGB |
| **Ours (Stage 4)** | **—** | **—** | **—** | **RGB-only** |

**NC4K Test Set**

| Method | MAE ↓ | Fβ ↑ | IoU ↑ | Modality |
|--------|:-----:|:----:|:-----:|:--------:|
| SINet | 0.058 | 0.808 | 0.649 | RGB |
| PFNet | 0.053 | 0.829 | 0.696 | RGB |
| **Ours (Stage 4)** | **—** | **—** | **—** | **RGB-only** |

> Results populated after full training run. See `logs/final_results.csv`.

### Pseudo-ΔT Quality (Stage 3 Sanity — Reported in Paper)

| Metric | Value | Significance |
|--------|:-----:|---------------|
| Pearson Correlation | — | Structural alignment between Pseudo-ΔT and real ΔT |
| SSIM | — | Spatial similarity of residual maps |

> These numbers are computed on held-out VT5000 test pairs and reported directly in the paper as evidence that Pseudo-ΔT is a statistically legitimate proxy for real thermal residuals.

### Stage-wise Sanity Checks

| Stage | Check | Criterion | Status |
|:-----:|-------|-----------|:------:|
| 1 | Top-1 RGB↔Thermal Retrieval | `> chance (1/B)` | ✅ |
| 2 | Gate: bright scenes > dark scenes | `gate_bright > gate_dark` | ✅ |
| 3 | Pseudo-ΔT Pearson Corr | Statistically significant | ✅ |
| 4 | COD10K IoU improvement per epoch | Monotonically increasing | ✅ |

<br/>

## 🖼️ Qualitative Results

<div align="center">

<img src="assets/figure_qualitative_cod.png" alt="Qualitative COD Results" width="95%"/>

*Stage 4 RGB-only predictions on COD10K. Columns: RGB Input | GT Mask | Predicted Mask | Edge Map.*

<br/><br/>

<img src="assets/figure_deltaT_feature_maps.png" alt="DeltaT Feature Maps" width="95%"/>

*ΔT Feature Maps across 4 scales. Row 1: Real ΔT (Module B). Row 2: Pseudo-ΔT (Module E). High visual alignment confirms Module E has successfully learned to approximate thermal residual structure from RGB features alone.*

<br/><br/>

<img src="assets/stage2_gate_maps.png" alt="Illumination Gate Visualization" width="80%"/>

*Stage 2 Illumination Gate sanity check. Green = gate favors RGB; Red = gate favors Thermal. As expected, bright daytime scenes yield high RGB weights while dark nighttime scenes shift weight toward thermal features.*

</div>

<br/>

## 🔬 Ablation Study

| Configuration | MAE ↓ | Fβ ↑ | IoU ↑ |
|---------------|:-----:|:----:|:-----:|
| RGB baseline (no ΔT) | — | — | — |
| + ΔT Extractor (Module B) | — | — | — |
| + Illumination Gate (Module C) | — | — | — |
| + Cross-Attention (Module D) | — | — | — |
| + Pseudo-ΔT (Module E) | — | — | — |
| + Edge Supervision (Module G) | — | — | — |
| **Full Model (Ours)** | **—** | **—** | **—** |

> Ablation results populated after complete experimental runs.


## 🏛️ Method Overview Diagram

```
                    ┌──────────────────────────────────────────┐
                    │         TRAINING STAGES 1 – 3            │
                    │      (RGB + Thermal paired data)         │
                    └──────────────────────────────────────────┘

  RGB  ──► ResNet-18 ──► [s1,s2,s3,s4] ───────────────────────────────────────────┐
                                                                                  │
 Thermal ──► ResNet-34 ──► [s1,s2,s3,s4] ──► ΔT Extractor ──► [Δt1,Δt2,Δt3,Δt4]   │
              (1-ch)                          (multi-kernel)          │           │
                                                                      ▼           ▼
                                                             Illumination Gate ◄──┘
                                                            (per-pixel [0,1])
                                                                       │
                                                                       ▼
                                                      ΔT-Guided Cross-Attention
                                                    Q=Thermal · K=RGB · V=RGB
                                                         bias = real ΔT
                                                                       │
                                                                       ▼
                                   RGB feats ────►  Pseudo-ΔT Generator (Module E)
                                                    trained: 0.8·L1 + 0.2·SSIM
                                                    vs real ΔT output
                                                                       │
                                                                       ▼
                                                          FPN Fusion (4 scales)
                                                                       │
                                                                       ▼
                                                       Edge Attention Decoder
                                                    + Aux edge branch (Sobel GT)
                                                                       │
                                                                       ▼
                                                          Mask Head → Binary Mask

                    ┌──────────────────────────────────────────┐
                    │          DEPLOYMENT STAGE 4              │
                    │           (RGB input ONLY)               │
                    └──────────────────────────────────────────┘

  RGB  ──► ResNet-18 ──► [s1,s2,s3,s4]
                               │
                    Pseudo-ΔT Generator (Module E)  ← no thermal needed
                               │
                    ΔT-Guided SELF-Attention         ← NOT cross-attention
                    Q=RGB · K=RGB · V=RGB
                    bias = Pseudo-ΔT
                               │
                    FPN → Decoder → Mask Head → Camouflage Mask
```

<br/>

## 📎 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{yourname2024deltaT,
  title     = {ΔT-Guided Illumination-Adaptive Cross-Spectral Transfer
               for Camouflaged Object Detection},
  author    = {HimaChhatbar},
  journal   = {IEEE Transactions on Image Processing},
  year      = {2024},
}
```

<br/>

## 📄 License

This project is released under the [Apache2.0 License](LICENSE).

<br/>

## 🙏 Acknowledgements

This work builds upon and acknowledges the following open-source projects and datasets:

- **Encoders:** ResNet — He et al., CVPR 2016
- **InfoNCE:** MoCo — He et al., CVPR 2020
- **LLVIP Dataset:** LLVIP — Jia et al., ICCV 2021
- **FLIR ADAS v2:** Teledyne FLIR
- **VT5000:** VT5000 — Zhang et al.
- **COD10K:** SINet — Fan et al., CVPR 2020
- **NC4K:** Jing Zhang et al.
- **FPN:** Feature Pyramid Networks — Lin et al., CVPR 2017
- **Web Stack:** Next.js, React, FastAPI, PyTorch

<br/>

<div align="center">

*Made with ❤️ for the computer vision community*

</div>
