const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface ModelMetrics {
  MAE: number;
  Sm: number;
  wFm: number;
  Em: number;
}

export interface ModelInfo {
  id: string;
  variant: "init" | "scratch";
  seed: number;
  label: string;
  loaded: boolean;
  best_Sm: number | null;
  metrics: { COD10K: ModelMetrics; NC4K: ModelMetrics } | null;
}

export interface ModelsResponse {
  default: string;
  models: ModelInfo[];
}

export interface PredictResponse {
  model: string;
  variant: string;
  seed: number;
  inference_ms: number;
  confidence: number;
  coverage: number;
  gate_mean: number;
  threshold: number;
  thermal_png: string;
  mask_png: string;
  overlay_png: string;
  heatmap_png: string;
  edge_png: string;
  gate_png: string;
  attn_png: string;
  deltat_png: string;
  reported_metrics: { COD10K: ModelMetrics; NC4K: ModelMetrics } | null;
}

export interface ResultsSummary {
  baselines: Record<string, unknown>[];
  ours_mean_std: Record<string, unknown>[];
  stage4_mean_std: Record<string, unknown>[];
  illumination_robustness: Record<string, unknown>[];
  pseudo_dt_quality: Record<string, unknown>[];
}

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function getHealth() {
  return fetchJSON<{ status: string; device: string; models_loaded: string[] }>("/api/health");
}

export function getModels() {
  return fetchJSON<ModelsResponse>("/api/models");
}

export function getResultsSummary() {
  return fetchJSON<ResultsSummary>("/api/results/summary");
}

export async function predict(file: File, model: string, threshold: number): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  const url = `${API_BASE}/api/predict?model=${encodeURIComponent(model)}&threshold=${threshold}`;
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export { API_BASE };
