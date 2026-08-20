"use client";

import type { PredictResponse } from "@/lib/api";

export default function ResultsGrid({
  result,
  previewUrl,
}: {
  result: PredictResponse | null;
  previewUrl: string | null;
}) {
  if (!previewUrl) return null;

  const tiles: { label: string; src: string | undefined; sub: string; legend?: string }[] = [
    { label: "Input", src: previewUrl, sub: "352\u00d7352 RGB" },
    {
      label: "Thermal",
      src: result?.thermal_png,
      sub: "pseudo-thermal proxy",
      legend: "RGB-derived \u2014 no sensor at deploy time",
    },
    { label: "Overlay", src: result?.overlay_png, sub: "mask over image" },
    { label: "Mask", src: result?.mask_png, sub: "sigmoid probability" },
    {
      label: "Heatmap",
      src: result?.heatmap_png,
      sub: "confidence heatmap",
      legend: "blue = low, red = high foreground probability",
    },
    { label: "Edge Map", src: result?.edge_png, sub: "edge branch" },
    {
      label: "Gate Map",
      src: result?.gate_png,
      sub: "illumination gate",
      legend: "green = RGB reliance, red = thermal-path reliance",
    },
    {
      label: "Attention Map",
      src: result?.attn_png,
      sub: "\u0394T-guided self-attn.",
      legend: "warm overlay = where attention concentrates",
    },
    {
      label: "Delta-T Map",
      src: result?.deltat_png,
      sub: "learned \u0394T attention bias",
      legend: "blue = suppresses attention, red = boosts attention at that location",
    },
  ];

  return (
    <div className="glass-card fade-in p-6">
      <div className="mb-5 flex items-center justify-between">
        <h3 className="font-display text-base font-bold text-[var(--text-primary)]">Prediction Output</h3>
        {result && (
          <span className="font-mono text-[0.65rem] text-[var(--text-muted)]">
            {result.inference_ms.toFixed(0)} ms &middot; {result.variant} &middot; seed {result.seed}
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        {tiles.map((t) => (
          <div key={t.label} className="rounded-2xl border border-[var(--border-light)] bg-white/50 p-2.5">
            <div className="mb-2 flex aspect-square items-center justify-center overflow-hidden rounded-xl bg-[rgba(10,51,35,0.05)]">
              {t.src ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={t.src} alt={t.label} className="h-full w-full object-cover" />
              ) : (
                <span className="text-[0.6rem] text-[var(--text-muted)]">run inference&hellip;</span>
              )}
            </div>
            <div className="text-xs font-medium text-[var(--text-primary)]">{t.label}</div>
            <div className="text-[0.6rem] text-[var(--text-muted)]">{t.sub}</div>
            {t.legend && <div className="mt-1 text-[0.58rem] leading-tight text-[var(--text-secondary)]">{t.legend}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}
