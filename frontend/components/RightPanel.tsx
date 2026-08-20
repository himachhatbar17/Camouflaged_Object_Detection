"use client";

import { Activity, BarChart } from "lucide-react";
import type { ModelInfo, PredictResponse } from "@/lib/api";

function statusDot(v: number, good: number, warn: number) {
  if (v >= good) return "high";
  if (v >= warn) return "medium";
  return "low";
}

export default function RightPanel({
  result,
  activeModel,
}: {
  result: PredictResponse | null;
  activeModel: ModelInfo | undefined;
}) {
  const rows = result
    ? [
        { name: "Foreground confidence", value: result.confidence, kind: statusDot(result.confidence, 0.8, 0.5) },
        { name: "Predicted coverage", value: result.coverage, kind: statusDot(1 - result.coverage, 0.6, 0.3) },
        { name: "Gate mean (RGB reliance)", value: result.gate_mean, kind: statusDot(result.gate_mean, 0.6, 0.3) },
      ]
    : [];

  const metrics = result?.reported_metrics ?? activeModel?.metrics ?? null;

  return (
    <aside className="fade-in flex flex-col gap-4">
      <div className="glass-card p-5">
        <div className="mb-3.5 flex items-center justify-between">
          <span className="flex items-center gap-2 text-[0.7rem] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            <Activity size={14} strokeWidth={1.8} /> Live Inference
          </span>
          {result && <span className="font-mono text-[0.6rem] text-[var(--text-muted)]">{result.inference_ms.toFixed(0)} ms</span>}
        </div>
        {rows.length === 0 ? (
          <p className="text-xs text-[var(--text-muted)]">Run inference to see live confidence breakdown.</p>
        ) : (
          <div className="flex flex-col gap-1">
            {rows.map((r) => (
              <div key={r.name} className="flex items-center justify-between rounded-xl px-3 py-2 hover:bg-[rgba(10,51,35,0.05)]">
                <span className="flex items-center gap-2.5 text-xs text-[var(--text-secondary)]">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{
                      background: r.kind === "high" ? "var(--good)" : r.kind === "medium" ? "var(--warn)" : "var(--bad)",
                    }}
                  />
                  {r.name}
                </span>
                <span className="flex items-center gap-2.5 font-mono text-xs">
                  {r.value.toFixed(3)}
                  <span className="h-1 w-12 overflow-hidden rounded-full bg-[rgba(10,51,35,0.09)]">
                    <span
                      className="block h-full rounded-full"
                      style={{
                        width: `${Math.round(r.value * 100)}%`,
                        background: "linear-gradient(90deg, var(--accent-blue), var(--accent-mint))",
                      }}
                    />
                  </span>
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="glass-card p-5">
        <div className="mb-3.5 flex items-center justify-between">
          <span className="flex items-center gap-2 text-[0.7rem] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            <BarChart size={14} strokeWidth={1.8} /> Reported Metrics
          </span>
          <span className="font-mono text-[0.6rem] text-[var(--text-muted)]">{activeModel?.label ?? ""}</span>
        </div>
        {!metrics ? (
          <p className="text-xs text-[var(--text-muted)]">Select a model to see its benchmark scores.</p>
        ) : (
          <div className="flex flex-col gap-3">
            {(["COD10K", "NC4K"] as const).map((ds) => (
              <div key={ds}>
                <div className="mb-1.5 text-[0.6rem] uppercase tracking-wide text-[var(--text-muted)]">{ds}</div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="metric-item">
                    <span className="text-[0.55rem] uppercase tracking-wide text-[var(--text-muted)]">MAE&darr;</span>
                    <span className="font-numbers text-base font-semibold">{metrics[ds].MAE.toFixed(3)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="text-[0.55rem] uppercase tracking-wide text-[var(--text-muted)]">S-measure&uarr;</span>
                    <span className="font-numbers text-base font-semibold">{metrics[ds].Sm.toFixed(3)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="text-[0.55rem] uppercase tracking-wide text-[var(--text-muted)]">wF&beta;&uarr;</span>
                    <span className="font-numbers text-base font-semibold">{metrics[ds].wFm.toFixed(3)}</span>
                  </div>
                  <div className="metric-item">
                    <span className="text-[0.55rem] uppercase tracking-wide text-[var(--text-muted)]">E-measure&uarr;</span>
                    <span className="font-numbers text-base font-semibold">{metrics[ds].Em.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
