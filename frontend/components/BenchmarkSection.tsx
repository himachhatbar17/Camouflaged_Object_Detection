"use client";

import type { ResultsSummary } from "@/lib/api";

type Row = Record<string, unknown>;

function num(r: Row, k: string, d = 3) {
  const v = r[k];
  return typeof v === "number" ? v.toFixed(d) : "\u2014";
}

export default function BenchmarkSection({ summary }: { summary: ResultsSummary | null }) {
  if (!summary) {
    return (
      <div id="benchmarks" className="glass-card fade-in p-6 text-sm text-[var(--text-muted)]">
        Loading benchmark tables&hellip;
      </div>
    );
  }

  const baselineRows = summary.baselines.filter((r) => r["Dataset"] === "COD10K-test");
  const oursRows = summary.ours_mean_std;
  const robustRows = summary.illumination_robustness;
  const pdt = summary.pseudo_dt_quality[0];

  return (
    <div className="flex flex-col gap-6">
      <div id="benchmarks" className="glass-card fade-in p-6">
        <h3 className="font-display mb-1 text-base font-bold text-[var(--text-primary)]">Benchmark Comparison &mdash; COD10K-test</h3>
        <p className="mb-4 text-[0.65rem] text-[var(--text-muted)]">
          Published baselines vs. Ours (mean&plusmn;std over 3 seeds, Stage-3-initialized variant). Weighted
          F-measure uses &beta;&sup2;=0.3, the field-standard CODToolbox convention.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-xs">
            <thead>
              <tr className="text-left text-[0.6rem] uppercase tracking-wide text-[var(--text-muted)]">
                <th className="py-2 pr-4">Method</th>
                <th className="py-2 pr-4">MAE&darr;</th>
                <th className="py-2 pr-4">S-measure&uarr;</th>
                <th className="py-2 pr-4">wF&beta;&uarr;</th>
                <th className="py-2 pr-4">E-measure&uarr;</th>
              </tr>
            </thead>
            <tbody>
              {baselineRows.map((r, i) => (
                <tr key={i} className="border-t border-[var(--border-light)] text-[var(--text-secondary)]">
                  <td className="py-2 pr-4">{String(r["Method"])}</td>
                  <td className="py-2 pr-4 font-mono">{num(r, "MAE")}</td>
                  <td className="py-2 pr-4 font-mono">{num(r, "Sm")}</td>
                  <td className="py-2 pr-4 font-mono">{num(r, "wFm")}</td>
                  <td className="py-2 pr-4 font-mono">{num(r, "Em")}</td>
                </tr>
              ))}
              {oursRows
                .filter((r) => String(r["Dataset"]) === "COD10K-test")
                .map((r, i) => (
                  <tr key={`ours-${i}`} className="border-t border-[var(--border-glow)]" style={{ color: "var(--accent-blue)" }}>
                    <td className="py-2 pr-4 font-semibold">{String(r["Method"])}</td>
                    <td className="py-2 pr-4 font-mono">{String(r["MAE"])}</td>
                    <td className="py-2 pr-4 font-mono">{String(r["Sm"])}</td>
                    <td className="py-2 pr-4 font-mono">{String(r["wFm"])}</td>
                    <td className="py-2 pr-4 font-mono">{String(r["Em"])}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      <div id="robustness" className="glass-card fade-in p-6">
        <h3 className="font-display mb-1 text-base font-bold text-[var(--text-primary)]">Illumination Robustness &mdash; NC4K</h3>
        <p className="mb-4 text-[0.65rem] text-[var(--text-muted)]">
          The illumination gate learns to lean on brightness-invariant thermal priors from training, giving it a
          measure of resilience to perturbed lighting/noise at inference &mdash; even RGB-only.
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          {robustRows.map((r, i) => (
            <div key={i} className="metric-item items-center text-center">
              <span className="text-[0.55rem] uppercase tracking-wide text-[var(--text-muted)]">{String(r["Perturbation"])}</span>
              <span className="font-numbers text-lg font-semibold">{num(r, "Sm")}</span>
              <span className="text-[0.55rem] text-[var(--text-muted)]">S-measure</span>
            </div>
          ))}
        </div>
      </div>

      {pdt && (
        <div className="glass-card fade-in p-6">
          <h3 className="font-display mb-1 text-base font-bold text-[var(--text-primary)]">Pseudo-\u0394T Quality</h3>
          <p className="mb-4 text-[0.65rem] text-[var(--text-muted)]">
            How closely the RGB-only pseudo-\u0394T generator (Module E) matches the real thermal-derived \u0394T signal
            on the VT5000 held-out test set.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <div className="metric-item items-center text-center">
              <span className="text-[0.55rem] uppercase tracking-wide text-[var(--text-muted)]">Pearson Correlation</span>
              <span className="font-numbers text-xl font-semibold">{num(pdt, "Pearson_Corr", 4)}</span>
            </div>
            <div className="metric-item items-center text-center">
              <span className="text-[0.55rem] uppercase tracking-wide text-[var(--text-muted)]">SSIM</span>
              <span className="font-numbers text-xl font-semibold">{num(pdt, "SSIM", 4)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
