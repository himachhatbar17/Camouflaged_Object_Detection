"use client";

import { Info } from "lucide-react";

export default function Sidebar({ activeVariant, activeSeed }: { activeVariant?: string; activeSeed?: number }) {
  const rows: { label: string; value: string; highlight?: boolean }[] = [
    { label: "Architecture", value: "CamoNet", highlight: true },
    { label: "Attention", value: "\u0394T-Guided SRA" },
    { label: "Backbone", value: "ResNet-18" },
    { label: "Input", value: "352\u00d7352\u00d73" },
    { label: "Output", value: "352\u00d7352" },
    { label: "Deploy mode", value: "RGB-only" },
    { label: "Variant", value: activeVariant ?? "\u2014", highlight: true },
    { label: "Seed", value: activeSeed?.toString() ?? "\u2014" },
  ];

  return (
    <aside className="fade-in flex flex-col gap-4">
      <div className="glass-card p-5">
        <div className="mb-4 flex items-center gap-2 text-[0.7rem] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
          <Info size={14} strokeWidth={1.8} /> Model Info
        </div>
        <dl className="flex flex-col divide-y divide-white/[0.05]">
          {rows.map((r) => (
            <div key={r.label} className="flex items-center justify-between gap-3 py-2.5 first:pt-0 last:pb-0">
              <dt className="text-[0.65rem] uppercase tracking-wide text-[var(--text-muted)]">{r.label}</dt>
              <dd
                className="font-mono text-[0.8rem] font-medium whitespace-nowrap"
                style={{ color: r.highlight ? "var(--accent-blue)" : "var(--text-primary)" }}
              >
                {r.value}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="glass-card p-5">
        <div className="mb-3 text-[0.7rem] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
          Training Curriculum
        </div>
        <ol className="space-y-2 text-xs text-[var(--text-secondary)]">
          <li><span className="font-mono text-[var(--accent-blue)]">S1</span> &middot; LLVIP InfoNCE encoder pretrain</li>
          <li><span className="font-mono text-[var(--accent-mint)]">S2</span> &middot; FLIR illumination gate</li>
          <li><span className="font-mono text-[var(--accent-lavender)]">S3</span> &middot; VT5000 full RGB-T backbone</li>
          <li><span className="font-mono text-[var(--accent-peach)]">S4</span> &middot; COD10K RGB-only fine-tune</li>
        </ol>
      </div>
    </aside>
  );
}
