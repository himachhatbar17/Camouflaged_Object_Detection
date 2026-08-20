"use client";

const BLOCKS = [
  { name: "RGB", sub: "352\u00d7352\u00d73" },
  { name: "Encoder", sub: "ResNet-18" },
  { name: "Pseudo-\u0394T", sub: "RGB \u2192 \u0394T" },
  { name: "Illum. Gate", sub: "[0,1] per-pixel" },
  { name: "Self-Attn", sub: "\u0394T-guided SRA" },
  { name: "FPN", sub: "4-scale fusion" },
  { name: "Decoder", sub: "edge-attention" },
  { name: "Mask Head", sub: "352\u00d7352" },
];

export default function ArchitectureStrip() {
  return (
    <div id="architecture" className="glass-card fade-in p-6">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-display flex items-center gap-3 text-base font-bold text-[var(--text-primary)]">
          Model Architecture
          <span className="font-body text-[0.65rem] font-normal text-[var(--text-muted)]">
            CamoNet &middot; \u0394T-guided cross-spectral transfer
          </span>
        </h3>
        <span className="text-[0.6rem] text-[var(--text-muted)]">Stage 4 deployment path (RGB-only)</span>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-1.5 py-2">
        {BLOCKS.map((b, i) => (
          <div key={b.name} className="flex items-center gap-1.5">
            <div className="arch-block">
              {b.name}
              <span className="mt-0.5 block text-[0.55rem] font-normal text-[var(--text-muted)]" style={{ fontFamily: "var(--font-body)" }}>
                {b.sub}
              </span>
            </div>
            {i < BLOCKS.length - 1 && <span className="text-lg font-light text-[var(--text-muted)]">&rarr;</span>}
          </div>
        ))}
      </div>
      <p className="mt-4 text-center text-[0.65rem] italic text-[var(--text-muted)]">
        Stages 1&ndash;3 train with paired RGB+Thermal data (LLVIP &rarr; FLIR &rarr; VT5000). Stage 4 fine-tunes on COD10K
        RGB-only, replacing real \u0394T with a predicted pseudo-\u0394T so deployment never needs a thermal sensor.
      </p>
    </div>
  );
}
