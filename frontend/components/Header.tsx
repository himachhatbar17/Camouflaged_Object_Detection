"use client";

import { Play, BookOpen, UploadCloud, BarChart2, GitBranch, ShieldCheck } from "lucide-react";

const NAV = [
  { href: "#demo", label: "Live Demo", icon: UploadCloud },
  { href: "#benchmarks", label: "Benchmarks", icon: BarChart2 },
  { href: "#architecture", label: "Architecture", icon: GitBranch },
  { href: "#robustness", label: "Robustness", icon: ShieldCheck },
];

export default function Header({ onScrollToDemo }: { onScrollToDemo: () => void }) {
  return (
    <header className="glass-card fade-in grid grid-cols-1 items-center gap-3 px-7 py-4 md:grid-cols-3">
      <div className="flex items-center gap-3 justify-self-start">
        <span className="font-display text-2xl font-extrabold tracking-tight text-[var(--text-primary)]">
          <span className="font-light" style={{ color: "var(--accent-blue)" }}>
            &Delta;
          </span>
          Fusion
          <span className="status-dot ml-2 align-middle" />
        </span>
        <span
          className="pill px-4 py-1 text-[0.6rem] font-medium uppercase tracking-wider"
          style={{ background: "rgba(31,157,85,0.09)", color: "var(--good)", border: "1px solid rgba(31,157,85,0.16)" }}
        >
          Live Demo
        </span>
      </div>

      <nav className="flex flex-wrap items-center justify-center gap-1 justify-self-center">
        {NAV.map((n) => (
          <a key={n.href} href={n.href} className="nav-item !px-3.5 !py-2 text-xs">
            <n.icon size={15} strokeWidth={1.7} />
            {n.label}
          </a>
        ))}
      </nav>

      <div className="flex gap-2.5 justify-self-start md:justify-self-end">
        <a href="#architecture" className="btn">
          <BookOpen size={14} strokeWidth={1.8} /> Architecture
        </a>
        <button className="btn primary" onClick={onScrollToDemo}>
          <Play size={14} strokeWidth={1.8} /> Try it
        </button>
      </div>
    </header>
  );
}
