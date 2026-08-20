"use client";

import { useEffect, useRef, useState } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import DemoPanel from "@/components/DemoPanel";
import ResultsGrid from "@/components/ResultsGrid";
import RightPanel from "@/components/RightPanel";
import ArchitectureStrip from "@/components/ArchitectureStrip";
import BenchmarkSection from "@/components/BenchmarkSection";
import { getHealth, getModels, getResultsSummary } from "@/lib/api";
import type { ModelInfo, PredictResponse, ResultsSummary } from "@/lib/api";

export default function Home() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [summary, setSummary] = useState<ResultsSummary | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const demoRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth()
      .catch(() => setBackendError("Could not reach the backend API. Is it running on the configured NEXT_PUBLIC_API_BASE?"));

    getModels()
      .then((r) => {
        setModels(r.models);
        setSelectedModel(r.default);
      })
      .catch(() => setBackendError("Could not reach the backend API. Is it running on the configured NEXT_PUBLIC_API_BASE?"));

    getResultsSummary()
      .then(setSummary)
      .catch(() => {});
  }, []);

  const activeModel = models.find((m) => m.id === selectedModel);

  return (
    <div className="relative">
      <div className="bg-grid" />
      <div className="relative z-10 mx-auto flex max-w-[1440px] flex-col gap-5 px-6 py-5">
        <Header onScrollToDemo={() => demoRef.current?.scrollIntoView({ behavior: "smooth" })} />

        {backendError && (
          <div className="glass-card fade-in border-red-400/20 bg-red-400/5 p-4 text-sm text-red-300">
            {backendError} Start it with <code className="font-mono">uvicorn app.main:app --reload --port 8000</code> from the{" "}
            <code className="font-mono">backend/</code> folder.
          </div>
        )}

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[270px_1fr_320px]">
          <Sidebar activeVariant={activeModel?.variant} activeSeed={activeModel?.seed} />

          <main className="flex flex-col gap-5" ref={demoRef}>
            <DemoPanel
              models={models}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
              onResult={(r, p) => {
                setResult(r);
                setPreviewUrl(p);
              }}
            />
            <ResultsGrid result={result} previewUrl={previewUrl} />
          </main>

          <RightPanel result={result} activeModel={activeModel} />
        </div>

        <ArchitectureStrip />
        <BenchmarkSection summary={summary} />

        <footer className="py-6 text-center text-[0.65rem] text-[var(--text-muted)]">
          &Delta;Fusion &middot; CamoNet &middot; \u0394T-Guided Cross-Spectral Transfer for Camouflaged Object Detection
        </footer>
      </div>
    </div>
  );
}
