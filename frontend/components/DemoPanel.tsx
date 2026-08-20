"use client";

import { useCallback, useRef, useState } from "react";
import { UploadCloud, Image as ImageIcon, Loader2, AlertCircle } from "lucide-react";
import type { ModelInfo, PredictResponse } from "@/lib/api";
import { predict } from "@/lib/api";

export default function DemoPanel({
  models,
  selectedModel,
  onSelectModel,
  onResult,
}: {
  models: ModelInfo[];
  selectedModel: string;
  onSelectModel: (id: string) => void;
  onResult: (r: PredictResponse | null, previewUrl: string | null) => void;
}) {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File | null) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file.");
      return;
    }
    setError(null);
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
    onResult(null, url);
  }, [onResult]);

  const runInference = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predict(file, selectedModel, threshold);
      onResult(res, preview);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inference failed.");
      onResult(null, preview);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div id="demo" className="glass-card fade-in p-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-display flex items-center gap-3 text-base font-bold text-[var(--text-primary)]">
          Upload &amp; Predict
          <span className="font-body text-[0.65rem] font-normal text-[var(--text-muted)]">
            RGB image &rarr; mask &middot; confidence &middot; heatmap
          </span>
        </h3>
        <select
          value={selectedModel}
          onChange={(e) => onSelectModel(e.target.value)}
          className="font-mono rounded-full border border-[var(--border-light)] bg-white/60 px-4 py-1.5 text-xs text-[var(--text-primary)] outline-none"
        >
          {models.length === 0 ? (
            <option value="" className="bg-[#FBF8E8]">
              Loading models&hellip;
            </option>
          ) : (
            models.map((m) => (
              <option key={m.id} value={m.id} className="bg-[#FBF8E8]">
                {m.label} {m.loaded ? "" : "(unavailable)"}
              </option>
            ))
          )}
        </select>
      </div>

      <div
        className={`dropzone ${dragActive ? "active" : ""} flex min-h-[320px] flex-col items-center justify-center gap-4 px-6 py-14 text-center`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          handleFile(e.dataTransfer.files?.[0] ?? null);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
      >
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt="preview" className="max-h-64 rounded-xl object-contain" />
        ) : (
          <UploadCloud size={36} strokeWidth={1.4} className="text-[var(--text-muted)]" />
        )}
        <div className="text-sm text-[var(--text-secondary)]">
          {file ? file.name : "Drag & drop an image, or click to browse"}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-5">
        <label className="flex items-center gap-3 text-xs text-[var(--text-secondary)]">
          Threshold
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-32 accent-[var(--accent-blue)]"
          />
          <span className="font-mono text-[var(--accent-blue)]">{threshold.toFixed(2)}</span>
        </label>

        <button
          className="btn primary ml-auto"
          disabled={!file || loading || !selectedModel}
          onClick={runInference}
        >
          {loading ? (
            <>
              <Loader2 size={14} className="animate-spin" /> Running inference&hellip;
            </>
          ) : !selectedModel ? (
            <>
              <Loader2 size={14} className="animate-spin" /> Loading models&hellip;
            </>
          ) : (
            <>
              <ImageIcon size={14} strokeWidth={1.8} /> Run Inference
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="mt-4 flex items-center gap-2 rounded-xl border border-red-400/20 bg-red-400/5 px-4 py-2.5 text-xs text-red-300">
          <AlertCircle size={14} /> {error}
        </div>
      )}
    </div>
  );
}
