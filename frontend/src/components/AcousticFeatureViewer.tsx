"use client";

import React from "react";
import { Sliders, AudioWaveform as WaveformIcon, Music2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface AcousticFeatureViewerProps {
  summary: any;
}

const CHROMA_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

export const AcousticFeatureViewer: React.FC<AcousticFeatureViewerProps> = ({ summary }) => {
  if (!summary) return null;

  const mfccMeans = summary.mfcc_means || [];
  const chromaMeans = summary.chroma_means || [];

  return (
    <Card className="glass-panel border-slate-800">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl flex items-center gap-2 text-white">
              <Sliders className="w-5 h-5 text-indigo-400" />
              Acoustic Feature Extraction Spectrum
            </CardTitle>
            <CardDescription className="text-slate-400">
              Low-level and prosodic speech signal representations extracted via librosa
            </CardDescription>
          </div>
          <Badge variant="outline" className="text-xs bg-slate-800 text-slate-300 border-slate-700">
            231 Acoustic Dimensions
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Core Metric Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/60">
            <span className="text-[11px] text-slate-400 block">Pitch (F0 Mean)</span>
            <span className="text-lg font-mono font-bold text-indigo-300">
              {summary.pitch_f0_mean} Hz
            </span>
            <span className="text-[10px] text-slate-500 block">±{summary.pitch_f0_std} Hz dev</span>
          </div>

          <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/60">
            <span className="text-[11px] text-slate-400 block">RMS Energy</span>
            <span className="text-lg font-mono font-bold text-emerald-300">
              {summary.rms_energy_mean}
            </span>
            <span className="text-[10px] text-slate-500 block">Acoustic intensity</span>
          </div>

          <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/60">
            <span className="text-[11px] text-slate-400 block">Zero-Crossing Rate</span>
            <span className="text-lg font-mono font-bold text-amber-300">
              {summary.zero_crossing_rate_mean}
            </span>
            <span className="text-[10px] text-slate-500 block">Noisiness / Voicing</span>
          </div>

          <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/60">
            <span className="text-[11px] text-slate-400 block">Spectral Centroid</span>
            <span className="text-lg font-mono font-bold text-cyan-300">
              {summary.spectral_centroid_mean} Hz
            </span>
            <span className="text-[10px] text-slate-500 block">Timbral brightness</span>
          </div>

          <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/60">
            <span className="text-[11px] text-slate-400 block">Spectral Rolloff</span>
            <span className="text-lg font-mono font-bold text-purple-300">
              {summary.spectral_rolloff_mean} Hz
            </span>
            <span className="text-[10px] text-slate-500 block">85% energy cutoff</span>
          </div>

          <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/60">
            <span className="text-[11px] text-slate-400 block">Duration & Tempo</span>
            <span className="text-lg font-mono font-bold text-rose-300">
              {summary.duration_seconds}s
            </span>
            <span className="text-[10px] text-slate-500 block">{summary.tempo} BPM</span>
          </div>
        </div>

        {/* MFCC Coefficient Spectrum (13 bins) */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <WaveformIcon className="w-4 h-4 text-indigo-400" />
              Mel-Frequency Cepstral Coefficients (MFCC 1–13)
            </h4>
            <span className="text-[11px] text-slate-500 font-mono">dB / Cepstral Units</span>
          </div>

          <div className="grid grid-cols-13 gap-1.5 h-24 items-end pt-4">
            {mfccMeans.map((val: number, i: number) => {
              const normalizedHeight = Math.min(Math.max(Math.abs(val) * 1.5, 8), 100);
              const isPositive = val >= 0;

              return (
                <div key={i} className="flex flex-col items-center gap-1 h-full justify-end">
                  <div
                    className={`w-full rounded-t transition-all ${
                      isPositive ? "bg-indigo-500" : "bg-purple-500"
                    }`}
                    style={{ height: `${normalizedHeight}%` }}
                    title={`MFCC-${i + 1}: ${val}`}
                  />
                  <span className="text-[9px] text-slate-500 font-mono">{i + 1}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Chroma Pitch Class Distribution (12 Semitones) */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Music2 className="w-4 h-4 text-cyan-400" />
              Chroma Pitch Class Profile (12-TET Harmonic Spectrum)
            </h4>
            <span className="text-[11px] text-slate-500 font-mono">Relative Pitch Energy</span>
          </div>

          <div className="grid grid-cols-12 gap-1.5 h-20 items-end pt-2">
            {chromaMeans.map((val: number, i: number) => {
              const heightPercent = Math.min(Math.max(val * 100, 10), 100);

              return (
                <div key={i} className="flex flex-col items-center gap-1 h-full justify-end">
                  <div
                    className="w-full rounded-t bg-cyan-500/80 hover:bg-cyan-400 transition-all"
                    style={{ height: `${heightPercent}%` }}
                    title={`${CHROMA_NOTES[i]}: ${val}`}
                  />
                  <span className="text-[9px] text-slate-400 font-mono">{CHROMA_NOTES[i]}</span>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
