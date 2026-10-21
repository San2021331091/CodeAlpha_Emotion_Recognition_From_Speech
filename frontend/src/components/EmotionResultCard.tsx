"use client";

import React, { useState } from "react";
import { Sparkles, Activity, BookmarkPlus, Check, X, Brain, Clock, HelpCircle } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EMOTIONS_CONFIG } from "@/lib/constants";
import { formatPercent, formatMs } from "@/lib/utils";

interface EmotionResultCardProps {
  prediction: any;
  expectedEmotion?: string | null;
  onSaveToHistory: (notes?: string) => Promise<void>;
  isSaving: boolean;
  hasSaved: boolean;
}

export const EmotionResultCard: React.FC<EmotionResultCardProps> = ({
  prediction,
  expectedEmotion,
  onSaveToHistory,
  isSaving,
  hasSaved,
}) => {
  const [notes, setNotes] = useState("");
  const [showNotesInput, setShowNotesInput] = useState(false);

  if (!prediction) return null;

  const topEmotion = prediction.top_emotion.toLowerCase();
  const meta = EMOTIONS_CONFIG[topEmotion] || EMOTIONS_CONFIG.neutral;
  const probabilities = prediction.probabilities || {};
  const valenceArousal = prediction.valence_arousal || { valence: 0, arousal: 0, quadrant: "Neutral" };

  // Calculate percentage position on 2D Circumplex grid: X: -1..1 -> 0..100%, Y: -1..1 -> 100..0%
  const circumplexX = Math.min(Math.max(((valenceArousal.valence + 1.0) / 2.0) * 100, 5), 95);
  const circumplexY = Math.min(Math.max(((1.0 - valenceArousal.arousal) / 2.0) * 100, 5), 95);

  const sortedEmotions = Object.entries(probabilities).sort((a: any, b: any) => b[1] - a[1]);

  return (
    <Card className={`glass-panel border-2 ${meta.borderColor} transition-all duration-300 shadow-2xl`}>
      <CardHeader className="pb-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className={`w-16 h-16 rounded-2xl ${meta.bgColor} border ${meta.borderColor} flex items-center justify-center text-4xl shadow-inner`}>
              {meta.emoji}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider font-semibold text-slate-400">
                  Primary Classification
                </span>
                <Badge className={meta.badgeClass}>{prediction.model_used}</Badge>
              </div>
              <h2 className={`text-3xl font-extrabold tracking-tight ${meta.textColor}`}>
                {meta.label}
              </h2>
            </div>
          </div>

          <div className="flex sm:flex-col items-end justify-between sm:justify-center border-t sm:border-t-0 border-slate-800 pt-3 sm:pt-0">
            <span className="text-xs text-slate-400">Model Confidence</span>
            <div className="flex items-baseline gap-1">
              <span className="text-3xl font-mono font-bold text-white">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <span className="text-[11px] text-slate-500 font-mono flex items-center gap-1 mt-0.5">
              <Clock className="w-3 h-3" />
              Inference: {formatMs(prediction.inference_time_ms)}
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {expectedEmotion && (
          <div
            role="status"
            className={`flex flex-wrap items-center gap-x-2 gap-y-1 rounded-xl border p-3 text-sm ${
              expectedEmotion.toLowerCase() === topEmotion
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-200"
            }`}
          >
            {expectedEmotion.toLowerCase() === topEmotion
              ? <Check className="h-4 w-4 shrink-0" />
              : <X className="h-4 w-4 shrink-0" />}
            <span>
              Benchmark ground truth: <strong>{EMOTIONS_CONFIG[expectedEmotion]?.label || expectedEmotion}</strong>
              {" · "}
              Model prediction: <strong>{meta.label}</strong>
              {" · "}
              {expectedEmotion.toLowerCase() === topEmotion ? "Match" : "Misclassified"}
            </span>
          </div>
        )}

        {/* Natural Language Acoustic Explanation */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex items-start gap-3">
          <Brain className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-xs font-semibold text-indigo-300 uppercase tracking-wider">
              Acoustic Feature Diagnostic
            </h4>
            <p className="text-sm text-slate-300 leading-relaxed">
              {prediction.explanation}
            </p>
          </div>
        </div>

        {/* 2-Column: Circumplex Affect Grid + Probability Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Russell's Circumplex Affect Map */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <h4 className="text-sm font-semibold text-white">Affect Circumplex (Valence-Arousal)</h4>
              </div>
              <span className="text-xs text-cyan-400 font-medium">{valenceArousal.quadrant}</span>
            </div>

            {/* 2D Coordinate Grid */}
            <div className="relative w-full aspect-square max-h-56 rounded-lg border border-slate-800 bg-slate-900/50 overflow-hidden flex items-center justify-center p-2">
              {/* Grid Lines */}
              <div className="absolute inset-x-0 top-1/2 border-t border-dashed border-slate-700/80" />
              <div className="absolute inset-y-0 left-1/2 border-l border-dashed border-slate-700/80" />

              {/* Quadrant Labels */}
              <span className="absolute top-2 left-2 text-[10px] text-rose-400/80 font-mono">Tense / Hostile</span>
              <span className="absolute top-2 right-2 text-[10px] text-emerald-400/80 font-mono">Excited / Elated</span>
              <span className="absolute bottom-2 left-2 text-[10px] text-blue-400/80 font-mono">Depressed / Melancholy</span>
              <span className="absolute bottom-2 right-2 text-[10px] text-cyan-400/80 font-mono">Calm / Serene</span>

              {/* Point Marker */}
              <div
                className="absolute w-5 h-5 rounded-full bg-white border-2 border-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.8)] -translate-x-1/2 -translate-y-1/2 transition-all duration-500 flex items-center justify-center text-xs"
                style={{ left: `${circumplexX}%`, top: `${circumplexY}%` }}
              >
                <div className="w-2 h-2 rounded-full bg-indigo-600 animate-ping" />
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-slate-400 mt-3 font-mono">
              <span>Valence: {valenceArousal.valence > 0 ? `+${valenceArousal.valence.toFixed(2)}` : valenceArousal.valence.toFixed(2)}</span>
              <span>Arousal: {valenceArousal.arousal > 0 ? `+${valenceArousal.arousal.toFixed(2)}` : valenceArousal.arousal.toFixed(2)}</span>
            </div>
          </div>

          {/* Probability Distribution */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 flex flex-col justify-between">
            <h4 className="text-sm font-semibold text-white mb-3">Emotion Probability Distribution</h4>

            <div className="space-y-2.5">
              {sortedEmotions.map(([emo, prob]: [string, any]) => {
                const conf = EMOTIONS_CONFIG[emo] || EMOTIONS_CONFIG.neutral;
                const percentage = (prob * 100).toFixed(1);

                return (
                  <div key={emo} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="flex items-center gap-1.5 font-medium text-slate-300">
                        <span>{conf.emoji}</span>
                        <span>{conf.label}</span>
                      </span>
                      <span className="font-mono text-slate-400">{percentage}%</span>
                    </div>

                    <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${conf.barColor}`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Save to History / Session Actions */}
        <div className="pt-4 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            {!showNotesInput ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowNotesInput(true)}
                className="text-xs border-slate-700 text-slate-300 hover:bg-slate-800"
              >
                + Add Session Notes
              </Button>
            ) : (
              <input
                type="text"
                placeholder="Enter notes (e.g. subject speaker identity, environment)..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 w-64 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            )}
          </div>

          <Button
            onClick={() => onSaveToHistory(notes)}
            disabled={isSaving || hasSaved}
            className={`text-xs flex items-center gap-2 ${
              hasSaved
                ? "bg-emerald-600 text-white"
                : "bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
            }`}
          >
            {hasSaved ? (
              <>
                <Check className="w-4 h-4" />
                Saved to Session History
              </>
            ) : (
              <>
                <BookmarkPlus className="w-4 h-4 text-indigo-400" />
                Save Result to Log
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
