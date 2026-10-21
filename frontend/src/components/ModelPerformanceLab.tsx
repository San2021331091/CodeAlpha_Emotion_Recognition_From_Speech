"use client";

import React, { useState, useEffect, useRef } from "react";
import { BarChart3, RefreshCw, Trophy,Zap, Activity,Loader2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EMOTIONS_CONFIG } from "@/lib/constants";
import { formatPercent } from "@/lib/utils";

interface ModelPerformanceLabProps {
  apiBaseUrl: string;
}

export const ModelPerformanceLab: React.FC<ModelPerformanceLabProps> = ({ apiBaseUrl }):React.JSX.Element => {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isRetraining, setIsRetraining] = useState<boolean>(false);
  const [trainingJob, setTrainingJob] = useState<any>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const lastCompletedJob = useRef<string | null>(null);

  useEffect(():void => {
    fetchMetrics();
  }, []);

  useEffect(() => {
    let disposed = false;
    const poll = async ():Promise<void> => {
      try {
        const response = await fetch(`${apiBaseUrl}/speech-emotion/retrain/status`, { cache: "no-store" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || data.detail || "Could not check training status.");
        if (disposed) return;
        setTrainingJob(data);
        setIsRetraining(["queued", "running"].includes(data.status));
        setTrainingError(data.status === "failed" ? data.error || data.message : null);
        if (data.status === "completed" && lastCompletedJob.current !== data.job_id) {
          lastCompletedJob.current = data.job_id;
          await fetchMetrics();
        }
      } catch (error: any) {
        if (!disposed) setTrainingError(error.message);
      }
    };
    void poll();
    const timer = setInterval(poll, 5000);
    return () => { disposed = true; clearInterval(timer); };
  }, [apiBaseUrl]);

  const fetchMetrics = async ():Promise<void> => {
    try {
      setLoading(true);
      const res = await fetch(`${apiBaseUrl}/speech-emotion/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error("Failed to load metrics", err);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async ():Promise<void> => {
    if (!confirm("Start training on 50,000 original and augmented clips? Validation and test recordings stay held out. The current model is retained unless validation improves.")) return;
    try {
      setIsRetraining(true);
      setTrainingError(null);
      const res = await fetch(`${apiBaseUrl}/speech-emotion/retrain`, {
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || data.detail || "Could not start training.");
      setTrainingJob(data);
      setIsRetraining(["queued", "running"].includes(data.status));
    } catch (err: any) {
      setTrainingError(err.message);
      setIsRetraining(false);
    }
  };

  if (loading) {
    return (
      <Card className="glass-panel border-slate-800 p-12 flex flex-col items-center justify-center text-slate-400 gap-3">
        <Loader2 className="w-6 h-6 animate-spin text-indigo-400" />
        <p className="text-sm">Fetching model benchmark metrics & confusion matrices...</p>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card className="glass-panel border-slate-800 p-8 text-center text-slate-400">
        <p>No model metrics available. Please train models first.</p>
      </Card>
    );
  }

  const modelsList = Object.entries(metrics.models || {});
  const emotionsList = metrics.dataset_summary?.emotions_list || Object.keys(EMOTIONS_CONFIG);
  const confusionMatrix = metrics.confusion_matrix || {};
  const perEmotionMetrics = metrics.per_emotion_metrics || {};
  const topFeatures = metrics.top_features || [];
  const datasets = metrics.dataset_summary?.datasets || {};
  const sampleSplits = metrics.dataset_summary?.samples_by_dataset || {};
  const bestMetrics = metrics.models?.[metrics.best_model];
  const bestTestByDataset = bestMetrics?.test_by_dataset || {};

  return (
    <div className="space-y-6">
      {trainingJob?.message && trainingJob.status !== "idle" && (
        <p role="status" className="text-sm text-indigo-300">{trainingJob.message}</p>
      )}
      {trainingError && <p role="alert" className="text-sm text-rose-400 whitespace-pre-wrap">{trainingError}</p>}
      {/* Top Benchmark Summary */}
      <Card className="glass-panel border-slate-800">
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <CardTitle className="text-xl flex items-center gap-2 text-white">
                  <Trophy className="w-5 h-5 text-amber-400" />
                  Model Performance Benchmark Lab
                </CardTitle>
                <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/40">
                  Best: {metrics.best_model}
                </Badge>
              </div>
              <CardDescription className="text-slate-400">
                {metrics.dataset_summary?.name}: {metrics.dataset_summary?.total_samples} original recordings; {metrics.dataset_summary?.train_samples} original training, {metrics.dataset_summary?.validation_samples} validation, {metrics.dataset_summary?.test_samples} test. Training speakers are held out from evaluation; TESS validation and test use separate clips from the same speaker.
                {metrics.dataset_summary?.expanded_training_candidate && (
                  <span className="block mt-2 text-indigo-300">
                    Latest expanded candidate: {metrics.dataset_summary.expanded_training_candidate.total.toLocaleString()} training clips
                    ({metrics.dataset_summary.expanded_training_candidate.original.toLocaleString()} originals
                    {" + "}{metrics.dataset_summary.expanded_training_candidate.augmented.toLocaleString()} augmented).
                    {" "}{Object.entries(metrics.dataset_summary.expanded_training_candidate.samples_by_dataset)
                      .map(([source, count]) => `${source}: ${Number(count).toLocaleString()}`)
                      .join("; ")}.
                    {" RAVDESS training weight: "}{metrics.dataset_summary.expanded_training_candidate.ravdess_training_weight}x.
                    {" Validation macro F1: "}{formatPercent(metrics.dataset_summary.expanded_training_candidate.validation_macro_f1)}.
                    {metrics.dataset_summary.expanded_training_candidate.deployed ? " Deployed." : " Evaluation candidate; deployed metrics are shown below."}
                  </span>
                )}
                {metrics.dataset_summary?.speech_encoder && (
                  <span className="block mt-2 text-emerald-300">
                    WavLM emotion classifier: {metrics.dataset_summary.speech_encoder.training_originals.toLocaleString()} original training recordings
                    with a frozen pretrained speech encoder. Augmented CNN runs are reported separately.
                  </span>
                )}
                {metrics.dataset_summary?.balanced_training_candidate && (
                  <span className="block mt-2 text-indigo-300">
                    Balanced CNN candidate: 8,333 training examples per emotion
                    (49,998 total, including original and augmented audio).
                    Select it in the studio to compare predictions with Best.
                  </span>
                )}
                {metrics.dataset_summary?.effective_training_samples && (
                  <span className="block mt-2 text-indigo-300">
                    Expanded training: {metrics.dataset_summary.effective_training_samples.toLocaleString()} examples
                    ({" "}{metrics.dataset_summary.train_samples.toLocaleString()} original recordings
                    {" + "}{metrics.dataset_summary.augmented_training_samples.toLocaleString()} augmented clips).
                    Validation and test clips are original recordings.
                    {metrics.dataset_summary.training_augmentation?.samples_by_emotion && (
                      <span className="block mt-1">
                        {Object.entries(metrics.dataset_summary.training_augmentation.samples_by_emotion)
                          .map(([emotion, count]) => `${EMOTIONS_CONFIG[emotion]?.label || emotion}: ${Number(count).toLocaleString()}`)
                          .join(" · ")}
                      </span>
                    )}
                  </span>
                )}
              </CardDescription>
            </div>

            <div className="flex items-center gap-2">
              <Button
                onClick={handleRetrain}
                disabled={isRetraining}
                size="sm"
                className="bg-indigo-600 hover:bg-indigo-700 text-white flex items-center gap-2 text-xs"
              >
                {isRetraining ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Training Models...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-3.5 h-3.5" />
                    Train 50,000-clip CNN
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {Object.keys(datasets).length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
              {Object.entries(datasets).map(([source, details]: [string, any]) => {
                const split = {
                  train: sampleSplits.train?.[source],
                  validation: sampleSplits.validation?.[source],
                  test: sampleSplits.test?.[source],
                };
                const result = bestTestByDataset[source];
                return (
                  <div key={source} className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-slate-200">{source}</span>
                      {result && (
                        <span className="text-xs font-mono text-emerald-400">
                          test {(result.accuracy * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-[11px] text-slate-400">
                      Train {split.train ?? 0} · Validation {split.validation ?? 0} · Test {split.test ?? 0}
                    </p>
                    {details?.source && (
                      <p className="mt-1 text-[10px] text-slate-500">{details.source}</p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase text-slate-400 border-b border-slate-800 bg-slate-900/40">
                <tr>
                  <th className="px-4 py-3">Architecture</th>
                  <th className="px-4 py-3">Accuracy</th>
                  <th className="px-4 py-3">Precision (Macro)</th>
                  <th className="px-4 py-3">Recall (Macro)</th>
                  <th className="px-4 py-3">F1-Score</th>
                  <th className="px-4 py-3">ROC-AUC</th>
                  <th className="px-4 py-3">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                {modelsList.map(([name, score]: [string, any]) => {
                  const isBest = name === metrics.best_model;
                  return (
                    <tr
                      key={name}
                      className={`hover:bg-slate-900/50 transition-colors ${
                        isBest ? "bg-indigo-950/20 font-bold" : ""
                      }`}
                    >
                      <td className="px-4 py-3 font-sans flex items-center gap-2 text-slate-200">
                        {isBest && <Trophy className="w-3.5 h-3.5 text-amber-400 shrink-0" />}
                        <span>{name}</span>
                      </td>
                      <td className="px-4 py-3 text-emerald-400 font-semibold">
                        {(score.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-slate-300">
                        {(score.precision * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-slate-300">
                        {(score.recall * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-indigo-300 font-semibold">
                        {(score.f1_score * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-cyan-300">
                        {score.roc_auc.toFixed(4)}
                      </td>
                      <td className="px-4 py-3 text-slate-400">
                        {score.inference_latency_ms} ms
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* 2-Column: Confusion Matrix + Per-Emotion Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Interactive Confusion Matrix */}
        <Card className="glass-panel border-slate-800">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2 text-white">
              <Activity className="w-5 h-5 text-indigo-400" />
              Confusion Matrix Heatmap ({metrics.best_model})
            </CardTitle>
            <CardDescription className="text-slate-400">
              Rows: Actual Ground Truth Label • Columns: Model Predicted Emotion
            </CardDescription>
          </CardHeader>

          <CardContent className="overflow-x-auto">
            <div className="min-w-[420px]">
              <div
                className="grid gap-1 text-[11px] font-mono text-center mb-1"
                style={{ gridTemplateColumns: `minmax(4rem, auto) repeat(${emotionsList.length}, minmax(0, 1fr))` }}
              >
                <span className="text-slate-500 font-sans font-bold text-left">Act \ Pred</span>
                {emotionsList.map((emo: string) => (
                  <span key={emo} className="text-slate-400 truncate" title={emo}>
                    {emo.slice(0, 3).toUpperCase()}
                  </span>
                ))}
              </div>

              {emotionsList.map((actEmo: string) => (
                <div
                  key={actEmo}
                  className="grid gap-1 text-xs font-mono items-center mb-1"
                  style={{ gridTemplateColumns: `minmax(4rem, auto) repeat(${emotionsList.length}, minmax(0, 1fr))` }}
                >
                  <span className="text-slate-400 text-left font-sans truncate pr-1" title={actEmo}>
                    {actEmo.slice(0, 4)}
                  </span>
                  {emotionsList.map((predEmo: string) => {
                    const count = confusionMatrix[actEmo]?.[predEmo] ?? 0;
                    const isDiagonal = actEmo === predEmo;
                    const bgIntensity = count > 0 ? (isDiagonal ? "bg-emerald-600/80 text-white font-bold" : "bg-rose-500/60 text-white") : "bg-slate-900/60 text-slate-600";

                    return (
                      <div
                        key={predEmo}
                        className={`h-7 rounded flex items-center justify-center transition-all ${bgIntensity}`}
                        title={`Actual: ${actEmo}, Predicted: ${predEmo} (${count} samples)`}
                      >
                        {count}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Per-Emotion Breakdown Metrics */}
        <Card className="glass-panel border-slate-800">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2 text-white">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              Per-Emotion Classification Diagnostics
            </CardTitle>
            <CardDescription className="text-slate-400">
              Granular Precision, Recall, and F1-Scores across the shared emotion classes
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-3">
            {emotionsList.map((emo: string) => {
              const meta = EMOTIONS_CONFIG[emo] || EMOTIONS_CONFIG.neutral;
              const per = perEmotionMetrics[emo] || { precision: 0, recall: 0, f1_score: 0, support: 0 };
              const f1Percent = (per.f1_score * 100).toFixed(1);

              return (
                <div key={emo} className="p-2.5 rounded-lg border border-slate-800 bg-slate-900/40 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 font-medium text-slate-200">
                      <span>{meta.emoji}</span>
                      <span>{meta.label}</span>
                    </span>
                    <div className="flex items-center gap-3 font-mono text-[11px]">
                      <span className="text-slate-400">P: {(per.precision * 100).toFixed(0)}%</span>
                      <span className="text-slate-400">R: {(per.recall * 100).toFixed(0)}%</span>
                      <span className="text-indigo-400 font-bold">F1: {f1Percent}%</span>
                    </div>
                  </div>

                  <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${meta.barColor}`}
                      style={{ width: `${f1Percent}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      </div>

      {/* Top Acoustic Feature Ranking */}
      <Card className="glass-panel border-slate-800">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2 text-white">
            <Zap className="w-5 h-5 text-amber-400" />
            Top Salient Acoustic Feature Importances
          </CardTitle>
          <CardDescription className="text-slate-400">
            Acoustic coefficients with highest Gini impurity reduction / mutual information
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {topFeatures.slice(0, 10).map((f: any, idx: number) => (
              <div key={idx} className="p-3 rounded-xl border border-slate-800 bg-slate-900/60 space-y-1">
                <span className="text-[10px] uppercase font-mono text-slate-500">Rank #{idx + 1}</span>
                <span className="text-xs font-semibold text-slate-200 block truncate" title={f.feature}>
                  {f.feature}
                </span>
                <span className="text-xs font-mono text-amber-400 block font-bold">
                  {(f.importance * 100).toFixed(2)}% weight
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
