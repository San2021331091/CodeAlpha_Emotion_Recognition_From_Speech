"use client";

import React, { useState, useEffect } from "react";
import {
  Play,
  Pause,
  Sparkles,
  Loader2,
  Music,
  Database,
  Globe,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { EMOTIONS_CONFIG } from "@/lib/constants";

interface BenchmarkSample {
  id: string;
  title: string;
  emotion: string;
  dataset?: string;
  speaker: string;
  language?: string;
  duration_seconds: number;
  description: string;
  filename: string;
}

interface BenchmarkSamplesProps {
  onAnalyzeAudio: (
    audioBlob: Blob,
    filename: string,
    modelName: string,
    expectedEmotion?: string,
  ) => Promise<void>;
  isAnalyzing: boolean;
  selectedModel: string;
  apiBaseUrl: string;
}

interface DataSetType {
  ravdess: string;
  tess: string;
  emodb: string;
}

function normalizeSample(sample: BenchmarkSample): BenchmarkSample {
  const corpus = sample.id.split("_")[0].toLowerCase();
  const datasets: DataSetType = {
    ravdess: "RAVDESS",
    tess: "TESS",
    emodb: "EMO-DB",
  };
  const dataset =
    sample.dataset?.trim().toUpperCase() || datasets[corpus] || "Unknown";
  const dataSetLanguageMap: Record<string, string> = {
    RAVDESS: "English",
    TESS: "English (Canada)",
    "EMO-DB": "German",
  };
  const language =
    sample.language?.trim() || dataSetLanguageMap[dataset] || "Unknown";
  return { ...sample, dataset, language };
}

export const BenchmarkSamples: React.FC<BenchmarkSamplesProps> = ({
  onAnalyzeAudio,
  isAnalyzing,
  selectedModel,
  apiBaseUrl,
}): React.JSX.Element => {
  const [samples, setSamples] = useState<BenchmarkSample[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>("all");
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(
    null,
  );

  useEffect((): void => {
    fetchSamples();
  }, []);

  const fetchSamples = async (): Promise<void> => {
    try {
      setLoading(true);
      const res = await fetch(`${apiBaseUrl}/speech-emotion/samples`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data.map(normalizeSample));
      }
    } catch (err) {
      console.error("Failed to load benchmark samples", err);
    } finally {
      setLoading(false);
    }
  };

  const playSample = (sampleId: string): void => {
    if (activeSampleId === sampleId && isPlaying) {
      audioElement?.pause();
      setIsPlaying(false);
      return;
    }

    if (audioElement) {
      audioElement.pause();
    }
    const audio = new Audio(`${apiBaseUrl}/speech-emotion/sample/${sampleId}`);
    audio.onended = () => setIsPlaying(false);
    audio.play();
    setAudioElement(audio);
    setActiveSampleId(sampleId);
    setIsPlaying(true);
  };

  const handleTestSample = async (sample: BenchmarkSample): Promise<void> => {
    try {
      const res = await fetch(
        `${apiBaseUrl}/speech-emotion/sample/${sample.id}`,
      );
      const blob = await res.blob();
      await onAnalyzeAudio(
        blob,
        sample.filename,
        selectedModel,
        sample.emotion,
      );
    } catch (err) {
      alert("Failed to load sample audio for inference.");
    }
  };

  const filteredSamples = samples.filter((s) => {
    if (selectedDataset === "all") return true;
    return s.dataset?.toLowerCase() === selectedDataset.toLowerCase();
  });
  const datasetCounts = {
    all: samples.length,
    ravdess: samples.filter(
      (s) => (s.dataset || "").toUpperCase() === "RAVDESS",
    ).length,
    tess: samples.filter((s) => (s.dataset || "").toUpperCase() === "TESS")
      .length,
    "emo-db": samples.filter(
      (s) => (s.dataset || "").toUpperCase() === "EMO-DB",
    ).length,
  };

  return (
    <Card className="glass-panel border-slate-800">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="text-xl flex items-center gap-2 text-white">
              <Music className="w-5 h-5 text-indigo-400" />
              Multi-Corpus Benchmark Audio Hub
            </CardTitle>
            <CardDescription className="text-slate-400">
              Held-out test speech recordings from <strong>RAVDESS</strong>{" "}
              (North American English), <strong>TESS</strong> (Canadian
              English), and <strong>EMO-DB</strong> (Berlin German). Emotion
              badges show dataset labels; model predictions can differ.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className="text-xs bg-indigo-500/10 text-indigo-300 border-indigo-500/30"
            >
              3 Standardized Datasets
            </Badge>
            <Badge
              variant="outline"
              className="text-xs bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
            >
              {samples.length} Test Audios
            </Badge>
          </div>
        </div>

        {/* Dataset Filter Tabs */}
        <div className="flex flex-wrap items-center gap-2 pt-4 border-t border-slate-800/80">
          <span className="text-xs text-slate-400 mr-1 flex items-center gap-1">
            <Database className="w-3.5 h-3.5 text-indigo-400" />
            Filter Corpus:
          </span>

          <button
            onClick={() => setSelectedDataset("all")}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
              selectedDataset === "all"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
            }`}
          >
            All Corpora ({datasetCounts.all})
          </button>

          <button
            onClick={() => setSelectedDataset("ravdess")}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
              selectedDataset === "ravdess"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
            }`}
          >
            RAVDESS ({datasetCounts.ravdess})
          </button>

          <button
            onClick={() => setSelectedDataset("tess")}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
              selectedDataset === "tess"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
            }`}
          >
            TESS ({datasetCounts.tess})
          </button>

          <button
            onClick={() => setSelectedDataset("emo-db")}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
              selectedDataset === "emo-db"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
            }`}
          >
            EMO-DB ({datasetCounts["emo-db"]})
          </button>
        </div>
      </CardHeader>

      <CardContent>
        {loading ? (
          <div className="flex items-center justify-center p-12 text-slate-400 gap-2">
            <Loader2 className="w-5 h-5 animate-spin" />
            Loading benchmark dataset...
          </div>
        ) : filteredSamples.length === 0 ? (
          <div className="text-center p-8 text-slate-400 text-xs">
            No audio samples match the selected dataset filter.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredSamples.map((sample) => {
              const meta =
                EMOTIONS_CONFIG[sample.emotion] || EMOTIONS_CONFIG.neutral;
              const isCurrentPlaying =
                activeSampleId === sample.id && isPlaying;
              const datasetName = sample.dataset;

              const datasetBadgeColor =
                datasetName === "RAVDESS"
                  ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/30"
                  : datasetName === "TESS"
                    ? "bg-purple-500/20 text-purple-300 border-purple-500/30"
                    : "bg-amber-500/20 text-amber-300 border-amber-500/30";

              return (
                <div
                  key={sample.id}
                  className={`p-4 rounded-xl border transition-all ${
                    meta.borderColor
                  } bg-slate-900/60 hover:bg-slate-900 flex flex-col justify-between gap-3`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">{meta.emoji}</span>
                        <Badge
                          variant="outline"
                          className={`text-[10px] ${datasetBadgeColor}`}
                        >
                          {datasetName}
                        </Badge>
                      </div>
                      <Badge className={meta.badgeClass}>{meta.label}</Badge>
                    </div>

                    <h4 className="text-sm font-semibold text-white tracking-tight">
                      {sample.title}
                    </h4>
                    <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                      {sample.description}
                    </p>
                  </div>

                  <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between">
                    <div className="space-y-0.5">
                      <span className="text-[11px] text-slate-400 font-mono block">
                        {sample.speaker}
                      </span>
                      <span className="text-[10px] text-slate-500 font-mono flex items-center gap-1">
                        <Globe className="w-3 h-3 text-slate-500" />
                        {sample.language} • {sample.duration_seconds}s
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <Button
                        onClick={() => playSample(sample.id)}
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-slate-300 hover:text-white hover:bg-slate-800"
                        title="Listen to utterance"
                      >
                        {isCurrentPlaying ? (
                          <Pause className="w-4 h-4 text-indigo-400" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                      </Button>

                      <Button
                        onClick={() => handleTestSample(sample)}
                        disabled={isAnalyzing}
                        size="sm"
                        className="h-8 px-2.5 text-xs bg-indigo-600/80 hover:bg-indigo-600 text-white flex items-center gap-1"
                        title="Run model inference"
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        Test
                      </Button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
