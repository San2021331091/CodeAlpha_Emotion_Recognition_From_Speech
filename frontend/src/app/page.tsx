"use client";
import React, { useState, useEffect, useRef } from "react";
import { Navbar } from "@/components/Navbar";
import { AudioRecorder } from "@/components/AudioRecorder";
import { AudioUploader } from "@/components/AudioUploader";
import { BenchmarkSamples } from "@/components/BenchmarkSamples";
import { EmotionResultCard } from "@/components/EmotionResultCard";
import { AcousticFeatureViewer } from "@/components/AcousticFeatureViewer";
import { ModelPerformanceLab } from "@/components/ModelPerformanceLab";
import { EmotionHistoryManager } from "@/components/EmotionHistoryManager";
import { Sparkles, ArrowRight} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { toWav } from "@/lib/audio";
import { DEFAULT_SPEECH_MODEL } from "@/lib/constants";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

const Home:React.FC = ():React.JSX.Element=> {
  const [activeTab, setActiveTab] = useState("studio");
  const [selectedModel, setSelectedModel] = useState(DEFAULT_SPEECH_MODEL);
  const [prediction, setPrediction] = useState<any>(null);
  const [benchmarkExpectedEmotion, setBenchmarkExpectedEmotion] = useState<string | null>(null);
  const [lastUploadedFilename, setLastUploadedFilename] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [hasSaved, setHasSaved] = useState<boolean>(false);
  const [historyCount, setHistoryCount] = useState<number>(0);
  const analysisGeneration = useRef<number>(0);
  const analysisAbort = useRef<AbortController | null>(null);
  
  const handleModelChange = (model: string) => {
    analysisGeneration.current += 1;
    analysisAbort.current?.abort();
    setSelectedModel(model);
    setPrediction(null);
    setAnalysisError(null);
    setBenchmarkExpectedEmotion(null);
    setHasSaved(false);
    setIsAnalyzing(false);
  };

  useEffect(() => {
    fetchHistoryCount();
    return () => { analysisAbort.current?.abort(); analysisGeneration.current += 1; };
  }, []);

  const fetchHistoryCount = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/emotion-history`);
      if (res.ok) {
        const data = await res.json();
        setHistoryCount(data.length);
      }
    } catch (err) {
      console.warn("Could not reach gateway for history count");
    }
  };

  const handleAnalyzeAudio = async (
    audioBlob: Blob,
    filename: string,
    modelName: string,
    expectedEmotion?: string,
  ) => {
    const generation = ++analysisGeneration.current;
    analysisAbort.current?.abort();
    const controller = new AbortController();
    analysisAbort.current = controller;
    try {
      setIsAnalyzing(true);
      setAnalysisError(null);
      setPrediction(null);
      setBenchmarkExpectedEmotion(expectedEmotion ?? null);
      setHasSaved(false);
      setLastUploadedFilename(filename);

      const formData = new FormData();
      const wav = await toWav(audioBlob);
      if (generation !== analysisGeneration.current) return;
      const requestId = crypto.randomUUID();
      const digest = await crypto.subtle.digest("SHA-256", await wav.arrayBuffer());
      const fingerprint = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
      if (generation !== analysisGeneration.current) return;
      formData.append("file", wav, filename.replace(/\.[^.]+$/, "") + ".wav");
      formData.append("requestId", requestId);
      if (modelName) {
        formData.append("modelName", modelName);
      }

      const res = await fetch(`${API_BASE_URL}/speech-emotion/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
        cache: "no-store",
      });

      if (!res.ok) {
        const errText = await res.text();
        let message = errText || "Analysis failed";
        try { const error = JSON.parse(errText); message = error.message || error.detail || message; } catch {}
        throw new Error(message);
      }

      const data = await res.json();
      if (generation !== analysisGeneration.current) return;
      if (data.request_id !== requestId || data.input_sha256 !== fingerprint) {
        throw new Error("The result does not match the submitted audio. Please try analyzing it again.");
      }
      setPrediction(data);
    } catch (err: any) {
      if (generation !== analysisGeneration.current || controller.signal.aborted) return;
      setAnalysisError(err instanceof Error ? err.message : "Audio analysis failed. Please try again.");
    } finally {
      if (generation === analysisGeneration.current) setIsAnalyzing(false);
    }
  };

  const handleSaveToHistory = async (notes?: string) => {
    if (!prediction) return;
    try {
      setIsSaving(true);
      const payload = {
        title: lastUploadedFilename || `Utterance ${Date.now()}`,
        predictedEmotion: prediction.top_emotion,
        confidence: prediction.confidence,
        modelUsed: prediction.model_used,
        ...(benchmarkExpectedEmotion ? { verifiedEmotion: benchmarkExpectedEmotion } : {}),
        notes: notes || "",
        acousticSummary: prediction.acoustic_summary,
        valenceArousal: prediction.valence_arousal,
      };

      const res = await fetch(`${API_BASE_URL}/emotion-history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Failed to save session record");
      if (res.ok) {
        setHasSaved(true);
        setHistoryCount((prev) => prev + 1);
      }
    } catch (err) {
      alert("Failed to save session record.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#090e1a]">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        historyCount={historyCount}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {analysisError && <div role="alert" className="mb-6 rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-rose-300">{analysisError}</div>}
        {/* Studio View */}
        {activeTab === "studio" && (
          <div className="space-y-8">
            {/* Hero Banner */}
            <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-gradient-to-r from-indigo-950/40 via-purple-950/20 to-slate-950 p-6 sm:p-8">
              <div className="max-w-3xl space-y-3">
                <div className="flex items-center gap-2">
                  <Badge className="bg-indigo-500/20 text-indigo-300 border-indigo-500/30">
                    Acoustic Intelligence Engine
                  </Badge>
                </div>

                <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white">
                  Real-time Speech Emotion Recognition
                </h1>

                <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
                  Extracts 231 acoustic dimensions including 40 Mel-Frequency Cepstral Coefficients (MFCCs),
                  chroma pitch class profiles, spectral contrast, and fundamental frequencies ($F_0$),
                  classified through deep CRNN, 1D-CNN, and Bi-LSTM neural architectures.
                </p>

                <div className="flex flex-wrap items-center gap-3 pt-2">
                  <Button
                    onClick={() => setActiveTab("benchmark")}
                    variant="outline"
                    size="sm"
                    className="border-slate-700 text-slate-200 hover:bg-slate-800 gap-1.5 text-xs"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                    Try Benchmark Audio Clips
                  </Button>
                  <Button
                    onClick={() => setActiveTab("models")}
                    variant="ghost"
                    size="sm"
                    className="text-indigo-400 hover:text-indigo-300 gap-1 text-xs"
                  >
                    View Model Lab <ArrowRight className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Input Studio (Recorder & Uploader) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AudioRecorder
                onAnalyzeAudio={handleAnalyzeAudio}
                isAnalyzing={isAnalyzing}
                selectedModel={selectedModel}
                setSelectedModel={handleModelChange}
              />

              <AudioUploader
                onAnalyzeAudio={handleAnalyzeAudio}
                isAnalyzing={isAnalyzing}
                selectedModel={selectedModel}
                setSelectedModel={handleModelChange}
              />
            </div>

          </div>
        )}

        {/* Benchmark Samples View */}
        {activeTab === "benchmark" && (
          <BenchmarkSamples
            onAnalyzeAudio={handleAnalyzeAudio}
            isAnalyzing={isAnalyzing}
            selectedModel={selectedModel}
            apiBaseUrl={API_BASE_URL}
          />
        )}

        {/* Results stay in the current audio workspace. */}
        {(activeTab === "studio" || activeTab === "benchmark") && prediction && (
          <div className="space-y-8 animate-in fade-in duration-500">
            <p className="text-sm text-slate-400">
              Analysis results: {lastUploadedFilename}
            </p>
            <EmotionResultCard
              prediction={prediction}
              expectedEmotion={benchmarkExpectedEmotion}
              onSaveToHistory={handleSaveToHistory}
              isSaving={isSaving}
              hasSaved={hasSaved}
            />
            <AcousticFeatureViewer summary={prediction.acoustic_summary} />
          </div>
        )}

        {/* Model Lab View */}
        {activeTab === "models" && (
          <ModelPerformanceLab apiBaseUrl={API_BASE_URL} />
        )}

        {/* Session Logs View */}
        {activeTab === "history" && (
          <EmotionHistoryManager
            apiBaseUrl={API_BASE_URL}
            onRefreshCount={fetchHistoryCount}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/60 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-400">CodeAlpha Internship</span>
            <span>•</span>
            <span>Task 2: Emotion Recognition from Speech</span>
          </div>

          <div className="flex items-center gap-4">
            <span>Next.js 15 (Frontend)</span>
            <span>•</span>
            <span>NestJS (Gateway)</span>
            <span>•</span>
            <span>FastAPI (ML Engine)</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Home;