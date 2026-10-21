"use client";

import React, { useState, useRef, useEffect } from "react";
import { UploadCloud, FileAudio, Play, Pause, X, Sparkles, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { AVAILABLE_MODELS } from "@/lib/constants";

interface AudioUploaderProps {
  onAnalyzeAudio: (audioBlob: Blob, filename: string, modelName: string) => Promise<void>;
  isAnalyzing: boolean;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
}

export const AudioUploader: React.FC<AudioUploaderProps> = ({
  onAnalyzeAudio,
  isAnalyzing,
  selectedModel,
  setSelectedModel,
}):React.JSX.Element => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => { return () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }; }, [audioUrl]);

  const handleFileChange = (file: File) => {
    setError(null);
    if (!file || !file.size || file.size > 25 * 1024 * 1024) { setError("Choose a non-empty audio file up to 25 MB."); return; }
    if (!file.type.includes("audio") && !file.name.match(/\.(wav|mp3|ogg|webm|m4a|flac)$/i)) {
      setError("Please upload a valid audio file (WAV, MP3, OGG, WebM, M4A, FLAC).");
      return;
    }
    audioRef.current?.pause();
    setIsPlaying(false);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const togglePlayback = async () => {
    if (!audioRef.current || !audioUrl) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      try {
        await audioRef.current.play();
        setIsPlaying(true);
      } catch { setIsPlaying(false); setError("Audio playback failed. Please record or select audio again."); }
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setAudioUrl(null);
    setIsPlaying(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;
    onAnalyzeAudio(selectedFile, selectedFile.name, selectedModel);
  };

  return (
    <Card className="glass-panel border-slate-800">
      <CardHeader>
        <div className="flex flex-col gap-3">
          <div>
            <CardTitle className="text-xl flex items-center gap-2 text-white">
              <UploadCloud className="w-5 h-5 text-indigo-400" />
              Upload Audio File
            </CardTitle>
            <CardDescription className="text-slate-400">
              Upload spoken audio from RAVDESS, TESS, EMO-DB, or your own voice recording.
            </CardDescription>
          </div>

          <div className="flex min-w-0 items-center gap-2">
            <span className="text-xs text-slate-400">Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="min-w-0 w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              {AVAILABLE_MODELS.map((m) => (
                <option key={m.id} value={m.name}>
                  {m.name} ({m.tag})
                </option>
              ))}
            </select>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <p className="text-xs text-slate-400">
          This model estimates emotion in a speaking voice. Music mood and singing are
          outside its training scope; even a high score on those files can be misleading.
        </p>
        {error && <p role="alert" className="text-sm text-rose-400">{error}</p>}
        {!selectedFile ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all ${
              isDragOver
                ? "border-indigo-500 bg-indigo-500/10"
                : "border-slate-800 hover:border-slate-700 hover:bg-slate-900/50"
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
              accept="audio/*,.wav,.mp3,.ogg,.webm,.m4a,.flac"
              className="hidden"
            />
            <div className="w-12 h-12 rounded-full bg-indigo-500/10 flex items-center justify-center mb-3">
              <UploadCloud className="w-6 h-6 text-indigo-400" />
            </div>
            <p className="text-sm font-medium text-slate-200">
              Click to upload or drag & drop audio
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports WAV, MP3, OGG, WebM, M4A, FLAC (Up to 25MB)
            </p>
          </div>
        ) : (
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-950 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-indigo-500/20 flex items-center justify-center text-indigo-400">
                <FileAudio className="w-5 h-5" />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-200">{selectedFile.name}</p>
                <p className="text-xs text-slate-500">
                  {(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type || "audio/wav"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {audioUrl && (
                <>
                  <Button
                    onClick={togglePlayback}
                    variant="outline"
                    size="icon"
                    className="border-slate-700 text-slate-300 hover:bg-slate-800"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </Button>
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    onEnded={() => setIsPlaying(false)}
                    className="hidden"
                  />
                </>
              )}

              <Button
                onClick={clearFile}
                variant="ghost"
                size="icon"
                className="text-slate-400 hover:text-rose-400"
              >
                <X className="w-4 h-4" />
              </Button>

              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="bg-indigo-600 hover:bg-indigo-700 text-white flex items-center gap-2 shadow-lg shadow-indigo-600/30"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Extracting MFCCs...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Analyze Emotion
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
