"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Play, Pause, RotateCcw, Sparkles, Loader2, Volume2, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { AVAILABLE_MODELS } from "@/lib/constants";
import { toWav } from "@/lib/audio";

enum RecordingPhase{
  Idle = "idle",
  Starting = "starting",
  Recording = "recording",
  Finalizing = "finalizing"
}
interface AudioRecorderProps {
  onAnalyzeAudio: (audioBlob: Blob, filename: string, modelName: string) => Promise<void>;
  isAnalyzing: boolean;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  onAnalyzeAudio,
  isAnalyzing,
  selectedModel,
  setSelectedModel,
}): React.JSX.Element => {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [recordDuration, setRecordDuration] = useState<number>(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingPhaseRef = useRef<RecordingPhase>(RecordingPhase.Idle);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const mountedRef = useRef<boolean>(true);
  const [isStarting, setIsStarting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  useEffect(() => { return () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }; }, [audioUrl]);

  // Clean up
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
      streamRef.current?.getTracks().forEach(track => track.stop());
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (audioContextRef.current?.state !== "closed") void audioContextRef.current?.close();
    };
  }, []);

  const startRecording = async () => {
    if (recordingPhaseRef.current !== RecordingPhase.Idle) return;
    recordingPhaseRef.current = RecordingPhase.Starting;
    setIsStarting(true);
    setError(null);
    try {
      audioPlayerRef.current?.pause();
      setIsPlaying(false);
      setRecordedBlob(null);
      setAudioUrl(null);
      setRecordDuration(0);
      // Each recorder owns its chunks; an old onstop callback must never read
      // chunks belonging to a later recording.
      const chunks: Blob[] = [];

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          autoGainControl: true,
          noiseSuppression: true,
          echoCancellation: true,
        },
      });
      if (!mountedRef.current) { stream.getTracks().forEach(track => track.stop()); recordingPhaseRef.current = RecordingPhase.Idle; return; }
      streamRef.current = stream;
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        if (audioContext.state !== "closed") void audioContext.close();
        recordingPhaseRef.current = RecordingPhase.Idle;
        if (!mountedRef.current) return;
        setIsFinalizing(false);
        setIsRecording(false);
        if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        const audioBlob = new Blob(chunks, { type: mediaRecorder.mimeType || chunks[0]?.type });
        if (!audioBlob.size) { setError("No audio was captured. Check the microphone and record again."); return; }
        setRecordedBlob(audioBlob);
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
      };

      mediaRecorder.start(100);
      recordingPhaseRef.current = RecordingPhase.Recording;
      setIsRecording(true);

      // Start duration counter
      const startTimestamp = Date.now();
      timerIntervalRef.current = setInterval(() => {
        setRecordDuration(Math.floor((Date.now() - startTimestamp) / 1000));
      }, 200);

      // Start canvas visualization
      drawWaveform();
    } catch (err: any) {
      recordingPhaseRef.current = RecordingPhase.Idle;
      streamRef.current?.getTracks().forEach(track => track.stop());
      if (audioContextRef.current?.state !== "closed") void audioContextRef.current?.close();
      setError("Unable to record audio. Check microphone permissions and browser support.");
    } finally {
      if (mountedRef.current) setIsStarting(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      recordingPhaseRef.current = RecordingPhase.Finalizing;
      setIsFinalizing(true);
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    }
  };

  const drawWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const render = () => {
      animationFrameRef.current = requestAnimationFrame(render);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = "rgba(15, 23, 42, 0.4)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#818cf8";
      ctx.beginPath();

      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };

    render();
  };

  const togglePlayback = async () => {
    if (!audioPlayerRef.current || !audioUrl) return;
    if (isPlaying) {
      audioPlayerRef.current.pause();
      setIsPlaying(false);
    } else {
      try {
        await audioPlayerRef.current.play();
        setIsPlaying(true);
      } catch { setIsPlaying(false); setError("Audio playback failed. Please record or select audio again."); }
    }
  };

  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const handleAnalyze = () => {
    if (!recordedBlob) return;
    const filename = `mic_recording_${Date.now()}.${recordedBlob.type.includes("mp4") ? "m4a" : recordedBlob.type.includes("ogg") ? "ogg" : "webm"}`;
    onAnalyzeAudio(recordedBlob, filename, selectedModel);
  };

  const downloadRecording = async () => {
    if (!recordedBlob) return;
    setIsExporting(true);
    try {
      const wav = await toWav(recordedBlob);
      const url = URL.createObjectURL(wav);
      const link = document.createElement("a");
      link.href = url;
      link.download = `mic_recording_${Date.now()}.wav`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not export the recording.");
    } finally {
      if (mountedRef.current) setIsExporting(false);
    }
  };

  return (
    <Card className="glass-panel border-slate-800">
      <CardHeader>
        <div className="flex flex-col gap-3">
          <div>
            <CardTitle className="text-xl flex items-center gap-2 text-white">
              <Mic className="w-5 h-5 text-indigo-400" />
              Live Microphone Recorder
            </CardTitle>
            <CardDescription className="text-slate-400">
              Record a clear sentence in a quiet room. Speak for 3–10 seconds with your natural tone.
            </CardDescription>
          </div>

          {/* Model Selector */}
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

      <CardContent className="space-y-6">
        {error && <p role="alert" className="text-sm text-rose-400">{error}</p>}
        {/* Real-time Oscilloscope Canvas */}
        <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-slate-950 h-32 flex items-center justify-center">
          <canvas ref={canvasRef} width={640} height={128} className="w-full h-full" />

          {!isRecording && !audioUrl && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/70 text-slate-400 gap-2">
              <Volume2 className="w-8 h-8 text-slate-600 animate-pulse" />
              <span className="text-xs">Click Start Recording to initialize audio visualizer</span>
            </div>
          )}

          {isRecording && (
            <div className="absolute top-3 right-3 flex items-center gap-2 px-2.5 py-1 rounded-full bg-rose-500/20 border border-rose-500/40 text-rose-400 text-xs font-mono">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
              RECORDING {formatTimer(recordDuration)}
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            {!isRecording ? (
              <Button
                onClick={startRecording}
                disabled={isAnalyzing || isStarting || isFinalizing}
                className="bg-rose-600 hover:bg-rose-700 text-white flex items-center gap-2 shadow-lg shadow-rose-600/20"
              >
                <Mic className="w-4 h-4" />
                Start Recording
              </Button>
            ) : (
              <Button
                onClick={stopRecording}
                className="bg-slate-800 hover:bg-slate-700 text-rose-400 border border-rose-500/40 flex items-center gap-2 pulse-recording"
              >
                <Square className="w-4 h-4 fill-current" />
                Stop Recording
              </Button>
            )}

            {audioUrl && (
              <>
                <Button
                  onClick={downloadRecording}
                  disabled={isExporting || isAnalyzing}
                  variant="outline"
                  className="border-slate-700 text-slate-300 hover:bg-slate-800 gap-2"
                >
                  <Download className="w-4 h-4" />
                  {isExporting ? "Exporting..." : "Download WAV"}
                </Button>
                <Button
                  onClick={togglePlayback}
                  variant="outline"
                  size="icon"
                  className="border-slate-700 text-slate-300 hover:bg-slate-800"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </Button>
                <audio
                  ref={audioPlayerRef}
                  src={audioUrl}
                  onEnded={() => setIsPlaying(false)}
                  className="hidden"
                />

                <Button
                  onClick={() => {
                    audioPlayerRef.current?.pause();
                    setIsPlaying(false);
                    setRecordedBlob(null);
                    setAudioUrl(null);
                    setRecordDuration(0);
                  }}
                  variant="ghost"
                  size="icon"
                  className="text-slate-400 hover:text-slate-200"
                >
                  <RotateCcw className="w-4 h-4" />
                </Button>
                <span className="text-xs text-slate-400 font-mono">
                  {formatTimer(recordDuration)} captured
                </span>
              </>
            )}
          </div>

          {recordedBlob && (
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing || isStarting}
              className="bg-indigo-600 hover:bg-indigo-700 text-white flex items-center gap-2 shadow-lg shadow-indigo-600/30"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing Acoustic Waves...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Analyze with Deep Learning
                </>
              )}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
