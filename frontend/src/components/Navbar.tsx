"use client";
import React from "react";
import { Mic, BarChart3, History, Sparkles } from "lucide-react";

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  historyCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, historyCount }):React.JSX.Element => {
  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-slate-950/80 border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Mic className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                VocalSense AI
              </span>             
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Speech Emotion Recognition
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 sm:gap-2">
          <button
            onClick={() => setActiveTab("studio")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              activeTab === "studio"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <Mic className="w-4 h-4" />
            <span className="hidden md:inline">Voice Studio</span>
          </button>

          <button
            onClick={() => setActiveTab("benchmark")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              activeTab === "benchmark"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <Sparkles className="w-4 h-4" />
            <span className="hidden md:inline">Benchmark Samples</span>
          </button>

          <button
            onClick={() => setActiveTab("models")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              activeTab === "models"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span className="hidden md:inline">Model Lab</span>
          </button>

          <button
            onClick={() => setActiveTab("history")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              activeTab === "history"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <History className="w-4 h-4" />
            <span className="hidden md:inline">Session Logs</span>
            {historyCount > 0 && (
              <span className="px-1.5 py-0.2 rounded-full text-xs bg-slate-800 text-slate-300 font-mono">
                {historyCount}
              </span>
            )}
          </button>
        </nav>
      </div>
    </header>
  );
};
