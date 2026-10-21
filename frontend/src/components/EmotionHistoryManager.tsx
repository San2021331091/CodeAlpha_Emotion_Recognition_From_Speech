"use client";
import React, { useState, useEffect } from "react";
import { History, Download, Trash2, Edit3, Check, Search, Filter } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EMOTIONS_CONFIG } from "@/lib/constants";

interface HistoryRecord {
  id: string;
  title: string;
  predictedEmotion: string;
  confidence: number;
  modelUsed: string;
  verifiedEmotion?: string;
  notes?: string;
  acousticSummary?: any;
  valenceArousal?: any;
  createdAt: string;
}

interface EmotionHistoryManagerProps {
  apiBaseUrl: string;
  onRefreshCount?: () => void;
}

export const EmotionHistoryManager: React.FC<EmotionHistoryManagerProps> = ({
  apiBaseUrl,
  onRefreshCount,
}):React.JSX.Element => {
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [analytics, setAnalytics] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedEmotion, setSelectedEmotion] = useState<string>("all");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editNotes, setEditNotes] = useState<string>("");
  const [editVerifiedEmotion, setEditVerifiedEmotion] = useState<string>("");

  useEffect(() => {
    fetchHistoryAndAnalytics();
  }, [selectedEmotion]);

  const fetchHistoryAndAnalytics = async (): Promise<void> => {
    try {
      setLoading(true);
      const url = `${apiBaseUrl}/emotion-history?emotion=${selectedEmotion}&search=${encodeURIComponent(searchQuery)}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }

      const analyticsRes = await fetch(`${apiBaseUrl}/analytics/summary`);
      if (analyticsRes.ok) {
        const aData = await analyticsRes.json();
        setAnalytics(aData);
      }
    } catch (err) {
      console.error("Failed to load history", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchHistoryAndAnalytics();
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this session record?")) return;
    try {
      const res = await fetch(`${apiBaseUrl}/emotion-history/${id}`, { method: "DELETE" });
      if (res.ok) {
        setHistory((prev) => prev.filter((item) => item.id !== id));
        onRefreshCount?.();
      }
    } catch (err) {
      alert("Failed to delete record.");
    }
  };

  const startEdit = (record: HistoryRecord) => {
    setEditingId(record.id);
    setEditNotes(record.notes || "");
    setEditVerifiedEmotion(record.verifiedEmotion || record.predictedEmotion);
  };

  const saveEdit = async (id: string) => {
    try {
      const res = await fetch(`${apiBaseUrl}/emotion-history/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          notes: editNotes,
          verifiedEmotion: editVerifiedEmotion,
        }),
      });
      if (res.ok) {
        const updated = await res.json();
        setHistory((prev) => prev.map((item) => (item.id === id ? updated : item)));
        setEditingId(null);
      }
    } catch (err) {
      alert("Failed to save edit.");
    }
  };

  const exportCsv = () => {
    window.open(`${apiBaseUrl}/emotion-history/export/csv`, "_blank");
  };

  return (
    <div className="space-y-6">
      {/* Top Metrics Cards */}
      {analytics && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Card className="glass-panel border-slate-800 p-4">
            <span className="text-xs text-slate-400 block">Total Analyzed Utterances</span>
            <span className="text-2xl font-bold font-mono text-white mt-1 block">
              {analytics.totalUtterances}
            </span>
            <span className="text-[11px] text-slate-500 mt-0.5 block">Session log records</span>
          </Card>

          <Card className="glass-panel border-slate-800 p-4">
            <span className="text-xs text-slate-400 block">Mean Model Confidence</span>
            <span className="text-2xl font-bold font-mono text-emerald-400 mt-1 block">
              {(analytics.averageConfidence * 100).toFixed(1)}%
            </span>
            <span className="text-[11px] text-slate-500 mt-0.5 block">Across all classified inputs</span>
          </Card>

          <Card className="glass-panel border-slate-800 p-4">
            <span className="text-xs text-slate-400 block">User Ground-Truth Agreement</span>
            <span className="text-2xl font-bold font-mono text-indigo-400 mt-1 block">
              {analytics.userAgreementRate}%
            </span>
            <span className="text-[11px] text-slate-500 mt-0.5 block">Verified classification accuracy</span>
          </Card>
        </div>
      )}

      {/* Main Table Card */}
      <Card className="glass-panel border-slate-800">
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <CardTitle className="text-xl flex items-center gap-2 text-white">
                <History className="w-5 h-5 text-indigo-400" />
                Session Voice Analysis Workspace
              </CardTitle>
              <CardDescription className="text-slate-400">
                Audited speech emotion recordings, acoustic diagnostics, and human feedback
              </CardDescription>
            </div>

            <Button
              onClick={exportCsv}
              variant="outline"
              size="sm"
              className="border-slate-700 text-slate-300 hover:bg-slate-800 flex items-center gap-2 text-xs"
            >
              <Download className="w-3.5 h-3.5 text-indigo-400" />
              Export to CSV
            </Button>
          </div>

          {/* Search & Filter Toolbar */}
          <div className="flex flex-col sm:flex-row items-center gap-3 pt-3">
            <form onSubmit={handleSearch} className="flex-1 w-full flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search by utterance title or notes..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              </div>
              <Button type="submit" size="sm" variant="secondary" className="text-xs">
                Search
              </Button>
            </form>

            <div className="flex items-center gap-2 w-full sm:w-auto">
              <Filter className="w-4 h-4 text-slate-500 shrink-0" />
              <select
                value={selectedEmotion}
                onChange={(e) => setSelectedEmotion(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500 w-full sm:w-auto"
              >
                <option value="all">All Emotions</option>
                {Object.entries(EMOTIONS_CONFIG).map(([key, val]) => (
                  <option key={key} value={key}>
                    {val.emoji} {val.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {loading ? (
            <div className="text-center p-8 text-slate-400 text-xs">Loading records...</div>
          ) : history.length === 0 ? (
            <div className="text-center p-12 text-slate-500 space-y-2">
              <History className="w-8 h-8 mx-auto text-slate-600" />
              <p className="text-sm font-medium">No speech analysis records found</p>
              <p className="text-xs">
                Record an audio sample or test a benchmark utterance in the Voice Studio to begin
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-slate-400 border-b border-slate-800 bg-slate-900/40">
                  <tr>
                    <th className="px-4 py-3">Utterance</th>
                    <th className="px-4 py-3">Predicted Emotion</th>
                    <th className="px-4 py-3">Confidence</th>
                    <th className="px-4 py-3">Verified Label</th>
                    <th className="px-4 py-3">Acoustic Specs</th>
                    <th className="px-4 py-3">Notes</th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                  {history.map((record) => {
                    const meta = EMOTIONS_CONFIG[record.predictedEmotion] || EMOTIONS_CONFIG.neutral;
                    const isEditing = editingId === record.id;

                    return (
                      <tr key={record.id} className="hover:bg-slate-900/40 transition-colors">
                        <td className="px-4 py-3 font-sans">
                          <p className="text-slate-200 font-medium truncate max-w-[180px]" title={record.title}>
                            {record.title}
                          </p>
                          <span className="text-[10px] text-slate-500 font-mono">
                            {new Date(record.createdAt).toLocaleTimeString()} • {record.modelUsed}
                          </span>
                        </td>

                        <td className="px-4 py-3 font-sans">
                          <Badge className={meta.badgeClass}>
                            {meta.emoji} {meta.label}
                          </Badge>
                        </td>

                        <td className="px-4 py-3 text-slate-200 font-bold">
                          {(record.confidence * 100).toFixed(1)}%
                        </td>

                        <td className="px-4 py-3 font-sans">
                          {isEditing ? (
                            <select
                              value={editVerifiedEmotion}
                              onChange={(e) => setEditVerifiedEmotion(e.target.value)}
                              className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white"
                            >
                              {Object.entries(EMOTIONS_CONFIG).map(([key, val]) => (
                                <option key={key} value={key}>
                                  {val.emoji} {val.label}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <span className="text-slate-300 font-medium">
                              {EMOTIONS_CONFIG[record.verifiedEmotion || record.predictedEmotion]?.emoji}{" "}
                              {EMOTIONS_CONFIG[record.verifiedEmotion || record.predictedEmotion]?.label}
                            </span>
                          )}
                        </td>

                        <td className="px-4 py-3 text-[11px] text-slate-400">
                          {record.acousticSummary ? (
                            <span>
                              {record.acousticSummary.pitch_f0_mean}Hz • {record.acousticSummary.duration_seconds}s
                            </span>
                          ) : (
                            "N/A"
                          )}
                        </td>

                        <td className="px-4 py-3 font-sans text-xs text-slate-300">
                          {isEditing ? (
                            <input
                              type="text"
                              value={editNotes}
                              onChange={(e) => setEditNotes(e.target.value)}
                              className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white w-full"
                            />
                          ) : (
                            <span className="truncate max-w-[180px] block" title={record.notes}>
                              {record.notes || <span className="text-slate-600 italic">None</span>}
                            </span>
                          )}
                        </td>

                        <td className="px-4 py-3 text-right">
                          <div className="flex items-center justify-end gap-1.5">
                            {isEditing ? (
                              <Button
                                onClick={() => saveEdit(record.id)}
                                size="icon"
                                variant="ghost"
                                className="h-7 w-7 text-emerald-400 hover:bg-emerald-950/30"
                              >
                                <Check className="w-3.5 h-3.5" />
                              </Button>
                            ) : (
                              <Button
                                onClick={() => startEdit(record)}
                                size="icon"
                                variant="ghost"
                                className="h-7 w-7 text-slate-400 hover:text-white hover:bg-slate-800"
                              >
                                <Edit3 className="w-3.5 h-3.5" />
                              </Button>
                            )}

                            <Button
                              onClick={() => handleDelete(record.id)}
                              size="icon"
                              variant="ghost"
                              className="h-7 w-7 text-slate-500 hover:text-rose-400 hover:bg-rose-950/30"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
