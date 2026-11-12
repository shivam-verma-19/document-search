"use client";

import { useEffect, useState } from "react";
import api from "../apiClient";
import { useApi } from "../../hooks/useAPI";

interface MetricItem {
  query?: string;
  latency?: number;
  source?: string;
  timestamp?: string | number;
}

interface RagasScores {
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
}

export default function MetricsPage() {
  const [activeTab, setActiveTab] = useState<"performance" | "ragas">("performance");
  const metricsApi = useApi();

  // RAGAS Tester state
  const [evalQuery, setEvalQuery] = useState(
    "How does AWS Lambda scale and how is it priced?"
  );
  const [evalAnswer, setEvalAnswer] = useState(
    "AWS Lambda automatically scales serverless compute up to account limits. It is priced per millisecond of compute duration and total request volume."
  );
  const [evalContext, setEvalContext] = useState(
    "AWS Lambda automatically scales serverless compute up to account concurrency limits. AWS Lambda pricing is charged per millisecond of compute duration."
  );
  const [evalGroundTruth, setEvalGroundTruth] = useState(
    "AWS Lambda scales automatically and is billed per millisecond of execution and request count."
  );
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalScores, setEvalScores] = useState<RagasScores | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  useEffect(() => {
    metricsApi.get("/metrics");
  }, []);

  const runRagasEvaluation = async () => {
    setEvalRunning(true);
    setEvalError(null);
    try {
      const res = await api.post("/eval", {
        query: evalQuery,
        answer: evalAnswer,
        contexts: [evalContext],
        ground_truth: evalGroundTruth,
      });
      setEvalScores(res.data);
    } catch (err: any) {
      setEvalError(err.response?.data?.error?.message || err.message || "Evaluation failed.");
    } finally {
      setEvalRunning(false);
    }
  };

  const getScoreBadgeColor = (score?: number) => {
    if (score === undefined || score === null) return "#94a3b8";
    if (score >= 0.8) return "#10b981"; // Green
    if (score >= 0.6) return "#f59e0b"; // Yellow
    return "#ef4444"; // Red
  };

  const rawMetrics: MetricItem[] = Array.isArray(metricsApi.data) ? metricsApi.data : [];

  // Compute aggregate stats
  const totalQueries = rawMetrics.length;
  const avgLatency =
    totalQueries > 0
      ? (rawMetrics.reduce((sum, item) => sum + (item.latency || 0), 0) / totalQueries).toFixed(3)
      : "0.000";

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 24,
          paddingBottom: 16,
          borderBottom: "1px solid var(--border-color)",
        }}
      >
        <div>
          <h1 style={{ fontSize: "1.75rem", fontWeight: 700 }}>Metrics & Quality Evaluation</h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem" }}>
            Monitor system retrieval latencies and benchmark answer quality using the RAGAS framework.
          </p>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <button
            onClick={() => setActiveTab("performance")}
            style={{
              padding: "8px 16px",
              borderRadius: 6,
              border: "1px solid var(--border-color)",
              backgroundColor: activeTab === "performance" ? "var(--accent-color)" : "var(--bg-secondary)",
              color: "#fff",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Latency & Retrieval
          </button>
          <button
            onClick={() => setActiveTab("ragas")}
            style={{
              padding: "8px 16px",
              borderRadius: 6,
              border: "1px solid var(--border-color)",
              backgroundColor: activeTab === "ragas" ? "var(--accent-color)" : "var(--bg-secondary)",
              color: "#fff",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            RAGAS Evaluation Hub
          </button>
        </div>
      </div>

      {activeTab === "performance" ? (
        <div>
          {/* Summary Stat Cards */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: 16,
              marginBottom: 24,
            }}
          >
            <div
              style={{
                padding: 16,
                borderRadius: 8,
                backgroundColor: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Total Queries</span>
              <p style={{ fontSize: "1.75rem", fontWeight: 700, marginTop: 4 }}>{totalQueries}</p>
            </div>
            <div
              style={{
                padding: 16,
                borderRadius: 8,
                backgroundColor: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Average Latency</span>
              <p style={{ fontSize: "1.75rem", fontWeight: 700, marginTop: 4, color: "var(--accent-success)" }}>
                {avgLatency}s
              </p>
            </div>
            <div
              style={{
                padding: 16,
                borderRadius: 8,
                backgroundColor: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Retrieval Engine</span>
              <p style={{ fontSize: "1.25rem", fontWeight: 700, marginTop: 8, color: "#60a5fa" }}>
                Hybrid (BM25 + Vectors)
              </p>
            </div>
          </div>

          <div
            style={{
              padding: 20,
              borderRadius: 8,
              backgroundColor: "var(--bg-secondary)",
              border: "1px solid var(--border-color)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h2 style={{ fontSize: "1.15rem", fontWeight: 600 }}>Query Latency Logs</h2>
              <button
                onClick={() => metricsApi.get("/metrics")}
                style={{
                  padding: "4px 12px",
                  borderRadius: 4,
                  backgroundColor: "#334155",
                  border: "none",
                  color: "#fff",
                  fontSize: "0.85rem",
                  cursor: "pointer",
                }}
              >
                Refresh
              </button>
            </div>

            {metricsApi.loading && <p style={{ color: "var(--text-secondary)" }}>Loading metrics...</p>}
            {rawMetrics.length === 0 && !metricsApi.loading && (
              <p style={{ color: "var(--text-secondary)" }}>No metrics recorded yet. Run a query in the Chat tab!</p>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {rawMetrics.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "10px 14px",
                    borderRadius: 6,
                    backgroundColor: "rgba(15, 23, 42, 0.5)",
                    border: "1px solid var(--border-color)",
                  }}
                >
                  <div style={{ flex: 1, paddingRight: 16 }}>
                    <p style={{ fontWeight: 500, fontSize: "0.95rem" }}>{item.query || "Query"}</p>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                      Source: {item.source || "Hybrid RAG"}
                    </span>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <span
                      style={{
                        padding: "4px 10px",
                        borderRadius: 4,
                        backgroundColor: "rgba(16, 185, 129, 0.15)",
                        color: "var(--accent-success)",
                        fontWeight: 600,
                        fontSize: "0.85rem",
                      }}
                    >
                      {item.latency ? item.latency.toFixed(3) : "0.000"}s
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* RAGAS Quality Evaluation Tester */
        <div>
          <div
            style={{
              padding: 24,
              borderRadius: 8,
              backgroundColor: "var(--bg-secondary)",
              border: "1px solid var(--border-color)",
              marginBottom: 24,
            }}
          >
            <h2 style={{ fontSize: "1.2rem", fontWeight: 600, marginBottom: 16 }}>
              Interactive RAGAS Metric Evaluator
            </h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>
                  Test Query
                </label>
                <input
                  type="text"
                  value={evalQuery}
                  onChange={(e) => setEvalQuery(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: 6,
                    backgroundColor: "var(--bg-primary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>
                  Generated Answer to Evaluate
                </label>
                <textarea
                  rows={3}
                  value={evalAnswer}
                  onChange={(e) => setEvalAnswer(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: 6,
                    backgroundColor: "var(--bg-primary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>
                  Retrieved Context Chunks
                </label>
                <textarea
                  rows={3}
                  value={evalContext}
                  onChange={(e) => setEvalContext(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: 6,
                    backgroundColor: "var(--bg-primary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>
                  Ground Truth (Expected Target Answer)
                </label>
                <input
                  type="text"
                  value={evalGroundTruth}
                  onChange={(e) => setEvalGroundTruth(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: 6,
                    backgroundColor: "var(--bg-primary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>

              <button
                onClick={runRagasEvaluation}
                disabled={evalRunning}
                style={{
                  padding: "12px",
                  borderRadius: 6,
                  backgroundColor: "var(--accent-color)",
                  color: "#fff",
                  fontWeight: 600,
                  border: "none",
                  cursor: evalRunning ? "not-allowed" : "pointer",
                  marginTop: 8,
                }}
              >
                {evalRunning ? "Running RAGAS Evaluation..." : "Calculate RAGAS Scores"}
              </button>
            </div>

            {evalError && (
              <p style={{ marginTop: 12, color: "var(--accent-danger)", fontSize: "0.9rem" }}>
                {evalError}
              </p>
            )}
          </div>

          {/* RAGAS Scores Cards */}
          {evalScores && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: 16,
              }}
            >
              {[
                { name: "Faithfulness", key: "faithfulness", desc: "Is the answer grounded in the retrieved context?" },
                { name: "Answer Relevancy", key: "answer_relevancy", desc: "Does the answer directly address the question?" },
                { name: "Context Precision", key: "context_precision", desc: "Are the most relevant chunks ranked at top?" },
                { name: "Context Recall", key: "context_recall", desc: "Did retrieval cover all necessary facts?" },
              ].map((metric) => {
                const score = (evalScores as any)[metric.key];
                const pct = score !== undefined && score !== null ? (score * 100).toFixed(1) : "N/A";
                const color = getScoreBadgeColor(score);
                return (
                  <div
                    key={metric.key}
                    style={{
                      padding: 20,
                      borderRadius: 8,
                      backgroundColor: "var(--bg-secondary)",
                      border: `1px solid ${color}`,
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: "0.9rem", fontWeight: 600 }}>{metric.name}</span>
                      <span style={{ fontSize: "1.25rem", fontWeight: 700, color }}>{pct}%</span>
                    </div>
                    <div
                      style={{
                        width: "100%",
                        height: 6,
                        backgroundColor: "#334155",
                        borderRadius: 3,
                        margin: "12px 0 8px",
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          width: `${score ? score * 100 : 0}%`,
                          height: "100%",
                          backgroundColor: color,
                          borderRadius: 3,
                        }}
                      />
                    </div>
                    <p style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>{metric.desc}</p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
