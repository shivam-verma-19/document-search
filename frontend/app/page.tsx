"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import api from "./apiClient";
import { useAuth } from "../components/AuthProvider";

export default function Home() {
  const { status } = useAuth();
  const [healthStatus, setHealthStatus] = useState<string>("checking...");
  const [healthOk, setHealthOk] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .get("/")
      .then((res) => {
        if (res.data?.status === "running") {
          setHealthStatus("Connected & Healthy");
          setHealthOk(true);
        } else {
          setHealthStatus("Online");
          setHealthOk(true);
        }
      })
      .catch(() => {
        setHealthStatus("Backend Offline or Unreachable");
        setHealthOk(false);
      });
  }, []);

  const features = [
    {
      title: "🔍 Hybrid BM25 + Vector Retrieval",
      description:
        "Dense embedding cosine similarity merged with sparse BM25 lexical ranking via Reciprocal Rank Fusion (RRF).",
      tag: "Dual Engine",
    },
    {
      title: "🧩 Adaptive & Semantic Chunking",
      description:
        "Extracts section headers (#, ##, Title Caps) and creates topic-coherent boundaries with dynamic length budgeting.",
      tag: "Section-Aware",
    },
    {
      title: "🎯 Cross-Encoder Reranking",
      description:
        "Passage reranking utilizing cross-encoder MiniLM with fast lexical fallback for sub-millisecond precision.",
      tag: "Reranker",
    },
    {
      title: "🧠 Multi-Step Query Decomposition",
      description:
        "Decomposes complex comparative questions into focused sub-queries for thorough multi-angle context retrieval.",
      tag: "Planner",
    },
    {
      title: "💬 DynamoDB Conversation Memory",
      description:
        "Stateful multi-turn conversation tracking with automated TTL expiration for contextual follow-up questions.",
      tag: "Memory",
    },
    {
      title: "📊 RAGAS Quality Evaluation",
      description:
        "Continuous evaluation of Faithfulness, Answer Relevancy, Context Precision, and Context Recall.",
      tag: "Quality",
    },
  ];

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 32,
          paddingBottom: 24,
          borderBottom: "1px solid var(--border-color)",
        }}
      >
        <div>
          <h1 style={{ fontSize: "2rem", fontWeight: 700, marginBottom: 8 }}>
            RAG Platform Dashboard
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "1rem" }}>
            Production-grade retrieval-augmented generation with stateful memory and hybrid ranking.
          </p>
        </div>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "8px 16px",
            borderRadius: 20,
            backgroundColor: healthOk ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
            border: `1px solid ${healthOk ? "var(--accent-success)" : "var(--accent-danger)"}`,
            fontSize: "0.9rem",
            color: healthOk ? "var(--accent-success)" : "var(--accent-danger)",
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: healthOk ? "var(--accent-success)" : "var(--accent-danger)",
            }}
          />
          {healthStatus}
        </div>
      </div>

      {status !== "authenticated" && (
        <section
          style={{
            padding: 24,
            marginBottom: 32,
            borderRadius: 12,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <h2 style={{ marginBottom: 8 }}>New here?</h2>
          <p style={{ color: "var(--text-secondary)", marginBottom: 16 }}>
            Sign in with your username, or create an account and verify your
            email to get started.
          </p>
          <Link
            href="/login"
            style={{
              display: "inline-block",
              padding: "10px 16px",
              borderRadius: 6,
              background: "var(--accent-color)",
              fontWeight: 600,
            }}
          >
            Sign in or create account
          </Link>
          {status === "error" && (
            <p role="alert" style={{ color: "var(--accent-warning)", marginTop: 12 }}>
              Your session could not be checked. Sign in again or retry later.
            </p>
          )}
        </section>
      )}

      {/* Quick Launch Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 20,
          marginBottom: 40,
        }}
      >
        <Link
          href="/chat"
          style={{
            padding: 24,
            borderRadius: 12,
            backgroundColor: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
            transition: "transform 0.15s ease",
          }}
        >
          <h2 style={{ fontSize: "1.25rem", marginBottom: 8 }}>💬 Interactive Chat</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", marginBottom: 16 }}>
            Ask questions with real-time SSE streaming, session memory, and metadata filtering.
          </p>
          <span style={{ color: "var(--accent-color)", fontWeight: 600 }}>Launch Chat &rarr;</span>
        </Link>

        <Link
          href="/upload"
          style={{
            padding: 24,
            borderRadius: 12,
            backgroundColor: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <h2 style={{ fontSize: "1.25rem", marginBottom: 8 }}>📁 Document Ingestion</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", marginBottom: 16 }}>
            Upload PDFs, text files, or CSVs with adaptive section parsing and vector indexing.
          </p>
          <span style={{ color: "var(--accent-color)", fontWeight: 600 }}>Upload Documents &rarr;</span>
        </Link>

        <Link
          href="/metrics"
          style={{
            padding: 24,
            borderRadius: 12,
            backgroundColor: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <h2 style={{ fontSize: "1.25rem", marginBottom: 8 }}>📊 Metrics & RAGAS</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", marginBottom: 16 }}>
            Inspect query latencies, cache hit rates, and run live RAGAS evaluation tests.
          </p>
          <span style={{ color: "var(--accent-color)", fontWeight: 600 }}>Inspect Metrics &rarr;</span>
        </Link>
      </div>

      {/* System Architecture Features */}
      <h2 style={{ fontSize: "1.4rem", marginBottom: 20 }}>System Capabilities & Enhancements</h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: 16,
        }}
      >
        {features.map((feat, idx) => (
          <div
            key={idx}
            style={{
              padding: 20,
              borderRadius: 8,
              backgroundColor: "rgba(30, 41, 59, 0.6)",
              border: "1px solid var(--border-color)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <h3 style={{ fontSize: "1.05rem", fontWeight: 600 }}>{feat.title}</h3>
              <span
                style={{
                  fontSize: "0.75rem",
                  padding: "2px 8px",
                  borderRadius: 4,
                  backgroundColor: "#334155",
                  color: "#93c5fd",
                }}
              >
                {feat.tag}
              </span>
            </div>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", lineHeight: 1.5 }}>
              {feat.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
