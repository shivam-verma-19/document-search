"use client";

import { useEffect, useRef, useState } from "react";
import api from "../apiClient";

interface SourceItem {
  doc_id?: string;
  text: string;
  score?: number;
  metadata?: Record<string, any>;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  sub_queries?: string[];
  sources?: SourceItem[];
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [useStreaming, setUseStreaming] = useState(true);

  // Metadata filter state
  const [showFilters, setShowFilters] = useState(false);
  const [fileType, setFileType] = useState("");
  const [sectionFilter, setSectionFilter] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize or restore session ID
  useEffect(() => {
    let sid = window.localStorage.getItem("rag_session_id");
    if (!sid) {
      sid = "sess_" + Math.random().toString(36).substring(2, 10);
      window.localStorage.setItem("rag_session_id", sid);
    }
    setSessionId(sid);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const resetSession = async () => {
    if (sessionId) {
      try {
        await api.delete(`/session/${sessionId}`);
      } catch (err) {
        console.warn("Could not clear session on backend:", err);
      }
    }
    const newSid = "sess_" + Math.random().toString(36).substring(2, 10);
    setSessionId(newSid);
    window.localStorage.setItem("rag_session_id", newSid);
    setMessages([]);
  };

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputQuery.trim() || loading) return;

    const userText = inputQuery.trim();
    setInputQuery("");

    const userMessage: Message = {
      id: "u_" + Date.now(),
      role: "user",
      content: userText,
      timestamp: new Date().toLocaleTimeString(),
    };

    const assistantPlaceholderId = "a_" + Date.now();
    const assistantMessage: Message = {
      id: assistantPlaceholderId,
      role: "assistant",
      content: "",
      timestamp: new Date().toLocaleTimeString(),
      sources: [],
      sub_queries: [],
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setLoading(true);

    const queryParams: Record<string, string> = {
      q: userText,
      session_id: sessionId,
    };
    if (fileType) queryParams.file_type = fileType;
    if (sectionFilter) queryParams.section = sectionFilter;
    if (dateFrom) queryParams.date_from = dateFrom;
    if (dateTo) queryParams.date_to = dateTo;

    if (useStreaming) {
      try {
        const searchStr = new URLSearchParams(queryParams).toString();
        const url = `/api/backend/ask/stream?${searchStr}`;
        const response = await fetch(url, {
          credentials: "same-origin",
          headers: { Accept: "text/event-stream" },
        });

        if (!response.ok || !response.body) {
          throw new Error(`Streaming failed: HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let streamedAnswer = "";
        let collectedSources: SourceItem[] = [];
        let collectedSubQueries: string[] = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data:")) {
              try {
                const payload = JSON.parse(trimmed.slice(5).trim());
                if (payload.type === "token") {
                  streamedAnswer += payload.content || "";
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantPlaceholderId
                        ? { ...msg, content: streamedAnswer }
                        : msg
                    )
                  );
                } else if (payload.type === "metadata") {
                  if (payload.sources) collectedSources = payload.sources;
                  if (payload.sub_queries) collectedSubQueries = payload.sub_queries;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantPlaceholderId
                        ? {
                            ...msg,
                            sources: collectedSources,
                            sub_queries: collectedSubQueries,
                          }
                        : msg
                    )
                  );
                } else if (payload.type === "error") {
                  streamedAnswer += `\n[Error: ${payload.error}]`;
                }
              } catch (parseErr) {
                console.warn("SSE parse error:", parseErr);
              }
            }
          }
        }
      } catch (err: any) {
        console.warn("Streaming error, falling back to sync:", err);
        // Fallback to standard /ask endpoint
        try {
          const res = await api.get("/ask", { params: queryParams });
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantPlaceholderId
                ? {
                    ...msg,
                    content: res.data.answer || "No response received.",
                    sources: res.data.sources || [],
                    sub_queries: res.data.sub_queries || [],
                  }
                : msg
            )
          );
        } catch (apiErr: any) {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantPlaceholderId
                ? { ...msg, content: `Error: ${apiErr.message || "Failed to get response"}` }
                : msg
            )
          );
        }
      } finally {
        setLoading(false);
      }
    } else {
      // Synchronous ask
      try {
        const res = await api.get("/ask", { params: queryParams });
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantPlaceholderId
              ? {
                  ...msg,
                  content: res.data.answer || "No response received.",
                  sources: res.data.sources || [],
                  sub_queries: res.data.sub_queries || [],
                }
              : msg
          )
        );
      } catch (err: any) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantPlaceholderId
              ? { ...msg, content: `Error: ${err.message || "Request failed"}` }
              : msg
          )
        );
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 130px)" }}>
      {/* Top Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          paddingBottom: 16,
          borderBottom: "1px solid var(--border-color)",
          marginBottom: 16,
        }}
      >
        <div>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 700 }}>Interactive RAG Chat</h1>
          <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            Session ID: <code style={{ color: "#93c5fd" }}>{sessionId || "loading..."}</code>
          </span>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <label style={{ fontSize: "0.85rem", display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
            />
            Stream SSE
          </label>
          <button
            onClick={() => setShowFilters(!showFilters)}
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              background: showFilters ? "#334155" : "transparent",
              border: "1px solid var(--border-color)",
              color: "var(--text-primary)",
              cursor: "pointer",
              fontSize: "0.85rem",
            }}
          >
            {showFilters ? "Hide Filters ▴" : "Filters ▾"}
          </button>
          <button
            onClick={resetSession}
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              background: "rgba(239, 68, 68, 0.15)",
              border: "1px solid var(--accent-danger)",
              color: "var(--accent-danger)",
              cursor: "pointer",
              fontSize: "0.85rem",
              fontWeight: 500,
            }}
          >
            New Chat / Reset
          </button>
        </div>
      </div>

      {/* Expandable Filter Bar */}
      {showFilters && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: 12,
            padding: 16,
            backgroundColor: "var(--bg-secondary)",
            borderRadius: 8,
            border: "1px solid var(--border-color)",
            marginBottom: 16,
          }}
        >
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>
              File Type
            </label>
            <select
              value={fileType}
              onChange={(e) => setFileType(e.target.value)}
              style={{
                width: "100%",
                padding: "6px 8px",
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
                color: "var(--text-primary)",
                borderRadius: 4,
              }}
            >
              <option value="">All Types</option>
              <option value="pdf">PDF</option>
              <option value="txt">TXT</option>
              <option value="csv">CSV</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>
              Section Header Filter
            </label>
            <input
              type="text"
              placeholder="e.g. Architecture"
              value={sectionFilter}
              onChange={(e) => setSectionFilter(e.target.value)}
              style={{
                width: "100%",
                padding: "6px 8px",
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
                color: "var(--text-primary)",
                borderRadius: 4,
              }}
            />
          </div>
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>
              Date From
            </label>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              style={{
                width: "100%",
                padding: "5px 8px",
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
                color: "var(--text-primary)",
                borderRadius: 4,
              }}
            />
          </div>
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "block", marginBottom: 4 }}>
              Date To
            </label>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              style={{
                width: "100%",
                padding: "5px 8px",
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
                color: "var(--text-primary)",
                borderRadius: 4,
              }}
            />
          </div>
        </div>
      )}

      {/* Messages Scroll Area */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          paddingRight: 8,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        {messages.length === 0 ? (
          <div
            style={{
              margin: "auto",
              textAlign: "center",
              color: "var(--text-secondary)",
              maxWidth: 400,
            }}
          >
            <p style={{ fontSize: "2rem", marginBottom: 12 }}>💬</p>
            <h3 style={{ fontSize: "1.1rem", marginBottom: 8, color: "var(--text-primary)" }}>
              No messages yet
            </h3>
            <p style={{ fontSize: "0.9rem" }}>
              Ask a question about your ingested documents. The engine will retrieve relevant chunks and cite sources.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: msg.role === "user" ? "flex-end" : "flex-start",
              }}
            >
              <div
                style={{
                  maxWidth: "80%",
                  padding: "12px 16px",
                  borderRadius: 12,
                  backgroundColor: msg.role === "user" ? "var(--accent-color)" : "var(--bg-secondary)",
                  color: "var(--text-primary)",
                  border: msg.role === "user" ? "none" : "1px solid var(--border-color)",
                  lineHeight: 1.6,
                  whiteSpace: "pre-wrap",
                }}
              >
                {msg.content || (loading && msg.role === "assistant" ? "Thinking..." : "")}

                {/* Sub-queries disclosure */}
                {msg.sub_queries && msg.sub_queries.length > 1 && (
                  <div
                    style={{
                      marginTop: 12,
                      padding: "8px 12px",
                      borderRadius: 6,
                      backgroundColor: "rgba(0,0,0,0.25)",
                      fontSize: "0.85rem",
                    }}
                  >
                    <span style={{ fontWeight: 600, color: "#93c5fd" }}>Planned Sub-Queries:</span>
                    <ul style={{ paddingLeft: 18, marginTop: 4 }}>
                      {msg.sub_queries.map((sq, i) => (
                        <li key={i}>{sq}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Citations & Sources disclosure */}
                {msg.sources && msg.sources.length > 0 && (
                  <details style={{ marginTop: 12, fontSize: "0.85rem", cursor: "pointer" }}>
                    <summary style={{ color: "#93c5fd", fontWeight: 600 }}>
                      Retrieved Sources ({msg.sources.length} chunks)
                    </summary>
                    <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
                      {msg.sources.map((src, i) => (
                        <div
                          key={i}
                          style={{
                            padding: 8,
                            borderRadius: 6,
                            backgroundColor: "rgba(15, 23, 42, 0.6)",
                            border: "1px solid var(--border-color)",
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                            <span style={{ fontWeight: 600, fontSize: "0.75rem", color: "#60a5fa" }}>
                              {src.metadata?.section ? `Section: ${src.metadata.section}` : `Doc: ${src.doc_id || i + 1}`}
                            </span>
                            {src.score !== undefined && (
                              <span style={{ fontSize: "0.75rem", color: "var(--accent-success)" }}>
                                Score: {(src.score * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                          <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", maxHeight: 80, overflowY: "auto" }}>
                            {src.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: 4, padding: "0 4px" }}>
                {msg.role === "user" ? "You" : "Assistant"} &bull; {msg.timestamp}
              </span>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSend} style={{ display: "flex", gap: 10, marginTop: 16 }}>
        <input
          type="text"
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          placeholder="Ask a question about your documents..."
          disabled={loading}
          style={{
            flex: 1,
            padding: "12px 16px",
            borderRadius: 8,
            backgroundColor: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
            color: "var(--text-primary)",
            fontSize: "0.95rem",
            outline: "none",
          }}
        />
        <button
          type="submit"
          disabled={loading || !inputQuery.trim()}
          style={{
            padding: "12px 24px",
            borderRadius: 8,
            backgroundColor: "var(--accent-color)",
            border: "none",
            color: "#fff",
            fontWeight: 600,
            cursor: loading || !inputQuery.trim() ? "not-allowed" : "pointer",
            opacity: loading || !inputQuery.trim() ? 0.6 : 1,
          }}
        >
          {loading ? "Searching..." : "Ask"}
        </button>
      </form>
    </div>
  );
}
