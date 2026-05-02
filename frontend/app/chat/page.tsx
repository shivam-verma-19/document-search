"use client";
import { useState } from "react";
import api from "../apiClient";

export default function Chat() {
  const [q, setQ] = useState("");
  const [a, setA] = useState("");
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    setLoading(true);
    try {
      const res = await api.get("/ask", { params: { q } });
      setA(res.data.answer);
    } catch (err: any) {
      setA(err.response?.data?.detail || "Error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a question" />
      <button onClick={ask} disabled={loading}>
        {loading ? "Loading..." : "Ask"}
      </button>
      <p>{a}</p>
    </div>
  );
}