"use client";

import { useEffect } from "react";
import { useApi } from "../../hooks/useAPI";

export default function MetricsPage() {
  const api = useApi();

  useEffect(() => {
    api.get("/metrics");
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h1>📊 Metrics Dashboard</h1>

      {api.loading && <p>Loading...</p>}

      {api.error && (
        <p style={{ color: "red" }}>
          {api.error.message}
          {api.error.retryable && (
            <button onClick={() => api.get("/metrics")} style={{ marginLeft: 8 }}>
              Retry
            </button>
          )}
        </p>
      )}

      {api.data && api.data.length === 0 && <p>No data available</p>}

      {api.data?.map((item: any, idx: number) => (
        <div
          key={idx}
          style={{ marginBottom: "10px", borderBottom: "1px solid #ccc" }}
        >
          <p>
            <b>Query:</b> {item.query}
          </p>
          <p>
            <b>Latency:</b> {item.latency?.toFixed(3)} sec
          </p>
          <p>
            <b>Source:</b> {item.source}
          </p>
        </div>
      ))}
    </div>
  );
}
