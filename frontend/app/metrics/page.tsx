"use client";

import { useEffect, useState } from "react";
import axios from "axios";

export default function MetricsPage() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    axios.get("/api/metrics").then((res) => {
      setData(res.data);
    });
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h1>📊 Metrics Dashboard</h1>

      {data.length === 0 && <p>No data available</p>}

      {data.map((item, idx) => (
        <div key={idx} style={{ marginBottom: "10px", borderBottom: "1px solid #ccc" }}>
          <p><b>Query:</b> {item.query}</p>
          <p><b>Latency:</b> {item.latency?.toFixed(3)} sec</p>
          <p><b>Source:</b> {item.source}</p>
        </div>
      ))}
    </div>
  );
}