"use client";
import { useState } from "react";
import api from "../apiClient";

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const upload = async () => {
    if (!file) {
      setMessage("Please select a file");
      return;
    }

    setLoading(true);
    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await api.post("/upload", fd);
      setMessage(`Uploaded: ${res.data.key}`);
      setFile(null);
    } catch (err: any) {
      setMessage(err.response?.data?.detail || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        disabled={loading}
      />
      <button onClick={upload} disabled={loading || !file}>
        {loading ? "Uploading..." : "Upload"}
      </button>
      {message && <p>{message}</p>}
    </div>
  );
}