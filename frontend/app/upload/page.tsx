"use client";

import { useState } from "react";
import { useApi } from "@/hooks/useApi";

const ALLOWED_TYPES = ["application/pdf", "text/plain", "text/csv"];
const MAX_SIZE_MB = 10;

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [validationError, setValidationError] = useState("");
  const api = useApi();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] || null;
    setValidationError("");
    setMessage("");

    if (!selected) {
      setFile(null);
      return;
    }

    if (!ALLOWED_TYPES.includes(selected.type)) {
      setValidationError("Only PDF, TXT, and CSV files are allowed.");
      setFile(null);
      return;
    }

    if (selected.size > MAX_SIZE_MB * 1024 * 1024) {
      setValidationError(`File must be under ${MAX_SIZE_MB}MB.`);
      setFile(null);
      return;
    }

    setFile(selected);
  };

  const upload = async () => {
    if (!file) {
      setValidationError("Please select a file.");
      return;
    }

    setMessage("");
    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await api.post("/upload", fd);
      setMessage(`Uploaded: ${res.key || file.name}`);
      setFile(null);
    } catch (err: any) {
      setMessage(err.message || "Upload failed.");
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <input
        type="file"
        accept=".pdf,.txt,.csv"
        onChange={handleFileChange}
        disabled={api.loading}
      />

      {validationError && (
        <p style={{ color: "red", marginTop: 8 }}>{validationError}</p>
      )}

      <button
        onClick={upload}
        disabled={api.loading || !file}
        style={{ marginTop: 12, display: "block" }}
      >
        {api.loading ? "Uploading..." : "Upload"}
      </button>

      {api.error && (
        <p style={{ color: "red", marginTop: 8 }}>
          {api.error.message}
          {api.error.retryable && (
            <button onClick={upload} style={{ marginLeft: 8 }}>
              Retry
            </button>
          )}
        </p>
      )}

      {message && <p style={{ marginTop: 8 }}>{message}</p>}
    </div>
  );
}
