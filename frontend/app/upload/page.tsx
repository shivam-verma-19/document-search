"use client";

import { useState } from "react";
import api from "../apiClient";
import { useApi } from "../../hooks/useAPI";

const ALLOWED_EXTENSIONS = [".pdf", ".txt", ".csv"];
const MAX_SIZE_MB = 10;

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);
  const [deleteDocId, setDeleteDocId] = useState("");
  const [deleteStatus, setDeleteStatus] = useState<string | null>(null);
  const [recentUploads, setRecentUploads] = useState<Array<{ name: string; key: string; date: string }>>([]);

  const apiHook = useApi();

  const validateAndSetFile = (selected: File | null) => {
    setMessage(null);
    if (!selected) {
      setFile(null);
      return;
    }

    const ext = "." + selected.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setMessage({
        text: `Invalid file type. Allowed extensions: ${ALLOWED_EXTENSIONS.join(", ")}`,
        type: "error",
      });
      setFile(null);
      return;
    }

    if (selected.size > MAX_SIZE_MB * 1024 * 1024) {
      setMessage({
        text: `File size exceeds ${MAX_SIZE_MB}MB limit.`,
        type: "error",
      });
      setFile(null);
      return;
    }

    setFile(selected);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const uploadFile = async () => {
    if (!file) {
      setMessage({ text: "Please select a file to upload.", type: "error" });
      return;
    }

    setMessage({ text: "Uploading and queuing for adaptive ingestion...", type: "info" });
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await apiHook.post("/upload", formData);
      const uploadedKey = res.key || file.name;
      setMessage({
        text: `Successfully uploaded: ${uploadedKey}. Vector indexing and BM25 indexing in progress.`,
        type: "success",
      });
      setRecentUploads((prev) => [
        { name: file.name, key: uploadedKey, date: new Date().toLocaleTimeString() },
        ...prev,
      ]);
      setFile(null);
    } catch (err: any) {
      setMessage({
        text: err.message || "Upload failed. Please check backend connection.",
        type: "error",
      });
    }
  };

  const handleDeleteDocument = async () => {
    if (!deleteDocId.trim()) return;
    setDeleteStatus("Deleting document chunks from vector index...");
    try {
      const res = await api.delete(`/document/${deleteDocId.trim()}`);
      setDeleteStatus(`Document deleted successfully. Removed chunks count: ${res.data?.deleted_count ?? 1}`);
      setDeleteDocId("");
    } catch (err: any) {
      setDeleteStatus(`Failed to delete document: ${err.response?.data?.detail || err.message}`);
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: "0 auto" }}>
      <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: 8 }}>
        Document Ingestion & Management
      </h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: 24, fontSize: "0.95rem" }}>
        Ingest PDFs, text documents, or CSVs into the Hybrid RAG pipeline. Documents are automatically
        chunked using adaptive section boundary detection and indexed into S3 Vectors & DynamoDB BM25.
      </p>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${isDragging ? "var(--accent-color)" : "var(--border-color)"}`,
          borderRadius: 12,
          padding: "40px 24px",
          textAlign: "center",
          backgroundColor: isDragging ? "rgba(59, 130, 246, 0.08)" : "var(--bg-secondary)",
          transition: "all 0.2s ease",
          marginBottom: 20,
        }}
      >
        <p style={{ fontSize: "2.5rem", marginBottom: 12 }}>📤</p>
        <h3 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: 6 }}>
          Drag & Drop your document here
        </h3>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: 16 }}>
          Supports PDF, TXT, CSV up to {MAX_SIZE_MB}MB
        </p>

        <input
          type="file"
          id="fileInput"
          accept=".pdf,.txt,.csv"
          onChange={(e) => validateAndSetFile(e.target.files?.[0] || null)}
          style={{ display: "none" }}
        />
        <label
          htmlFor="fileInput"
          style={{
            display: "inline-block",
            padding: "8px 18px",
            borderRadius: 6,
            backgroundColor: "#334155",
            color: "#fff",
            fontSize: "0.9rem",
            fontWeight: 500,
            cursor: "pointer",
          }}
        >
          Browse Files
        </label>

        {file && (
          <div
            style={{
              marginTop: 16,
              padding: "10px 16px",
              borderRadius: 6,
              backgroundColor: "rgba(59, 130, 246, 0.15)",
              color: "#93c5fd",
              fontSize: "0.9rem",
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <span>📄</span>
            <span>
              {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </span>
          </div>
        )}
      </div>

      <button
        onClick={uploadFile}
        disabled={!file || apiHook.loading}
        style={{
          width: "100%",
          padding: "12px",
          borderRadius: 8,
          backgroundColor: "var(--accent-color)",
          color: "#fff",
          border: "none",
          fontWeight: 600,
          cursor: !file || apiHook.loading ? "not-allowed" : "pointer",
          opacity: !file || apiHook.loading ? 0.6 : 1,
          marginBottom: 24,
        }}
      >
        {apiHook.loading ? "Processing Upload..." : "Upload & Ingest Document"}
      </button>

      {message && (
        <div
          style={{
            padding: "12px 16px",
            borderRadius: 8,
            marginBottom: 24,
            fontSize: "0.9rem",
            backgroundColor:
              message.type === "success"
                ? "rgba(16, 185, 129, 0.15)"
                : message.type === "error"
                ? "rgba(239, 68, 68, 0.15)"
                : "rgba(59, 130, 246, 0.15)",
            color:
              message.type === "success"
                ? "var(--accent-success)"
                : message.type === "error"
                ? "var(--accent-danger)"
                : "var(--accent-color)",
            border: `1px solid ${
              message.type === "success"
                ? "var(--accent-success)"
                : message.type === "error"
                ? "var(--accent-danger)"
                : "var(--accent-color)"
            }`,
          }}
        >
          {message.text}
        </div>
      )}

      {/* Document Removal Section */}
      <div
        style={{
          padding: 20,
          borderRadius: 8,
          backgroundColor: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
          marginBottom: 24,
        }}
      >
        <h3 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: 8 }}>
          🗑 Delete Document by ID
        </h3>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: 12 }}>
          Remove all indexed chunks belonging to a document base ID from the S3 Vector index.
        </p>
        <div style={{ display: "flex", gap: 10 }}>
          <input
            type="text"
            placeholder="Enter doc_id (e.g. guide or UUID)"
            value={deleteDocId}
            onChange={(e) => setDeleteDocId(e.target.value)}
            style={{
              flex: 1,
              padding: "8px 12px",
              borderRadius: 6,
              background: "var(--bg-primary)",
              border: "1px solid var(--border-color)",
              color: "var(--text-primary)",
              fontSize: "0.9rem",
            }}
          />
          <button
            onClick={handleDeleteDocument}
            disabled={!deleteDocId.trim()}
            style={{
              padding: "8px 16px",
              borderRadius: 6,
              backgroundColor: "rgba(239, 68, 68, 0.2)",
              border: "1px solid var(--accent-danger)",
              color: "var(--accent-danger)",
              fontWeight: 600,
              cursor: !deleteDocId.trim() ? "not-allowed" : "pointer",
            }}
          >
            Delete
          </button>
        </div>
        {deleteStatus && (
          <p style={{ marginTop: 10, fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            {deleteStatus}
          </p>
        )}
      </div>

      {/* Recently Uploaded Documents */}
      {recentUploads.length > 0 && (
        <div
          style={{
            padding: 20,
            borderRadius: 8,
            backgroundColor: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 12 }}>
            Recently Ingested in this Session
          </h3>
          <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: 8 }}>
            {recentUploads.map((doc, idx) => (
              <li
                key={idx}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "8px 12px",
                  borderRadius: 6,
                  backgroundColor: "rgba(15, 23, 42, 0.4)",
                  fontSize: "0.85rem",
                }}
              >
                <span>📄 {doc.name}</span>
                <span style={{ color: "var(--text-secondary)" }}>{doc.date}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
