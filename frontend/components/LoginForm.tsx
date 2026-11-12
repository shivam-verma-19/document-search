"use client";

import { useEffect, useState } from "react";

export default function LoginForm() {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const errorCode = new URLSearchParams(window.location.search).get("error");
    if (errorCode === "callback_failed") {
      setError("Sign-in could not be completed. Please try again.");
    }
  }, []);

  const startSignIn = () => {
    const params = new URLSearchParams(window.location.search);
    const returnTo = params.get("returnTo") || "/";
    window.location.assign(
      `/api/auth/login?returnTo=${encodeURIComponent(returnTo)}`,
    );
  };

  return (
    <section
      style={{
        padding: 28,
        maxWidth: 480,
        margin: "48px auto",
        border: "1px solid var(--border-color)",
        borderRadius: 12,
        background: "var(--bg-secondary)",
      }}
    >
      <h1 style={{ marginBottom: 12 }}>Sign in to RAG Platform</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: 20 }}>
        Use your username and password, or create an account. New accounts
        confirm a verified email address; your username remains your sign-in
        name.
      </p>
      <button
        type="button"
        onClick={startSignIn}
        style={{ width: "100%", padding: 12 }}
      >
        Continue to sign in or create an account
      </button>
      {error && (
        <p role="alert" style={{ color: "var(--accent-danger)", marginTop: 12 }}>
          {error}
        </p>
      )}
      <p
        style={{
          color: "var(--text-secondary)",
          fontSize: "0.85rem",
          marginTop: 16,
        }}
      >
        Password reset and account confirmation are handled securely by
        Amazon Cognito.
      </p>
    </section>
  );
}
