"use client";

import Link from "next/link";
import { useAuth } from "./AuthProvider";

export default function AuthControls() {
  const { status, signOut } = useAuth();

  if (status === "checking") {
    return <span aria-live="polite">Checking session...</span>;
  }
  if (status === "authenticated") {
    return (
      <button
        type="button"
        onClick={() => void signOut()}
        className="nav-link"
        style={{ background: "transparent", border: "1px solid var(--border-color)" }}
      >
        Sign out
      </button>
    );
  }

  return (
    <Link href="/login" className="nav-link">
      Sign in
    </Link>
  );
}
