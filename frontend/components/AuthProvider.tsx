"use client";

import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";

type AuthStatus = "checking" | "authenticated" | "anonymous" | "error";

type AuthContextValue = {
  status: AuthStatus;
  refreshSession: () => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

function isProtectedPath(pathname: string): boolean {
  return ["/chat", "/upload", "/metrics"].some(
    (path) => pathname === path || pathname.startsWith(`${path}/`),
  );
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("checking");
  const pathname = usePathname();
  const router = useRouter();

  const refreshSession = useCallback(async () => {
    setStatus("checking");
    try {
      const response = await fetch("/api/auth/session", {
        cache: "no-store",
        credentials: "same-origin",
      });
      if (!response.ok) {
        throw new Error(`Session check failed with HTTP ${response.status}.`);
      }
      const result = (await response.json()) as { authenticated: boolean };
      setStatus(result.authenticated ? "authenticated" : "anonymous");
    } catch (error) {
      console.error("Could not verify the sign-in session:", error);
      setStatus("error");
    }
  }, []);

  const signOut = useCallback(async () => {
    try {
      const response = await fetch("/api/auth/logout", {
        method: "POST",
        credentials: "same-origin",
      });
      setStatus("anonymous");
      if (response.ok) {
        const result = (await response.json()) as {
          logoutUrl?: string;
          revocationWarning?: boolean;
        };
        if (result.revocationWarning) {
          console.warn("Cognito could not revoke the previous refresh token.");
        }
        if (result.logoutUrl) {
          window.location.assign(result.logoutUrl);
          return;
        }
      }
    } catch (error) {
      console.error("Could not complete Cognito logout:", error);
      setStatus("anonymous");
    }
    window.location.assign("/login");
  }, []);

  useEffect(() => {
    window.localStorage.removeItem("access_token");
    void refreshSession();
  }, [refreshSession]);

  useEffect(() => {
    const handleAuthRequired = () => setStatus("anonymous");
    window.addEventListener("rag:auth-required", handleAuthRequired);
    return () =>
      window.removeEventListener("rag:auth-required", handleAuthRequired);
  }, []);

  useEffect(() => {
    if (status === "anonymous" && isProtectedPath(pathname)) {
      router.replace(`/login?returnTo=${encodeURIComponent(pathname)}`);
    }
  }, [pathname, router, status]);

  const value = useMemo(
    () => ({ status, refreshSession, signOut }),
    [refreshSession, signOut, status],
  );

  if (isProtectedPath(pathname) && status !== "authenticated") {
    return (
      <AuthContext.Provider value={value}>
        <main className="main-container" role="status">
          {status === "error" ? (
            <section>
              <h1>Unable to verify your sign-in</h1>
              <p>Please retry the session check or sign in again.</p>
              <button onClick={() => void refreshSession()}>Retry</button>
              <Link href="/login">Sign in</Link>
            </section>
          ) : (
            <p>Checking your sign-in session...</p>
          )}
        </main>
      </AuthContext.Provider>
    );
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider.");
  }
  return context;
}
