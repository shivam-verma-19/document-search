"use client";

import { useState, useEffect } from "react";
import { CognitoUser, AuthenticationDetails, CognitoUserPool, CognitoUserSession } from "amazon-cognito-identity-js";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const getUserPool = () => {
    const poolId = process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID;
    const clientId = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID;

    if (!poolId || !clientId) {
      throw new Error("Cognito credentials not configured");
    }

    return new CognitoUserPool({
      UserPoolId: poolId,
      ClientId: clientId,
    });
  };

  const signIn = () => {
    if (!email || !password) {
      setError("Email and password are required");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const userPool = getUserPool();
      const authDetails = new AuthenticationDetails({
        Username: email,
        Password: password,
      });

      const user = new CognitoUser({ Username: email, Pool: userPool });

      user.authenticateUser(authDetails, {
        onSuccess: (result: CognitoUserSession) => {
          const token = result.getAccessToken().getJwtToken();
          if (typeof window !== "undefined") {
            window.localStorage.setItem("access_token", token);
            window.location.href = "/";
          }
        },
        onFailure: (err: any) => {
          setError(err.message || "Login failed");
          setLoading(false);
        },
      });
    } catch (err: any) {
      setError(err.message || "Configuration error");
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 400, margin: "0 auto" }}>
      <h1>Login</h1>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
        disabled={loading}
        style={{ display: "block", width: "100%", marginBottom: 12, padding: 8 }}
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
        disabled={loading}
        style={{ display: "block", width: "100%", marginBottom: 12, padding: 8 }}
      />
      <button onClick={signIn} disabled={loading} style={{ width: "100%", padding: 10 }}>
        {loading ? "Signing in..." : "Sign In"}
      </button>
      {error && <p style={{ color: "red", marginTop: 12 }}>{error}</p>}
    </div>
  );
}