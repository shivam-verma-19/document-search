"use client";

import { useState } from "react";
import {
  CognitoUser,
  AuthenticationDetails,
  CognitoUserPool,
  CognitoUserSession,
} from "amazon-cognito-identity-js";

const COGNITO_ERROR_MESSAGES: Record<string, string> = {
  NotAuthorizedException: "Incorrect email or password.",
  UserNotFoundException: "No account found with this email.",
  UserNotConfirmedException: "Please verify your email before logging in.",
  PasswordResetRequiredException: "Your password must be reset. Check your email.",
  TooManyRequestsException: "Too many attempts. Please wait and try again.",
  NetworkError: "Network error. Check your connection.",
};

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const getUserPool = () => {
    const poolId = process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID;
    const clientId = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID;

    if (!poolId || !clientId) {
      throw new Error("Authentication is not configured. Contact support.");
    }

    return new CognitoUserPool({ UserPoolId: poolId, ClientId: clientId });
  };

  const validate = (): boolean => {
    if (!email.trim()) {
      setError("Email is required.");
      return false;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("Enter a valid email address.");
      return false;
    }
    if (!password) {
      setError("Password is required.");
      return false;
    }
    return true;
  };

  const signIn = () => {
    setError(null);

    if (!validate()) return;

    setLoading(true);

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
          const friendly =
            COGNITO_ERROR_MESSAGES[err.code] ||
            err.message ||
            "Login failed. Please try again.";
          setError(friendly);
          setLoading(false);
        },
      });
    } catch (err: any) {
      setError(err.message || "Configuration error. Contact support.");
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !loading) signIn();
  };

  return (
    <div style={{ padding: 24, maxWidth: 400, margin: "0 auto" }}>
      <h1>Login</h1>

      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Email"
        disabled={loading}
        style={{ display: "block", width: "100%", marginBottom: 12, padding: 8 }}
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Password"
        disabled={loading}
        style={{ display: "block", width: "100%", marginBottom: 12, padding: 8 }}
      />

      <button
        onClick={signIn}
        disabled={loading}
        style={{ width: "100%", padding: 10 }}
      >
        {loading ? "Signing in..." : "Sign In"}
      </button>

      {error && (
        <p style={{ color: "red", marginTop: 12 }}>{error}</p>
      )}
    </div>
  );
}
