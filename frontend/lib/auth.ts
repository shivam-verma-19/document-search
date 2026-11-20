import { NextRequest, NextResponse } from "next/server";

export const ACCESS_COOKIE = "rag_access";
export const REFRESH_COOKIE = "rag_refresh";
export const OAUTH_STATE_COOKIE = "rag_oauth_state";
export const OAUTH_VERIFIER_COOKIE = "rag_oauth_verifier";
export const RETURN_TO_COOKIE = "rag_auth_return_to";

export type CognitoTokens = {
  access_token: string;
  refresh_token?: string;
  expires_in?: number;
};

export type AuthConfig = {
  domain: string;
  clientId: string;
  baseUrl: URL;
  callbackUrl: string;
};

const ACCESS_REFRESH_BUFFER_SECONDS = 60;
const REFRESH_TOKEN_MAX_AGE_SECONDS = 30 * 24 * 60 * 60;

export function getAuthConfig(): AuthConfig {
  const domain = process.env.COGNITO_DOMAIN?.replace(/\/+$/, "");
  const clientId = process.env.COGNITO_CLIENT_ID;
  const baseUrlValue = process.env.AUTH_BASE_URL;

  if (!domain || !clientId || !baseUrlValue) {
    throw new Error(
      "Authentication is not configured. Set COGNITO_DOMAIN, COGNITO_CLIENT_ID, and AUTH_BASE_URL.",
    );
  }

  const domainUrl = new URL(domain);
  const baseUrl = new URL(baseUrlValue);
  if (domainUrl.protocol !== "https:") {
    throw new Error("COGNITO_DOMAIN must use HTTPS.");
  }
  if (baseUrl.pathname !== "/" || baseUrl.search || baseUrl.hash) {
    throw new Error("AUTH_BASE_URL must be an origin without a path or query.");
  }
  if (process.env.NODE_ENV === "production" && baseUrl.protocol !== "https:") {
    throw new Error("AUTH_BASE_URL must use HTTPS in production.");
  }

  return {
    domain,
    clientId,
    baseUrl,
    callbackUrl: new URL("/api/auth/callback", baseUrl).toString(),
  };
}

export function getCookieOptions(path: string, maxAge: number) {
  return {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax" as const,
    path,
    maxAge,
  };
}

export function getTokenExpiry(token: string): number | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const claims: { exp?: unknown } = JSON.parse(
      Buffer.from(payload, "base64url").toString("utf8"),
    );
    return typeof claims.exp === "number" ? claims.exp : null;
  } catch {
    return null;
  }
}

export function isAccessTokenUsable(token: string): boolean {
  const expiry = getTokenExpiry(token);
  return (
    expiry !== null &&
    expiry > Math.floor(Date.now() / 1000) + ACCESS_REFRESH_BUFFER_SECONDS
  );
}

export function setAuthCookies(
  response: NextResponse,
  tokens: CognitoTokens,
  existingRefreshToken?: string,
): void {
  const expiry = getTokenExpiry(tokens.access_token);
  if (expiry === null) {
    throw new Error("Cognito returned an access token without a valid expiry.");
  }

  const accessMaxAge = Math.max(0, expiry - Math.floor(Date.now() / 1000));
  response.cookies.set(
    ACCESS_COOKIE,
    tokens.access_token,
    getCookieOptions("/api", accessMaxAge),
  );

  const refreshToken = tokens.refresh_token ?? existingRefreshToken;
  if (refreshToken) {
    response.cookies.set(
      REFRESH_COOKIE,
      refreshToken,
      getCookieOptions("/api", REFRESH_TOKEN_MAX_AGE_SECONDS),
    );
  }
}

export function clearAuthCookies(response: NextResponse): void {
  for (const name of [ACCESS_COOKIE, REFRESH_COOKIE]) {
    response.cookies.set(name, "", getCookieOptions("/api", 0));
  }
  for (const name of [
    OAUTH_STATE_COOKIE,
    OAUTH_VERIFIER_COOKIE,
    RETURN_TO_COOKIE,
  ]) {
    response.cookies.set(name, "", getCookieOptions("/api/auth", 0));
  }
}

export function clearOAuthCookies(response: NextResponse): void {
  for (const name of [
    OAUTH_STATE_COOKIE,
    OAUTH_VERIFIER_COOKIE,
    RETURN_TO_COOKIE,
  ]) {
    response.cookies.set(name, "", getCookieOptions("/api/auth", 0));
  }
}

export function isSameOriginRequest(request: NextRequest): boolean {
  try {
    const { baseUrl } = getAuthConfig();
    return request.headers.get("origin") === baseUrl.origin;
  } catch {
    return false;
  }
}

export async function refreshCognitoTokens(
  refreshToken: string,
): Promise<CognitoTokens | null> {
  const { domain, clientId } = getAuthConfig();
  const response = await fetch(`${domain}/oauth2/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      client_id: clientId,
      refresh_token: refreshToken,
    }),
    cache: "no-store",
    signal: AbortSignal.timeout(10000),
  });

  if (response.status === 400 || response.status === 401) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Cognito token refresh failed with HTTP ${response.status}.`);
  }

  const tokens: Partial<CognitoTokens> = await response.json();
  if (typeof tokens.access_token !== "string") {
    throw new Error("Cognito token refresh returned no access token.");
  }
  return {
    access_token: tokens.access_token,
    refresh_token:
      typeof tokens.refresh_token === "string" ? tokens.refresh_token : undefined,
    expires_in: typeof tokens.expires_in === "number" ? tokens.expires_in : undefined,
  };
}

export function getReturnTo(value: string | null, baseUrl: URL): string {
  if (!value || !value.startsWith("/") || value.startsWith("//")) {
    return "/";
  }
  const destination = new URL(value, baseUrl);
  if (destination.origin !== baseUrl.origin) {
    return "/";
  }
  return `${destination.pathname}${destination.search}${destination.hash}`;
}
