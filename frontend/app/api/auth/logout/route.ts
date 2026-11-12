import { NextRequest, NextResponse } from "next/server";
import {
  clearAuthCookies,
  getAuthConfig,
  isSameOriginRequest,
  REFRESH_COOKIE,
} from "../../../../lib/auth";

export async function POST(request: NextRequest) {
  let config;
  try {
    config = getAuthConfig();
  } catch (error) {
    console.error("Cognito logout configuration error:", error);
    const response = NextResponse.json(
      { detail: "Authentication is not configured." },
      { status: 503 },
    );
    clearAuthCookies(response);
    return response;
  }

  if (!isSameOriginRequest(request)) {
    return NextResponse.json({ detail: "Invalid request origin." }, { status: 403 });
  }

  const logoutUrl = new URL("/logout", config.domain);
  logoutUrl.search = new URLSearchParams({
    client_id: config.clientId,
    logout_uri: config.baseUrl.origin,
  }).toString();

  let revocationWarning = false;
  const refreshToken = request.cookies.get(REFRESH_COOKIE)?.value;
  if (refreshToken) {
    try {
      const revokeResponse = await fetch(`${config.domain}/oauth2/revoke`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          client_id: config.clientId,
          token: refreshToken,
        }),
        cache: "no-store",
        signal: AbortSignal.timeout(5000),
      });
      if (!revokeResponse.ok) {
        revocationWarning = true;
        console.error(
          "Cognito refresh-token revocation failed with HTTP",
          revokeResponse.status,
        );
      }
    } catch (error) {
      revocationWarning = true;
      console.error("Cognito refresh-token revocation failed:", error);
    }
  }

  const response = NextResponse.json({
    logoutUrl: logoutUrl.toString(),
    revocationWarning,
  });
  clearAuthCookies(response);
  return response;
}
