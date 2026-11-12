import { createHash, randomBytes } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import {
  getAuthConfig,
  getCookieOptions,
  getReturnTo,
  OAUTH_STATE_COOKIE,
  OAUTH_VERIFIER_COOKIE,
  RETURN_TO_COOKIE,
} from "../../../../lib/auth";

const OAUTH_COOKIE_MAX_AGE_SECONDS = 10 * 60;

export async function GET(request: NextRequest) {
  let config;
  try {
    config = getAuthConfig();
  } catch (error) {
    console.error("Cognito login configuration error:", error);
    return NextResponse.json(
      { detail: "Authentication is not configured." },
      { status: 503 },
    );
  }

  const state = randomBytes(32).toString("base64url");
  const verifier = randomBytes(32).toString("base64url");
  const challenge = createHash("sha256").update(verifier).digest("base64url");
  const authorizeUrl = new URL("/oauth2/authorize", config.domain);
  authorizeUrl.search = new URLSearchParams({
    client_id: config.clientId,
    response_type: "code",
    scope: "openid email profile",
    redirect_uri: config.callbackUrl,
    state,
    code_challenge_method: "S256",
    code_challenge: challenge,
  }).toString();

  const response = NextResponse.redirect(authorizeUrl);
  const cookieOptions = getCookieOptions(
    "/api/auth",
    OAUTH_COOKIE_MAX_AGE_SECONDS,
  );
  response.cookies.set(OAUTH_STATE_COOKIE, state, cookieOptions);
  response.cookies.set(OAUTH_VERIFIER_COOKIE, verifier, cookieOptions);
  response.cookies.set(
    RETURN_TO_COOKIE,
    getReturnTo(request.nextUrl.searchParams.get("returnTo"), config.baseUrl),
    cookieOptions,
  );
  return response;
}
