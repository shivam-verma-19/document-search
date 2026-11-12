import { NextRequest, NextResponse } from "next/server";
import {
  clearAuthCookies,
  clearOAuthCookies,
  getAuthConfig,
  getReturnTo,
  OAUTH_STATE_COOKIE,
  OAUTH_VERIFIER_COOKIE,
  RETURN_TO_COOKIE,
  setAuthCookies,
  type CognitoTokens,
} from "../../../../lib/auth";

function failedLogin(baseUrl: URL, message: string): NextResponse {
  const response = NextResponse.redirect(
    new URL("/login?error=callback_failed", baseUrl),
  );
  clearAuthCookies(response);
  console.error("Cognito OAuth callback failed:", message);
  return response;
}

export async function GET(request: NextRequest) {
  let config;
  try {
    config = getAuthConfig();
  } catch (error) {
    console.error("Cognito callback configuration error:", error);
    return NextResponse.json(
      { detail: "Authentication is not configured." },
      { status: 503 },
    );
  }

  const returnedState = request.nextUrl.searchParams.get("state");
  const expectedState = request.cookies.get(OAUTH_STATE_COOKIE)?.value;
  const verifier = request.cookies.get(OAUTH_VERIFIER_COOKIE)?.value;
  const code = request.nextUrl.searchParams.get("code");
  const returnTo = request.cookies.get(RETURN_TO_COOKIE)?.value || "/";

  if (
    !returnedState ||
    !expectedState ||
    returnedState !== expectedState ||
    !verifier ||
    !code
  ) {
    return failedLogin(config.baseUrl, "Invalid OAuth state or missing code.");
  }

  let tokens: CognitoTokens;
  try {
    const tokenResponse = await fetch(`${config.domain}/oauth2/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "authorization_code",
        client_id: config.clientId,
        code,
        redirect_uri: config.callbackUrl,
        code_verifier: verifier,
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(10000),
    });

    if (!tokenResponse.ok) {
      return failedLogin(
        config.baseUrl,
        `Token exchange returned HTTP ${tokenResponse.status}.`,
      );
    }

    const responseBody = (await tokenResponse.json()) as Partial<CognitoTokens>;
    if (
      typeof responseBody.access_token !== "string" ||
      typeof responseBody.refresh_token !== "string"
    ) {
      return failedLogin(
        config.baseUrl,
        "Token exchange did not return an access and refresh token.",
      );
    }
    tokens = {
      access_token: responseBody.access_token,
      refresh_token: responseBody.refresh_token,
      expires_in: responseBody.expires_in,
    };
  } catch (error) {
    return failedLogin(
      config.baseUrl,
      error instanceof Error ? error.message : "Token exchange failed.",
    );
  }

  const destination = getReturnTo(returnTo, config.baseUrl);
  const response = NextResponse.redirect(new URL(destination, config.baseUrl));
  setAuthCookies(response, tokens);
  clearOAuthCookies(response);
  return response;
}
