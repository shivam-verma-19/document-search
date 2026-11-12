import { NextRequest, NextResponse } from "next/server";
import {
  ACCESS_COOKIE,
  REFRESH_COOKIE,
  clearAuthCookies,
  isAccessTokenUsable,
  refreshCognitoTokens,
  setAuthCookies,
  type CognitoTokens,
} from "../../../../lib/auth";

export async function GET(request: NextRequest) {
  const accessToken = request.cookies.get(ACCESS_COOKIE)?.value;
  const refreshToken = request.cookies.get(REFRESH_COOKIE)?.value;

  if (accessToken && isAccessTokenUsable(accessToken)) {
    return NextResponse.json({ authenticated: true });
  }
  if (!refreshToken) {
    const response = NextResponse.json({ authenticated: false });
    clearAuthCookies(response);
    return response;
  }

  let tokens: CognitoTokens | null;
  try {
    tokens = await refreshCognitoTokens(refreshToken);
  } catch (error) {
    console.error("Cognito session refresh failed:", error);
    return NextResponse.json(
      { detail: "Authentication provider is unavailable." },
      { status: 503 },
    );
  }

  if (!tokens) {
    const response = NextResponse.json({ authenticated: false });
    clearAuthCookies(response);
    return response;
  }

  const response = NextResponse.json({ authenticated: true });
  setAuthCookies(response, tokens, refreshToken);
  return response;
}
