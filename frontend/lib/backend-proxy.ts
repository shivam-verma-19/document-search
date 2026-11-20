import { NextRequest, NextResponse } from "next/server";
import {
  ACCESS_COOKIE,
  REFRESH_COOKIE,
  clearAuthCookies,
  getAuthConfig,
  isAccessTokenUsable,
  isSameOriginRequest,
  refreshCognitoTokens,
  setAuthCookies,
  type CognitoTokens,
} from "./auth";
import { apiErrorResponse } from "./api-error-response";

const STATE_CHANGING_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);
const RESPONSE_HEADERS = [
  "cache-control",
  "content-disposition",
  "content-type",
  "x-accel-buffering",
];

function unauthorized(): NextResponse {
  return apiErrorResponse(
    401,
    "AUTHENTICATION_REQUIRED",
    "Please sign in again to continue.",
  );
}

function backendUrl(path: string[], request: NextRequest): URL {
  const configuredUrl =
    process.env.RAG_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000";
  const baseUrl = new URL(configuredUrl);
  const basePath = `${baseUrl.toString().replace(/\/+$/, "")}/`;
  const target = new URL(path.map(encodeURIComponent).join("/"), basePath);
  target.search = request.nextUrl.search;
  return target;
}

function isTimeoutError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  if (error.name === "TimeoutError") return true;

  const errorWithCause = error as Error & { cause?: unknown };
  const cause = errorWithCause.cause;
  if (!cause || typeof cause !== "object" || !("code" in cause)) return false;
  return (
    cause.code === "ETIMEDOUT" ||
    cause.code === "UND_ERR_CONNECT_TIMEOUT" ||
    cause.code === "UND_ERR_HEADERS_TIMEOUT"
  );
}

function attachRefreshedTokens(
  response: NextResponse,
  tokens: CognitoTokens | null,
  refreshToken?: string,
): NextResponse {
  if (tokens) setAuthCookies(response, tokens, refreshToken);
  return response;
}

export async function proxyToBackend(
  request: NextRequest,
  path: string[],
): Promise<NextResponse> {
  const isPublicHealthCheck =
    path.length === 0 && request.method === "GET";

  let authConfig;
  if (!isPublicHealthCheck) {
    try {
      authConfig = getAuthConfig();
    } catch (error) {
      console.error("Backend proxy authentication configuration error:", error);
      return apiErrorResponse(
        503,
        "AUTH_CONFIGURATION_ERROR",
        "Sign-in is temporarily unavailable. Please try again later.",
      );
    }

    if (
      STATE_CHANGING_METHODS.has(request.method) &&
      !isSameOriginRequest(request)
    ) {
      return apiErrorResponse(
        403,
        "INVALID_REQUEST_ORIGIN",
        "This request could not be verified. Refresh and try again.",
      );
    }
  }

  let target: URL;
  try {
    target = backendUrl(path, request);
  } catch (error) {
    console.error("Backend URL configuration error:", error);
    return apiErrorResponse(
      503,
      "BACKEND_CONFIGURATION_ERROR",
      "The backend service is not configured correctly.",
    );
  }

  const refreshToken = request.cookies.get(REFRESH_COOKIE)?.value;
  let accessToken = request.cookies.get(ACCESS_COOKIE)?.value;
  let refreshedTokens: CognitoTokens | null = null;
  let refreshed = false;

  if (!isPublicHealthCheck && (!accessToken || !isAccessTokenUsable(accessToken))) {
    if (!refreshToken) {
      const response = unauthorized();
      clearAuthCookies(response);
      return response;
    }
    try {
      refreshedTokens = await refreshCognitoTokens(refreshToken);
    } catch (error) {
      console.error("Cognito token refresh failed:", error);
      return apiErrorResponse(
        503,
        "AUTH_PROVIDER_UNAVAILABLE",
        "Sign-in is temporarily unavailable. Please try again later.",
      );
    }
    if (!refreshedTokens) {
      const response = unauthorized();
      clearAuthCookies(response);
      return response;
    }
    accessToken = refreshedTokens.access_token;
    refreshed = true;
  }

  let body: ArrayBuffer | undefined;
  try {
    body =
      request.method === "GET" || request.method === "HEAD"
        ? undefined
        : await request.arrayBuffer();
  } catch (error) {
    console.error("Could not read request body for backend:", error);
    return apiErrorResponse(
      400,
      "INVALID_REQUEST",
      "The request could not be read. Please try again.",
    );
  }

  const sendRequest = (token?: string) => {
    const headers = new Headers();
    const contentType = request.headers.get("content-type");
    const accept = request.headers.get("accept");
    if (contentType) headers.set("content-type", contentType);
    if (accept) headers.set("accept", accept);
    if (token) headers.set("authorization", `Bearer ${token}`);

    return fetch(target, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
      signal: request.signal,
    });
  };

  let upstream: Response;
  try {
    upstream = await sendRequest(accessToken);
    if (
      !isPublicHealthCheck &&
      upstream.status === 401 &&
      !refreshed &&
      refreshToken
    ) {
      try {
        refreshedTokens = await refreshCognitoTokens(refreshToken);
      } catch (error) {
        console.error("Cognito token refresh failed:", error);
        return apiErrorResponse(
          503,
          "AUTH_PROVIDER_UNAVAILABLE",
          "Sign-in is temporarily unavailable. Please try again later.",
        );
      }
      if (refreshedTokens) {
        accessToken = refreshedTokens.access_token;
        refreshed = true;
        upstream = await sendRequest(accessToken);
      }
    }
  } catch (error) {
    console.error("Backend request failed:", error);
    if (isTimeoutError(error)) {
      return attachRefreshedTokens(
        apiErrorResponse(
          504,
          "BACKEND_TIMEOUT",
          "The backend request timed out. Please try again.",
        ),
        refreshedTokens,
        refreshToken,
      );
    }
    return attachRefreshedTokens(
      apiErrorResponse(
        502,
        "BACKEND_UNAVAILABLE",
        "The backend service is unavailable. Check that it is running and try again.",
      ),
      refreshedTokens,
      refreshToken,
    );
  }

  if (upstream.status >= 500) {
    console.error("Backend returned an error response:", upstream.status);
    const code =
      upstream.status === 504
        ? "GATEWAY_TIMEOUT"
        : upstream.status === 502
          ? "BAD_GATEWAY"
          : upstream.status === 503
            ? "SERVICE_UNAVAILABLE"
            : "INTERNAL_SERVER_ERROR";
    return attachRefreshedTokens(
      apiErrorResponse(
        upstream.status,
        code,
        upstream.status === 504
          ? "The request took too long. Please try again."
          : upstream.status === 503
            ? "The service is temporarily unavailable. Please try again shortly."
            : "The service encountered a problem. Please try again.",
      ),
      refreshedTokens,
      refreshToken,
    );
  }

  const headers = new Headers();
  for (const headerName of RESPONSE_HEADERS) {
    const value = upstream.headers.get(headerName);
    if (value) headers.set(headerName, value);
  }
  const response = new NextResponse(upstream.body, {
    status: upstream.status,
    headers,
  });

  if (refreshedTokens) {
    setAuthCookies(response, refreshedTokens, refreshToken);
  }
  if (!isPublicHealthCheck && upstream.status === 401) {
    clearAuthCookies(response);
  }
  return response;
}
