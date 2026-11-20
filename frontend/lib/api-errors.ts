type ErrorEnvelope = {
  error?: {
    code?: unknown;
    message?: unknown;
  };
};

const SAFE_CODE_MESSAGES: Record<string, string> = {
  AUTHENTICATION_REQUIRED: "Your session has expired. Please sign in again.",
  FORBIDDEN: "You don't have permission to perform this action.",
  NOT_FOUND: "The requested resource was not found.",
  METHOD_NOT_ALLOWED: "This action is not supported.",
  PAYLOAD_TOO_LARGE: "The uploaded file exceeds the allowed size.",
  UNSUPPORTED_MEDIA_TYPE: "This file type is not supported.",
  VALIDATION_ERROR: "Please check your input and try again.",
  RATE_LIMITED: "Too many requests. Wait a moment and try again.",
  INTERNAL_SERVER_ERROR: "The service encountered a problem. Please try again.",
  BAD_GATEWAY: "The service is temporarily unavailable. Please try again shortly.",
  BACKEND_UNAVAILABLE:
    "The backend service is unavailable. Check that it is running and try again.",
  BACKEND_CONFIGURATION_ERROR:
    "The backend service is not configured correctly.",
  SERVICE_UNAVAILABLE:
    "The service is temporarily unavailable. Please try again shortly.",
  GATEWAY_TIMEOUT: "The request took too long. Please try again.",
  BACKEND_TIMEOUT: "The backend request timed out. Please try again.",
  AUTH_CONFIGURATION_ERROR:
    "Sign-in is temporarily unavailable. Please try again later.",
  AUTH_PROVIDER_UNAVAILABLE:
    "Sign-in is temporarily unavailable. Please try again later.",
  MODEL_UNAVAILABLE:
    "Answer generation is temporarily unavailable. Please try again.",
  QUERY_NOT_ALLOWED: "This query is not allowed.",
  INVALID_REQUEST_ORIGIN: "This request could not be verified. Refresh and try again.",
  INVALID_REQUEST: "Please check your input and try again.",
  REQUEST_FAILED: "The request could not be completed. Please try again.",
};

const MESSAGE_CODES_ALLOWING_DETAILS = new Set([
  "INVALID_REQUEST",
  "PAYLOAD_TOO_LARGE",
  "UNSUPPORTED_MEDIA_TYPE",
  "VALIDATION_ERROR",
]);

const STATUS_MESSAGES: Record<number, string> = {
  400: SAFE_CODE_MESSAGES.INVALID_REQUEST,
  401: SAFE_CODE_MESSAGES.AUTHENTICATION_REQUIRED,
  403: SAFE_CODE_MESSAGES.FORBIDDEN,
  404: SAFE_CODE_MESSAGES.NOT_FOUND,
  405: SAFE_CODE_MESSAGES.METHOD_NOT_ALLOWED,
  413: SAFE_CODE_MESSAGES.PAYLOAD_TOO_LARGE,
  415: SAFE_CODE_MESSAGES.UNSUPPORTED_MEDIA_TYPE,
  422: SAFE_CODE_MESSAGES.VALIDATION_ERROR,
  429: SAFE_CODE_MESSAGES.RATE_LIMITED,
  500: SAFE_CODE_MESSAGES.INTERNAL_SERVER_ERROR,
  502: SAFE_CODE_MESSAGES.BAD_GATEWAY,
  503: SAFE_CODE_MESSAGES.SERVICE_UNAVAILABLE,
  504: SAFE_CODE_MESSAGES.GATEWAY_TIMEOUT,
};

export function getApiErrorMessage(status: number, data?: unknown): string {
  const envelope = data as ErrorEnvelope | null;
  const code =
    typeof envelope?.error?.code === "string" ? envelope.error.code : undefined;
  const message =
    typeof envelope?.error?.message === "string"
      ? envelope.error.message
      : undefined;

  if (
    code &&
    MESSAGE_CODES_ALLOWING_DETAILS.has(code) &&
    message &&
    status < 500
  ) {
    return message;
  }

  if (code && SAFE_CODE_MESSAGES[code]) {
    return SAFE_CODE_MESSAGES[code];
  }

  return (
    STATUS_MESSAGES[status] ??
    "The request could not be completed. Please try again."
  );
}

export function getApiErrorMessageForCode(code: unknown): string {
  return typeof code === "string" && SAFE_CODE_MESSAGES[code]
    ? SAFE_CODE_MESSAGES[code]
    : "The request could not be completed. Please try again.";
}

export function getNetworkErrorMessage(isTimeout: boolean): string {
  return isTimeout
    ? "The request took too long. Please try again."
    : "The backend service is unavailable. Check that it is running and try again.";
}
