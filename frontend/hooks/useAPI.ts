import { useState, useCallback, useRef } from "react";
import api from "../app/apiClient";
import axios from "axios";

export type ApiErrorType =
  | "timeout"
  | "network"
  | "auth"
  | "rate_limit"
  | "server"
  | "validation"
  | "unknown";

export interface ApiError {
  type: ApiErrorType;
  message: string;
  retryable: boolean;
  statusCode?: number;
  details?: Record<string, any>;
}

export interface UseApiOptions {
  maxRetries?: number;
  timeoutMs?: number;
  onSuccess?: (data: any) => void;
  onError?: (error: ApiError) => void;
  onRetry?: (attempt: number) => void;
}

export interface UseApiState {
  loading: boolean;
  error: ApiError | null;
  data: any | null;
  retryCount: number;
  lastAttemptTime: number | null;
}

const categorizeError = (error: any): ApiError => {
  // Network/Timeout errors
  if (error.code === "ECONNABORTED" || error.message?.includes("timeout")) {
    return {
      type: "timeout",
      message: "Request took too long. Please try again.",
      retryable: true,
      details: { originalError: error.message },
    };
  }

  if (!error.response) {
    return {
      type: "network",
      message: "Network error. Check your connection and try again.",
      retryable: true,
      details: { originalError: error.message },
    };
  }

  const status = error.response?.status;
  const errorData = error.response?.data;

  // Authentication errors
  if (status === 401 || status === 403) {
    return {
      type: "auth",
      message: "Your session has expired. Please log in again.",
      retryable: false,
      statusCode: status,
    };
  }

  // Rate limiting
  if (status === 429) {
    return {
      type: "rate_limit",
      message: "Too many requests. Please wait a moment and try again.",
      retryable: true,
      statusCode: status,
      details: errorData?.error?.details || {},
    };
  }

  // Validation errors
  if (status === 400) {
    return {
      type: "validation",
      message: errorData?.error?.message || "Invalid input. Please check your request.",
      retryable: false,
      statusCode: status,
    };
  }

  // Server errors (retryable)
  if (status && status >= 500) {
    return {
      type: "server",
      message: "Server error. Please try again in a moment.",
      retryable: true,
      statusCode: status,
      details: {
        category: errorData?.error?.category,
        severity: errorData?.error?.severity,
      },
    };
  }

  return {
    type: "unknown",
    message: errorData?.error?.message || error.message || "An unexpected error occurred.",
    retryable: errorData?.error?.retryable || false,
    statusCode: status,
  };
};

export const useApi = (options: UseApiOptions = {}) => {
  const {
    maxRetries = 3,
    timeoutMs = 5000,
    onSuccess,
    onError,
    onRetry,
  } = options;

  const [state, setState] = useState<UseApiState>({
    loading: false,
    error: null,
    data: null,
    retryCount: 0,
    lastAttemptTime: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const execute = useCallback(
    async (
      method: string,
      url: string,
      config: any = {}
    ) => {
      let attempt = 0;

      const executeAttempt = async (): Promise<any> => {
        attempt++;

        // Cancel previous request if still pending
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
        }

        abortControllerRef.current = new AbortController();

        setState((prev) => ({
          ...prev,
          loading: true,
          error: null,
          retryCount: attempt - 1,
          lastAttemptTime: Date.now(),
        }));

        try {
          // Set timeout
          const timeoutId = setTimeout(() => {
            abortControllerRef.current?.abort();
          }, timeoutMs);

          const response = await api({
            method,
            url,
            timeout: timeoutMs,
            signal: abortControllerRef.current.signal,
            ...config,
          });

          clearTimeout(timeoutId);

          setState((prev) => ({
            ...prev,
            loading: false,
            data: response.data,
            error: null,
          }));

          onSuccess?.(response.data);
          return response.data;
        } catch (error: any) {
          if (error.name === "AbortError") {
            error.code = "ECONNABORTED";
          }

          const apiError = categorizeError(error);

          // Determine if we should retry
          const shouldRetry = apiError.retryable && attempt < maxRetries;

          if (shouldRetry) {
            onRetry?.(attempt);

            // Exponential backoff
            const delayMs = Math.min(100 * Math.pow(2, attempt - 1), 5000);
            await new Promise((resolve) => setTimeout(resolve, delayMs));

            return executeAttempt();
          }

          setState((prev) => ({
            ...prev,
            loading: false,
            error: apiError,
          }));

          onError?.(apiError);
          throw apiError;
        }
      };

      return executeAttempt();
    },
    [maxRetries, timeoutMs, onSuccess, onError, onRetry]
  );

  const get = useCallback(
    (url: string, config?: any) => execute("GET", url, config),
    [execute]
  );

  const post = useCallback(
    (url: string, data?: any, config?: any) =>
      execute("POST", url, { ...config, data }),
    [execute]
  );

  const put = useCallback(
    (url: string, data?: any, config?: any) =>
      execute("PUT", url, { ...config, data }),
    [execute]
  );

  const cancel = useCallback(() => {
    abortControllerRef.current?.abort();
    setState((prev) => ({ ...prev, loading: false }));
  }, []);

  const reset = useCallback(() => {
    setState({
      loading: false,
      error: null,
      data: null,
      retryCount: 0,
      lastAttemptTime: null,
    });
  }, []);

  return {
    ...state,
    get,
    post,
    put,
    cancel,
    reset,
  };
};