import { useState, useCallback, useRef } from "react";
import api from "../app/apiClient";
import {
  getApiErrorMessage,
  getNetworkErrorMessage,
} from "../lib/api-errors";

export type ApiErrorType =
  | "timeout"
  | "network"
  | "auth"
  | "forbidden"
  | "not_found"
  | "rate_limit"
  | "server"
  | "validation"
  | "unknown";

export interface ApiError {
  type: ApiErrorType;
  message: string;
  retryable: boolean;
  statusCode?: number;
  code?: string;
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
  const status = error.response?.status as number | undefined;
  const code = error.response?.data?.error?.code as string | undefined;

  if (error.code === "ECONNABORTED") {
    return {
      type: "timeout",
      message: getNetworkErrorMessage(true),
      retryable: true,
    };
  }

  if (status === undefined) {
    return {
      type: "network",
      message: getNetworkErrorMessage(false),
      retryable: true,
    };
  }

  if (status === 401) {
    return {
      type: "auth",
      message: getApiErrorMessage(status, error.response.data),
      retryable: false,
      statusCode: status,
      code,
    };
  }
  if (status === 403) {
    return {
      type: "forbidden",
      message: getApiErrorMessage(status, error.response.data),
      retryable: false,
      statusCode: status,
      code,
    };
  }
  if (status === 404) {
    return {
      type: "not_found",
      message: getApiErrorMessage(status, error.response.data),
      retryable: false,
      statusCode: status,
      code,
    };
  }
  if (status === 429) {
    return {
      type: "rate_limit",
      message: getApiErrorMessage(status, error.response.data),
      retryable: true,
      statusCode: status,
      code,
    };
  }
  if (status >= 500) {
    return {
      type: status === 504 ? "timeout" : "server",
      message: getApiErrorMessage(status, error.response.data),
      retryable: true,
      statusCode: status,
      code,
    };
  }

  return {
    type:
      status === 400 || status === 413 || status === 415 || status === 422
        ? "validation"
        : "unknown",
    message: getApiErrorMessage(status, error.response.data),
    retryable: false,
    statusCode: status,
    code,
  };
};

export const useApi = (options: UseApiOptions = {}) => {
  const {
    maxRetries = 3,
    timeoutMs = 45000,
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

  const del = useCallback(
    (url: string, config?: any) => execute("DELETE", url, config),
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
    del,
    cancel,
    reset,
  };
};