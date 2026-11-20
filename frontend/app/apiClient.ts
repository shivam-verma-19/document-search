import axios from "axios";
import {
  getApiErrorMessage,
  getNetworkErrorMessage,
} from "../lib/api-errors";

const api = axios.create({
  baseURL: "/api/backend",
  timeout: 45000,
  withCredentials: true,
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && typeof window !== "undefined") {
      window.dispatchEvent(new Event("rag:auth-required"));
    }
    if (!error.response) {
      error.message = getNetworkErrorMessage(error.code === "ECONNABORTED");
    } else {
      error.message = getApiErrorMessage(
        error.response.status,
        error.response.data,
      );
    }
    return Promise.reject(error);
  }
);

export default api;