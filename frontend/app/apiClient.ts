import axios from "axios";

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
      if (error.code === "ECONNABORTED") {
        error.message = "Request timed out. The RAG pipeline took longer than expected.";
      } else {
        error.message = "Network error. Check your connection or verify backend server is running.";
      }
    }
    return Promise.reject(error);
  }
);

export default api;