import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  timeout: 5000,
});

api.interceptors.request.use((config) => {
  if (typeof window !== "undefined" && config.headers) {
    const token = window.localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (!error.response) {
      if (error.code === "ECONNABORTED") {
        error.message = "Request timed out. Please try again.";
      } else {
        error.message = "Network error. Check your connection.";
      }
    }
    return Promise.reject(error);
  }
);

export default api;