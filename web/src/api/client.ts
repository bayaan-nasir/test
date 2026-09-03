import axios, { type AxiosError, type InternalAxiosRequestConfig } from "axios";

const ACCESS_TOKEN_KEY = "access_token";
const REFRESH_TOKEN_KEY = "refresh_token";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

const refreshClient = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

let refreshPromise: Promise<string> | null = null;

function getAccessToken() {
  return sessionStorage.getItem(ACCESS_TOKEN_KEY);
}

function getRefreshToken() {
  return sessionStorage.getItem(REFRESH_TOKEN_KEY);
}

function saveAccessToken(accessToken: string) {
  sessionStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
}

function clearTokens() {
  sessionStorage.removeItem(ACCESS_TOKEN_KEY);

  sessionStorage.removeItem(REFRESH_TOKEN_KEY);
}

async function refreshAccessToken() {
  const refreshToken = getRefreshToken();

  if (!refreshToken) {
    throw new Error("No refresh token available.");
  }

  const response = await refreshClient.post<{
    access: string;
    refresh?: string;
  }>("/auth/token/refresh/", {
    refresh: refreshToken,
  });

  const newAccessToken = response.data.access;

  saveAccessToken(newAccessToken);

  if (response.data.refresh) {
    sessionStorage.setItem(REFRESH_TOKEN_KEY, response.data.refresh);
  }

  return newAccessToken;
}

/*
 * Attach access token to every request.
 */
api.interceptors.request.use(
  (config) => {
    const accessToken = getAccessToken();

    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    }

    return config;
  },
  (error) => Promise.reject(error),
);

/*
 * Refresh expired access tokens
 * and retry the original request.
 */
api.interceptors.response.use(
  (response) => response,

  async (error: AxiosError) => {
    const originalRequest = error.config as
      | (InternalAxiosRequestConfig & {
          _retry?: boolean;
        })
      | undefined;

    if (
      error.response?.status !== 401 ||
      !originalRequest ||
      originalRequest._retry
    ) {
      return Promise.reject(error);
    }

    /*
     * Don't attempt to refresh when
     * the request itself is authentication.
     */
    if (
      originalRequest.url?.includes("/auth/login/") ||
      originalRequest.url?.includes("/auth/register/") ||
      originalRequest.url?.includes("/auth/refresh/")
    ) {
      return Promise.reject(error);
    }

    originalRequest._retry = true;

    try {
      /*
       * If another request is already
       * refreshing, wait for it.
       */
      if (!refreshPromise) {
        refreshPromise = refreshAccessToken().finally(() => {
          refreshPromise = null;
        });
      }

      const newAccessToken = await refreshPromise;

      originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;

      return api(originalRequest);
    } catch (refreshError) {
      clearTokens();

      /*
       * Let AuthContext / application
       * navigation handle the final logout.
       */
      window.dispatchEvent(new Event("auth:expired"));

      return Promise.reject(refreshError);
    }
  },
);

export default api;
