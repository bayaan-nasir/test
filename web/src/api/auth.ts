import api from "./client";

import type {
  AuthResponse,
  LoginRequest,
  RegisterRequest,
  User,
} from "../types/auth";

export async function register(data: RegisterRequest) {
  const response = await api.post<User>("/auth/register/", data);

  return response.data;
}

export async function login(data: LoginRequest) {
  const response = await api.post<AuthResponse>("/auth/login/", data);

  return response.data;
}

export async function refreshToken(refresh: string) {
  const response = await api.post<{
    access: string;
    refresh?: string;
  }>("/auth/refresh/", {
    refresh,
  });

  return response.data;
}

export async function logout(refresh: string) {
  return api.post("/auth/logout/", {
    refresh,
  });
}

export async function getCurrentUser() {
  const response = await api.get<User>("/auth/me/");

  return response.data;
}
