import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowRight } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import {
  useLocation,
  useNavigate,
} from "react-router-dom";

import { useAuth } from "../../auth/AuthContext";
import { AuthInput } from "../../components/auth/AuthInput";
import {
  loginSchema,
  type LoginFormValues,
} from "../../validation/auth";

export function Login() {
  const { login } = useAuth();

  const navigate = useNavigate();
  const location = useLocation();

  const [serverError, setServerError] =
    useState("");

  const {
    register,
    handleSubmit,
    setError,
    formState: {
      errors,
      isSubmitting,
    },
  } = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
    mode: "onBlur",
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const onSubmit = async (
    values: LoginFormValues,
  ) => {
    setServerError("");

    try {
      await login(values);

      const from =
        (
          location.state as {
            from?: {
              pathname?: string;
            };
          }
        )?.from?.pathname ?? "/app";

      navigate(from, {
        replace: true,
      });
    } catch (error: any) {
      const data = error?.response?.data;

      if (
        data?.non_field_errors &&
        Array.isArray(data.non_field_errors)
      ) {
        setServerError(
          data.non_field_errors.join(" "),
        );
        return;
      }

      if (
        typeof data?.detail === "string"
      ) {
        setServerError(data.detail);
        return;
      }

      if (
        typeof data === "object" &&
        data !== null
      ) {
        let hasFieldError = false;

        if (data.email) {
          setError("email", {
            type: "server",
            message: Array.isArray(data.email)
              ? data.email.join(" ")
              : String(data.email),
          });

          hasFieldError = true;
        }

        if (data.password) {
          setError("password", {
            type: "server",
            message: Array.isArray(
              data.password,
            )
              ? data.password.join(" ")
              : String(data.password),
          });

          hasFieldError = true;
        }

        if (!hasFieldError) {
          const message = Object.values(data)
            .flat()
            .join(" ");

          setServerError(
            message ||
            "Unable to sign in. Please try again.",
          );
        }

        return;
      }

      setServerError(
        "Unable to sign in. Please check your credentials.",
      );
    }
  };

  return (
    <div>
      <div className="mb-8">
        <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-blue-600">
          Welcome back
        </p>

        <h2 className="mt-2 text-[28px] font-semibold tracking-tight text-gray-950">
          Sign in to MEDAI
        </h2>

        <p className="mt-2 text-[13px] leading-5 text-gray-500">
          Access your clinical workspace and
          AI-assisted analyses.
        </p>
      </div>

      {serverError && (
        <div
          role="alert"
          className="mb-5 rounded-lg border border-red-100 bg-red-50 px-3.5 py-3 text-[11px] leading-4 text-red-600"
        >
          {serverError}
        </div>
      )}

      <form
        className="space-y-5"
        onSubmit={handleSubmit(onSubmit)}
        noValidate
      >
        <AuthInput
          label="Email address"
          type="email"
          placeholder="you@hospital.com"
          autoComplete="email"
          error={errors.email?.message}
          {...register("email")}
        />

        <div>
          <AuthInput
            label="Password"
            type="password"
            placeholder="Enter your password"
            autoComplete="current-password"
            error={errors.password?.message}
            {...register("password")}
          />

          <div className="mt-2 flex justify-end">
            <button
              type="button"
              onClick={() =>
                navigate("/forgot-password")
              }
              className="text-[11px] font-semibold text-blue-600 hover:text-blue-700"
            >
              Forgot password?
            </button>
          </div>
        </div>

        <button
          type="submit"
          disabled={isSubmitting}
          className="flex h-11 w-full items-center justify-center gap-2 rounded-lg bg-gray-950 text-[12px] font-semibold text-white transition hover:bg-gray-800 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isSubmitting
            ? "Signing in..."
            : "Sign in"}

          {!isSubmitting && (
            <ArrowRight size={15} />
          )}
        </button>
      </form>

      <div className="my-7 flex items-center gap-3">
        <div className="h-px flex-1 bg-gray-200" />

        <span className="text-[10px] font-medium uppercase tracking-wider text-gray-400">
          or
        </span>

        <div className="h-px flex-1 bg-gray-200" />
      </div>

      <p className="text-center text-[12px] text-gray-500">
        Don't have an account?{" "}
        <button
          type="button"
          onClick={() =>
            navigate("/register")
          }
          className="font-semibold text-blue-600 hover:text-blue-700"
        >
          Create an account
        </button>
      </p>
    </div>
  );
}