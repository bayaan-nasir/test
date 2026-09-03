import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowRight } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "../../auth/AuthContext";
import { AuthInput } from "../../components/auth/AuthInput";
import { registerSchema, type RegisterFormValues } from "../../validation/auth";

export function Register() {
  const { register } = useAuth();

  const navigate = useNavigate();
  const location = useLocation();

  const [serverError, setServerError] = useState("");

  const {
    register: registerField,
    handleSubmit,
    setError,
    formState: { errors, isSubmitting },
  } = useForm<RegisterFormValues>({
    resolver: zodResolver(registerSchema),
    mode: "onBlur",
    defaultValues: {
      first_name: "",
      last_name: "",
      email: "",
      phone_number: "",
      professional_role: undefined,
      password: "",
      confirm_password: "",
      accepted_terms: false,
    },
  });

  const onSubmit = async (values: RegisterFormValues) => {
    setServerError("");

    try {
      await register({
        email: values.email,
        first_name: values.first_name,
        last_name: values.last_name,
        phone_number: values.phone_number,
        role: "CLINICIAN",
        professional_role: values.professional_role,
        password: values.password,
        confirm_password: values.confirm_password,
      });

      const from =
        (
          location.state as {
            from?: {
              pathname?: string;
            };
          }
        )?.from?.pathname ?? "/app";

      navigate(from, { replace: true });
    } catch (error: any) {
      const data = error?.response?.data;

      if (typeof data === "object" && data !== null) {
        let handledFieldError = false;

        for (const [field, messages] of Object.entries(data)) {
          if (
            field in
            {
              first_name: true,
              last_name: true,
              email: true,
              phone_number: true,
              professional_role: true,
              password: true,
            }
          ) {
            const message = Array.isArray(messages)
              ? messages.join(" ")
              : String(messages);

            setError(field as keyof RegisterFormValues, {
              type: "server",
              message,
            });

            handledFieldError = true;
          }
        }

        if (!handledFieldError) {
          const message = Object.values(data).flat().join(" ");

          setServerError(message || "Unable to create your account.");
        }
      } else {
        setServerError("Unable to create your account. Please try again.");
      }
    }
  };

  return (
    <div>
      <div className="mb-7">
        <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-blue-600">
          Get started
        </p>

        <h2 className="mt-2 text-[28px] font-semibold tracking-tight text-gray-950">
          Create your account
        </h2>

        <p className="mt-2 text-[13px] leading-5 text-gray-500">
          Set up your clinical profile to access AI Diagnosis Assistant.
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

      <form className="space-y-4" onSubmit={handleSubmit(onSubmit)} noValidate>
        <div className="grid gap-4 sm:grid-cols-2">
          <AuthInput
            label="First name"
            placeholder="First name"
            autoComplete="given-name"
            error={errors.first_name?.message}
            {...registerField("first_name")}
          />

          <AuthInput
            label="Last name"
            placeholder="Last name"
            autoComplete="family-name"
            error={errors.last_name?.message}
            {...registerField("last_name")}
          />
        </div>

        <AuthInput
          label="Email address"
          type="email"
          placeholder="you@hospital.com"
          autoComplete="email"
          error={errors.email?.message}
          {...registerField("email")}
        />

        <AuthInput
          label="Phone number"
          type="tel"
          placeholder="+233 20 000 0000"
          autoComplete="tel"
          error={errors.phone_number?.message}
          {...registerField("phone_number")}
        />

        <label className="block">
          <span className="mb-1.5 block text-[12px] font-medium text-gray-700">
            Professional role
          </span>

          <select
            className={`h-11 w-full rounded-lg border bg-white px-3.5 text-[13px] outline-none transition focus:ring-4 ${
              errors.professional_role
                ? "border-red-300 focus:border-red-400 focus:ring-red-50"
                : "border-gray-200 text-gray-700 focus:border-blue-400 focus:ring-blue-50"
            }`}
            {...registerField("professional_role")}
          >
            <option value="">Select your role</option>

            <option value="DOCTOR">Doctor</option>

            <option value="NURSE">Nurse</option>

            <option value="CLINICAL_RESEARCHER">Clinical Researcher</option>

            <option value="OTHER">Other</option>
          </select>

          {errors.professional_role && (
            <span className="mt-1 block text-[10px] text-red-500">
              {errors.professional_role.message}
            </span>
          )}
        </label>

        <AuthInput
          label="Password"
          type="password"
          placeholder="Create a password"
          autoComplete="new-password"
          error={errors.password?.message}
          {...registerField("password")}
        />

        <AuthInput
          label="Confirm password"
          type="password"
          placeholder="Confirm your password"
          autoComplete="new-password"
          error={errors.confirm_password?.message}
          {...registerField("confirm_password")}
        />

        <label className="flex items-start gap-2.5 pt-1">
          <input
            type="checkbox"
            className={`mt-0.5 h-3.5 w-3.5 rounded border-gray-300 accent-blue-600 ${
              errors.accepted_terms ? "border-red-400" : ""
            }`}
            {...registerField("accepted_terms")}
          />

          <span className="text-[10px] leading-4 text-gray-500">
            I agree to the platform's terms of use and understand that AI
            outputs are decision-support information, not a substitute for
            clinical judgment.
          </span>
        </label>

        {errors.accepted_terms && (
          <p className="-mt-2 text-[10px] text-red-500">
            {errors.accepted_terms.message}
          </p>
        )}

        <button
          type="submit"
          disabled={isSubmitting}
          className="flex h-11 w-full items-center justify-center gap-2 rounded-lg bg-gray-950 text-[12px] font-semibold text-white transition hover:bg-gray-800 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isSubmitting ? "Creating account..." : "Create account"}

          {!isSubmitting && <ArrowRight size={15} />}
        </button>
      </form>

      <p className="mt-6 text-center text-[12px] text-gray-500">
        Already have an account?{" "}
        <button
          type="button"
          onClick={() => navigate("/login")}
          className="font-semibold text-blue-600 hover:text-blue-700"
        >
          Sign in
        </button>
      </p>
    </div>
  );
}
