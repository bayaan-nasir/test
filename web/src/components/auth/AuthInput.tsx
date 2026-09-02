import type { InputHTMLAttributes } from "react";

interface AuthInputProps
  extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
}

export function AuthInput({
  label,
  error,
  className = "",
  ...props
}: AuthInputProps) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[12px] font-medium text-gray-700">
        {label}
      </span>

      <input
        {...props}
        className={`h-11 w-full rounded-lg border bg-white px-3.5 text-[13px] text-gray-900 outline-none transition placeholder:text-gray-400 focus:ring-4 ${error
            ? "border-red-300 focus:border-red-400 focus:ring-red-50"
            : "border-gray-200 focus:border-blue-400 focus:ring-blue-50"
          } ${className}`}
      />

      {error && (
        <span className="mt-1 block text-[10px] text-red-500">
          {error}
        </span>
      )}
    </label>
  );
}