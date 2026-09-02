import {
  Activity,
  Brain,
  ScanLine,
  ShieldCheck,
} from "lucide-react";
import { AuthLogo } from "./AuthLogo";

export function AuthShell({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-white">
      <div className="grid min-h-screen lg:grid-cols-[minmax(420px,0.9fr)_minmax(500px,1.1fr)]">
        {/* Desktop branding panel */}
        <section className="relative hidden overflow-hidden bg-gray-950 p-10 text-white lg:flex lg:flex-col">
          <AuthLogo />

          <div className="relative z-10 flex flex-1 items-center">
            <div className="max-w-lg">
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-gray-300">
                <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
                Clinical AI Platform
              </div>

              <h1 className="max-w-md text-4xl font-semibold leading-[1.1] tracking-tight">
                Intelligence for the clinical workflow.
              </h1>

              <p className="mt-5 max-w-md text-sm leading-6 text-gray-400">
                Analyze structured clinical data, medical images, and
                multimodal patient information through one inference
                workspace.
              </p>

              <div className="mt-10 grid max-w-md grid-cols-3 gap-3">
                {[
                  [Brain, "Multimodal", "Inference"],
                  [ScanLine, "Image", "Analysis"],
                  [Activity, "Clinical", "Signals"],
                ].map(([Icon, title, subtitle]) => (
                  <div
                    key={title as string}
                    className="rounded-xl border border-white/10 bg-white/4 p-4"
                  >
                    <Icon
                      size={18}
                      strokeWidth={1.7}
                      className="text-blue-400"
                    />

                    <div className="mt-5 text-[11px] font-semibold text-gray-200">
                      {title as string}
                    </div>

                    <div className="mt-0.5 text-[10px] text-gray-500">
                      {subtitle as string}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="relative z-10 flex items-center gap-2 text-[10px] text-gray-500">
            <ShieldCheck size={14} />
            Secure clinical intelligence workspace
          </div>

          {/* Decorative inference rings */}
          <div className="pointer-events-none absolute -bottom-48 -right-32 h-130 w-130 rounded-full border border-blue-500/10" />

          <div className="pointer-events-none absolute -bottom-32 -right-16 h-90 w-90 rounded-full border border-blue-500/[0.07]" />

          <div className="pointer-events-none absolute right-16 top-1/3 h-2 w-2 rounded-full bg-blue-400/60" />
        </section>

        {/* Form panel */}
        <section className="flex min-h-screen flex-col">
          {/* Mobile logo */}
          <div className="flex items-center justify-between px-5 py-6 sm:px-8 lg:hidden">
            <AuthLogo />
          </div>

          <div className="flex flex-1 items-center justify-center px-5 py-8 sm:px-8">
            <div className="w-full max-w-105">
              {children}
            </div>
          </div>

          <div className="px-5 pb-6 text-center text-[10px] text-gray-400 sm:px-8">
            © 2026
          </div>
        </section>
      </div>
    </div>
  );
}