import {
  Activity,
  Brain,
  ClipboardList,
  Users,
} from "lucide-react";
import { StatCard } from "../../components/ui/StatCard";

const cases = [
  ["Ama K.", "Chest symptoms", "Reviewing", "Available"],
  ["Kwame A.", "Skin lesion", "Complete", "Completed"],
  ["Akosua M.", "Diabetes screening", "Pending", "Not started"],
  ["Kofi B.", "Respiratory symptoms", "Reviewing", "Running"],
];

const models = [
  ["Chest X-Ray Model", "42", 91.4],
  ["Skin Lesion Model", "31", 87.8],
  ["Diabetes Risk Model", "54", 93.1],
];

function Status({ children }: { children: string }) {
  const styles: Record<string, string> = {
    Reviewing: "bg-amber-50 text-amber-700",
    Complete: "bg-green-50 text-green-700",
    Pending: "bg-gray-100 text-gray-600",
    Available: "bg-blue-50 text-blue-700",
    Completed: "bg-green-50 text-green-700",
    "Not started": "bg-gray-100 text-gray-500",
    Running: "bg-blue-50 text-blue-700",
  };

  return (
    <span
      className={`inline-flex w-fit rounded-md px-2 py-1 text-[10px] font-semibold ${
        styles[children] ?? "bg-gray-100 text-gray-600"
      }`}
    >
      {children}
    </span>
  );
}

export function Dashboard() {
  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Page heading */}
      <div className="mb-6 sm:mb-8">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-blue-600">
          Clinical workspace
        </p>

        <h1 className="mt-1.5 text-[23px] font-semibold tracking-tight text-gray-950 sm:text-[28px]">
          Good afternoon, Dr. Mensah
        </h1>

        <p className="mt-1 max-w-2xl text-[12px] leading-5 text-gray-500 sm:text-[13px]">
          An overview of your clinical activity and AI-assisted analyses.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 sm:gap-4 xl:grid-cols-4">
        <StatCard
          label="Active Patients"
          value="248"
          change="+12%"
          trend="up"
          icon={Users}
        />

        <StatCard
          label="Open Cases"
          value="31"
          change="-4%"
          trend="down"
          icon={ClipboardList}
        />

        <StatCard
          label="AI Analyses"
          value="127"
          change="+18%"
          trend="up"
          icon={Brain}
        />

        <StatCard
          label="Analyses Today"
          value="24"
          change="+8%"
          trend="up"
          icon={Activity}
        />
      </div>

      {/* Main dashboard panels */}
      <div className="mt-5 grid gap-5 sm:mt-6 sm:gap-6 xl:grid-cols-[1.55fr_1fr]">
        {/* Recent cases */}
        <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-4 sm:px-5">
            <div>
              <h2 className="text-[14px] font-semibold text-gray-950">
                Recent Cases
              </h2>

              <p className="mt-0.5 text-[11px] text-gray-400">
                Latest clinical activity
              </p>
            </div>

            <button className="text-[11px] font-semibold text-blue-600 hover:text-blue-700">
              View all
            </button>
          </div>

          {/* Desktop/tablet */}
          <div className="hidden divide-y divide-gray-100 sm:block">
            {cases.map(([patient, caseName, status, ai]) => (
              <div
                key={patient}
                className="grid grid-cols-[1fr_1.5fr_110px_110px] items-center gap-4 px-5 py-4"
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[10px] font-semibold text-gray-600">
                    {patient
                      .split(" ")
                      .map((n) => n[0])
                      .join("")}
                  </div>

                  <span className="text-[12px] font-medium text-gray-800">
                    {patient}
                  </span>
                </div>

                <span className="text-[12px] text-gray-600">
                  {caseName}
                </span>

                <Status>{status}</Status>
                <Status>{ai}</Status>
              </div>
            ))}
          </div>

          {/* Mobile cards */}
          <div className="divide-y divide-gray-100 sm:hidden">
            {cases.map(([patient, caseName, status, ai]) => (
              <div key={patient} className="px-4 py-4">
                <div className="flex items-start gap-3">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[10px] font-semibold text-gray-600">
                    {patient
                      .split(" ")
                      .map((n) => n[0])
                      .join("")}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-[12px] font-semibold text-gray-900">
                          {patient}
                        </p>

                        <p className="mt-0.5 text-[11px] text-gray-500">
                          {caseName}
                        </p>
                      </div>

                      <Status>{status}</Status>
                    </div>

                    <div className="mt-3 flex items-center justify-between">
                      <span className="text-[10px] uppercase tracking-wide text-gray-400">
                        AI analysis
                      </span>

                      <Status>{ai}</Status>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* AI activity */}
        <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
          <div className="border-b border-gray-100 px-4 py-4 sm:px-5">
            <h2 className="text-[14px] font-semibold text-gray-950">
              AI Model Activity
            </h2>

            <p className="mt-0.5 text-[11px] text-gray-400">
              Current inference performance
            </p>
          </div>

          <div className="divide-y divide-gray-100">
            {models.map(([model, analyses, confidence]) => (
              <div key={model} className="px-4 py-4 sm:px-5">
                <div className="flex items-start justify-between gap-4">
                  <span className="text-[12px] font-medium text-gray-800">
                    {model}
                  </span>

                  <span className="shrink-0 text-[12px] font-semibold text-gray-950">
                    {confidence}%
                  </span>
                </div>

                <div className="mt-2 flex items-center justify-between text-[10px] text-gray-400">
                  <span>{analyses} analyses</span>
                  <span>avg. confidence</span>
                </div>

                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-gray-100">
                  <div
                    className="h-full rounded-full bg-blue-600"
                    style={{ width: `${confidence}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* AI callout */}
      <section className="relative mt-5 overflow-hidden rounded-2xl bg-gray-950 p-5 text-white sm:mt-6 sm:p-6">
        <div className="pointer-events-none absolute -right-16 -top-20 h-52 w-52 rounded-full border border-blue-500/15" />
        <div className="pointer-events-none absolute -right-8 -top-12 h-36 w-36 rounded-full border border-blue-500/10" />

        <div className="relative flex flex-col justify-between gap-5 md:flex-row md:items-center">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-white/10">
                <Brain size={14} />
              </div>

              <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-gray-400">
                AI Inference Engine
              </span>
            </div>

            <h2 className="text-[17px] font-semibold tracking-tight sm:text-[19px]">
              Run a new clinical analysis
            </h2>

            <p className="mt-1 max-w-xl text-[11px] leading-5 text-gray-400 sm:text-[12px]">
              Analyze tabular clinical data, medical images, or combine both
              through the multimodal inference pipeline.
            </p>
          </div>

          <button className="w-full shrink-0 rounded-lg bg-white px-4 py-2.5 text-[12px] font-semibold text-gray-950 transition hover:bg-gray-100 sm:w-auto">
            New inference →
          </button>
        </div>
      </section>
    </div>
  );
}