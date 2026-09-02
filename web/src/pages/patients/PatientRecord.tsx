import {
  ArrowLeft,
  Brain,
  CalendarDays,
  FileText,
  HeartPulse,
  Image as ImageIcon,
  MoreHorizontal,
  Plus,
  UserRound,
} from "lucide-react";
import { PatientAvatar } from "../../components/clinical/PatientAvatar";

const analyses = [
  {
    model: "Chest X-Ray Classifier",
    prediction: "No acute abnormality",
    confidence: "96.2%",
    date: "29 Aug 2026",
    modality: "Image",
  },
  {
    model: "Cardiovascular Risk",
    prediction: "Low risk",
    confidence: "91.7%",
    date: "27 Aug 2026",
    modality: "Tabular",
  },
  {
    model: "Respiratory Assessment",
    prediction: "Normal",
    confidence: "94.1%",
    date: "22 Aug 2026",
    modality: "Multimodal",
  },
];

export function PatientRecord() {
  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Back */}
      <button className="mb-5 flex items-center gap-2 text-[11px] font-medium text-gray-500 hover:text-gray-900">
        <ArrowLeft size={14} />
        Back to patients
      </button>

      {/* Patient heading */}
      <section className="rounded-xl border border-gray-200 bg-white p-5 sm:p-6">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <PatientAvatar name="Ama Mensah" size="lg" />

            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-[20px] font-semibold tracking-tight text-gray-950">
                  Ama Mensah
                </h1>

                <span className="rounded-md bg-green-50 px-2 py-1 text-[9px] font-semibold text-green-700">
                  Active
                </span>
              </div>

              <p className="mt-1 text-[11px] text-gray-400">
                PT-10482 · 34 years · Female
              </p>
            </div>
          </div>

          <div className="flex gap-2">
            <button className="flex h-9 items-center gap-2 rounded-lg border border-gray-200 px-3 text-[11px] font-semibold text-gray-600 hover:bg-gray-50">
              <MoreHorizontal size={15} />
              Actions
            </button>

            <button className="flex h-9 items-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[11px] font-semibold text-white hover:bg-gray-800">
              <Brain size={14} />
              New inference
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="-mb-5 mt-6 flex gap-6 overflow-x-auto border-t border-gray-100 pt-4 sm:-mb-6 sm:mt-7 sm:pt-4">
          {["Overview", "Clinical history", "AI analyses", "Documents"].map(
            (tab, index) => (
              <button
                key={tab}
                className={[
                  "shrink-0 pb-1 text-[11px] font-semibold",
                  index === 0
                    ? "border-b-2 border-blue-600 text-blue-600"
                    : "text-gray-400 hover:text-gray-700",
                ].join(" ")}
              >
                {tab}
              </button>
            ),
          )}
        </div>
      </section>

      {/* Content */}
      <div className="mt-5 grid gap-5 lg:grid-cols-[1.35fr_0.65fr]">
        <div className="space-y-5">
          {/* Clinical summary */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 px-5 py-4">
              <div>
                <h2 className="text-[13px] font-semibold text-gray-950">
                  Clinical summary
                </h2>

                <p className="mt-0.5 text-[10px] text-gray-400">
                  Current patient information
                </p>
              </div>

              <button className="text-[10px] font-semibold text-blue-600">
                Edit
              </button>
            </div>

            <div className="grid gap-px bg-gray-100 sm:grid-cols-2">
              {[
                ["Date of birth", "14 Feb 1992"],
                ["Blood group", "O+"],
                ["Primary condition", "Hypertension"],
                ["Allergies", "None recorded"],
                ["Last consultation", "29 Aug 2026"],
                ["Assigned clinician", "Dr. Mensah"],
              ].map(([label, value]) => (
                <div key={label} className="bg-white px-5 py-4">
                  <div className="text-[10px] uppercase tracking-wide text-gray-400">
                    {label}
                  </div>

                  <div className="mt-1 text-[12px] font-medium text-gray-800">
                    {value}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* AI analyses */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 px-5 py-4">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
                  <Brain size={15} />
                </div>

                <div>
                  <h2 className="text-[13px] font-semibold text-gray-950">
                    AI analyses
                  </h2>

                  <p className="mt-0.5 text-[10px] text-gray-400">
                    Previous inference results
                  </p>
                </div>
              </div>

              <button className="flex items-center gap-1 text-[10px] font-semibold text-blue-600">
                View all
              </button>
            </div>

            <div className="divide-y divide-gray-100">
              {analyses.map((analysis) => (
                <div
                  key={analysis.model}
                  className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div className="min-w-0">
                    <div className="text-[12px] font-semibold text-gray-800">
                      {analysis.prediction}
                    </div>

                    <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-gray-400">
                      <span>{analysis.model}</span>
                      <span>·</span>
                      <span>{analysis.modality}</span>
                      <span>·</span>
                      <span>{analysis.date}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className="text-[12px] font-semibold text-gray-900">
                        {analysis.confidence}
                      </div>

                      <div className="text-[9px] text-gray-400">
                        confidence
                      </div>
                    </div>

                    <button className="text-[10px] font-semibold text-blue-600">
                      Open
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Right column */}
        <div className="space-y-5">
          {/* Vitals */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 px-5 py-4">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-50 text-red-500">
                  <HeartPulse size={15} />
                </div>

                <div>
                  <h2 className="text-[13px] font-semibold text-gray-950">
                    Latest vitals
                  </h2>

                  <p className="mt-0.5 text-[10px] text-gray-400">
                    Recorded 29 Aug 2026
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-px bg-gray-100">
              {[
                ["Blood pressure", "128/82", "mmHg"],
                ["Heart rate", "74", "bpm"],
                ["Temperature", "36.7", "°C"],
                ["SpO₂", "98", "%"],
              ].map(([label, value, unit]) => (
                <div key={label} className="bg-white p-4">
                  <div className="text-[10px] text-gray-400">{label}</div>

                  <div className="mt-1">
                    <span className="text-[17px] font-semibold text-gray-900">
                      {value}
                    </span>

                    <span className="ml-1 text-[9px] text-gray-400">
                      {unit}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Quick actions */}
          <section className="rounded-xl border border-gray-200 bg-white p-5">
            <h2 className="text-[13px] font-semibold text-gray-950">
              Quick actions
            </h2>

            <div className="mt-4 space-y-2">
              {[
                [Brain, "Run AI inference"],
                [FileText, "Add clinical note"],
                [ImageIcon, "Upload medical image"],
                [CalendarDays, "Schedule consultation"],
              ].map(([Icon, label]) => (
                <button
                  key={label as string}
                  className="flex w-full items-center gap-3 rounded-lg border border-gray-100 px-3 py-2.5 text-left text-[11px] font-medium text-gray-600 transition hover:border-gray-200 hover:bg-gray-50"
                >
                  <Icon size={15} className="text-gray-400" />
                  {label as string}
                </button>
              ))}
            </div>
          </section>

          {/* Notes */}
          <section className="rounded-xl border border-gray-200 bg-white p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-[13px] font-semibold text-gray-950">
                Clinical notes
              </h2>

              <button className="flex h-7 w-7 items-center justify-center rounded-md text-gray-400 hover:bg-gray-50">
                <Plus size={14} />
              </button>
            </div>

            <p className="mt-4 text-[11px] leading-5 text-gray-500">
              Patient reports improved blood pressure control since the last
              consultation. Continue current management and monitor readings.
            </p>

            <div className="mt-4 flex items-center gap-2 text-[9px] text-gray-400">
              <UserRound size={12} />
              Dr. Mensah · 29 Aug 2026
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}