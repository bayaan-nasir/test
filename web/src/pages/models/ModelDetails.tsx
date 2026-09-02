import {
  Activity,
  ArrowLeft,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  ChevronDown,
  Clock3,
  Code2,
  Database,
  ExternalLink,
  FileImage,
  FileSpreadsheet,
  GitBranch,
  Layers3,
  MoreHorizontal,
  Play,
  Server,
  Settings2,
  ShieldCheck,
  Table2,
} from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";

type ModelStatus = "Production" | "Staging" | "Inactive";

interface ModelData {
  id: string;
  name: string;
  description: string;
  version: string;
  status: ModelStatus;
  modality: string;
  task: string;
  framework: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  latency: string;
  inferences: number;
  updated: string;
  endpoint: string;
}

const modelData: Record<string, ModelData> = {
  "mdl-respiratory": {
    id: "mdl-respiratory",
    name: "Respiratory Assessment",
    description:
      "Multimodal respiratory condition assessment combining structured clinical observations with chest imaging to produce an assisted diagnostic prediction.",
    version: "v2.4.1",
    status: "Production",
    modality: "Multimodal",
    task: "Classification",
    framework: "PyTorch",
    accuracy: 94.7,
    precision: 93.8,
    recall: 92.9,
    f1: 93.3,
    latency: "2.84s",
    inferences: 428,
    updated: "29 Aug 2026",
    endpoint: "/api/v1/inference/respiratory",
  },
  "mdl-cxr": {
    id: "mdl-cxr",
    name: "Chest X-Ray Classifier",
    description:
      "Image-based classification model for detecting abnormalities in chest radiographs.",
    version: "v3.1.0",
    status: "Production",
    modality: "Image",
    task: "Classification",
    framework: "PyTorch",
    accuracy: 96.2,
    precision: 95.8,
    recall: 94.6,
    f1: 95.2,
    latency: "1.42s",
    inferences: 617,
    updated: "28 Aug 2026",
    endpoint: "/api/v1/inference/chest-xray",
  },
  "mdl-cardio": {
    id: "mdl-cardio",
    name: "Cardiovascular Risk",
    description:
      "Risk stratification model using structured patient clinical and demographic data.",
    version: "v1.8.3",
    status: "Production",
    modality: "Tabular",
    task: "Risk prediction",
    framework: "XGBoost",
    accuracy: 88.4,
    precision: 87.9,
    recall: 86.7,
    f1: 87.3,
    latency: "0.82s",
    inferences: 239,
    updated: "26 Aug 2026",
    endpoint: "/api/v1/inference/cardiovascular",
  },
  "mdl-derma": {
    id: "mdl-derma",
    name: "Dermatology Assistant",
    description:
      "Experimental image classification model for dermatological observations.",
    version: "v0.9.2",
    status: "Staging",
    modality: "Image",
    task: "Classification",
    framework: "PyTorch",
    accuracy: 91.3,
    precision: 90.7,
    recall: 89.8,
    f1: 90.2,
    latency: "1.76s",
    inferences: 87,
    updated: "24 Aug 2026",
    endpoint: "/api/v1/inference/dermatology",
  },
  "mdl-lab": {
    id: "mdl-lab",
    name: "Laboratory Risk Engine",
    description:
      "Experimental model for identifying elevated clinical risk from laboratory measurements.",
    version: "v0.4.0",
    status: "Staging",
    modality: "Tabular",
    task: "Risk prediction",
    framework: "XGBoost",
    accuracy: 86.9,
    precision: 85.6,
    recall: 84.9,
    f1: 85.2,
    latency: "0.63s",
    inferences: 42,
    updated: "20 Aug 2026",
    endpoint: "/api/v1/inference/laboratory",
  },
};

const recentRuns = [
  {
    id: "INF-7F82A1",
    patient: "Ama Mensah",
    result: "Pneumonia",
    confidence: "94.7%",
    status: "Review required",
    time: "29 Aug · 14:32",
  },
  {
    id: "INF-4C83B2",
    patient: "Kofi Boateng",
    result: "Other abnormality",
    confidence: "79.1%",
    status: "Review required",
    time: "28 Aug · 16:07",
  },
  {
    id: "INF-1A39C7",
    patient: "Esi Ofori",
    result: "Pneumonia",
    confidence: "89.6%",
    status: "Completed",
    time: "27 Aug · 09:52",
  },
  {
    id: "INF-0C18D3",
    patient: "Adwoa Yeboah",
    result: "Normal",
    confidence: "97.1%",
    status: "Completed",
    time: "26 Aug · 13:44",
  },
];

const versions = [
  {
    version: "v2.4.1",
    date: "29 Aug 2026",
    status: "Current",
    changes: "Improved multimodal fusion",
  },
  {
    version: "v2.4.0",
    date: "17 Aug 2026",
    status: "Previous",
    changes: "Updated image preprocessing",
  },
  {
    version: "v2.3.2",
    date: "02 Aug 2026",
    status: "Previous",
    changes: "Calibration improvements",
  },
  {
    version: "v2.3.0",
    date: "21 Jul 2026",
    status: "Previous",
    changes: "New training dataset",
  },
];

function StatusBadge({ status }: { status: ModelStatus }) {
  const styles = {
    Production: "bg-green-50 text-green-700",
    Staging: "bg-amber-50 text-amber-700",
    Inactive: "bg-gray-100 text-gray-500",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[8px] font-semibold ${styles[status]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${status === "Production"
          ? "bg-green-500"
          : status === "Staging"
            ? "bg-amber-500"
            : "bg-gray-400"
          }`}
      />
      {status}
    </span>
  );
}

function Metric({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
        {label}
      </div>

      <div className="mt-2 text-[20px] font-semibold tracking-tight text-gray-950">
        {value}
      </div>

      <div className="mt-1 text-[8px] text-gray-400">
        {description}
      </div>
    </div>
  );
}

export function ModelDetails() {
  const navigate = useNavigate();
  const { modelId } = useParams();

  const model =
    modelData[modelId ?? "mdl-respiratory"] ??
    modelData["mdl-respiratory"];

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Back */}
      <button
        type="button"
        onClick={() => navigate("/app/models")}
        className="mb-5 flex items-center gap-2 text-[9px] font-medium text-gray-400 transition hover:text-gray-800"
      >
        <ArrowLeft size={12} />
        Back to models
      </button>

      {/* Header */}
      <section className="mb-6 rounded-xl border border-gray-200 bg-white">
        <div className="p-5 sm:p-6">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="flex min-w-0 gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gray-950 text-white">
                <BrainCircuit size={20} />
              </div>

              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h1 className="text-[20px] font-semibold tracking-tight text-gray-950 sm:text-[23px]">
                    {model.name}
                  </h1>

                  <StatusBadge status={model.status} />
                </div>

                <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="font-mono text-[8px] text-gray-400">
                    {model.id}
                  </span>

                  <span className="text-gray-200">•</span>

                  <span className="text-[8px] text-gray-500">
                    {model.version}
                  </span>

                  <span className="text-gray-200">•</span>

                  <span className="text-[8px] text-gray-500">
                    {model.framework}
                  </span>
                </div>

                <p className="mt-3 max-w-3xl text-[9px] leading-5 text-gray-400">
                  {model.description}
                </p>
              </div>
            </div>

            <div className="flex shrink-0 flex-wrap gap-2">
              <button
                type="button"
                onClick={() =>
                  navigate("/app/inference")
                }
                className="flex h-9 items-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[9px] font-semibold text-white hover:bg-gray-800"
              >
                <Play size={11} />
                Run inference
              </button>

              <button
                type="button"
                className="flex h-9 items-center justify-center rounded-lg border border-gray-200 px-3 text-gray-500 hover:bg-gray-50"
              >
                <Settings2 size={13} />
              </button>

              <button
                type="button"
                className="flex h-9 items-center justify-center rounded-lg border border-gray-200 px-3 text-gray-500 hover:bg-gray-50"
              >
                <MoreHorizontal size={14} />
              </button>
            </div>
          </div>
        </div>

        {/* Metadata */}
        <div className="grid border-t border-gray-100 sm:grid-cols-2 lg:grid-cols-4">
          <div className="border-b border-gray-100 px-5 py-4 sm:border-r lg:border-b-0">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Modality
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <Layers3 size={12} className="text-gray-400" />
              {model.modality}
            </div>
          </div>

          <div className="border-b border-gray-100 px-5 py-4 lg:border-b-0 lg:border-r">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Task
            </div>

            <div className="mt-2 text-[9px] font-medium text-gray-700">
              {model.task}
            </div>
          </div>

          <div className="border-b border-gray-100 px-5 py-4 sm:border-r lg:border-b-0">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Framework
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <Code2 size={12} className="text-gray-400" />
              {model.framework}
            </div>
          </div>

          <div className="px-5 py-4">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Last updated
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <Clock3 size={12} className="text-gray-400" />
              {model.updated}
            </div>
          </div>
        </div>
      </section>

      {/* Performance */}
      <div className="mb-6">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h2 className="text-[12px] font-semibold text-gray-900">
              Performance
            </h2>

            <p className="mt-0.5 text-[8px] text-gray-400">
              Evaluation metrics for the current production version.
            </p>
          </div>

          <button
            type="button"
            className="flex items-center gap-1.5 text-[8px] font-medium text-gray-400 hover:text-gray-700"
          >
            <BarChart3 size={11} />
            Evaluation report
            <ArrowUpRight size={10} />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          <Metric
            label="Accuracy"
            value={`${model.accuracy}%`}
            description="Overall correctness"
          />

          <Metric
            label="Precision"
            value={`${model.precision}%`}
            description="Positive predictive value"
          />

          <Metric
            label="Recall"
            value={`${model.recall}%`}
            description="Sensitivity"
          />

          <Metric
            label="F1 score"
            value={`${model.f1}%`}
            description="Harmonic mean"
          />

          <Metric
            label="Latency"
            value={model.latency}
            description="Average inference time"
          />
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.45fr_1fr]">
        {/* Left */}
        <div className="space-y-6">
          {/* Inputs */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 p-5">
              <div className="flex items-center gap-2">
                <Database size={14} className="text-gray-400" />

                <h2 className="text-[11px] font-semibold text-gray-900">
                  Input configuration
                </h2>
              </div>

              <p className="mt-1 text-[8px] text-gray-400">
                Data accepted by this inference model.
              </p>
            </div>

            <div className="divide-y divide-gray-100">
              {model.modality === "Multimodal" && (
                <>
                  <div className="flex items-center gap-3 p-4">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gray-50">
                      <Table2 size={14} className="text-gray-500" />
                    </div>

                    <div className="flex-1">
                      <div className="text-[9px] font-semibold text-gray-700">
                        Clinical data
                      </div>

                      <div className="mt-1 text-[8px] text-gray-400">
                        Structured patient observations and clinical
                        measurements
                      </div>
                    </div>

                    <span className="rounded-md bg-green-50 px-2 py-1 text-[7px] font-semibold text-green-700">
                      Required
                    </span>
                  </div>

                  <div className="flex items-center gap-3 p-4">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gray-50">
                      <FileImage size={14} className="text-gray-500" />
                    </div>

                    <div className="flex-1">
                      <div className="text-[9px] font-semibold text-gray-700">
                        Medical image
                      </div>

                      <div className="mt-1 text-[8px] text-gray-400">
                        Chest radiograph in JPEG, PNG, or DICOM format
                      </div>
                    </div>

                    <span className="rounded-md bg-green-50 px-2 py-1 text-[7px] font-semibold text-green-700">
                      Required
                    </span>
                  </div>
                </>
              )}

              {model.modality === "Image" && (
                <div className="flex items-center gap-3 p-4">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gray-50">
                    <FileImage size={14} className="text-gray-500" />
                  </div>

                  <div className="flex-1">
                    <div className="text-[9px] font-semibold text-gray-700">
                      Medical image
                    </div>

                    <div className="mt-1 text-[8px] text-gray-400">
                      Supported image input for model inference
                    </div>
                  </div>

                  <span className="rounded-md bg-green-50 px-2 py-1 text-[7px] font-semibold text-green-700">
                    Required
                  </span>
                </div>
              )}

              {model.modality === "Tabular" && (
                <div className="flex items-center gap-3 p-4">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gray-50">
                    <FileSpreadsheet
                      size={14}
                      className="text-gray-500"
                    />
                  </div>

                  <div className="flex-1">
                    <div className="text-[9px] font-semibold text-gray-700">
                      Structured data
                    </div>

                    <div className="mt-1 text-[8px] text-gray-400">
                      Clinical and demographic features required by
                      the model
                    </div>
                  </div>

                  <span className="rounded-md bg-green-50 px-2 py-1 text-[7px] font-semibold text-green-700">
                    Required
                  </span>
                </div>
              )}
            </div>
          </section>

          {/* Endpoint */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 p-5">
              <div className="flex items-center gap-2">
                <Server size={14} className="text-gray-400" />

                <h2 className="text-[11px] font-semibold text-gray-900">
                  Deployment
                </h2>
              </div>
            </div>

            <div className="p-5">
              <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                Inference endpoint
              </div>

              <div className="mt-2 flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 p-3">
                <Code2 size={12} className="shrink-0 text-gray-400" />

                <code className="min-w-0 flex-1 truncate font-mono text-[8px] text-gray-600">
                  {model.endpoint}
                </code>

                <button
                  type="button"
                  className="shrink-0 text-gray-400 hover:text-gray-700"
                  aria-label="Open endpoint"
                >
                  <ExternalLink size={11} />
                </button>
              </div>

              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Environment
                  </div>

                  <div className="mt-1.5 flex items-center gap-1.5 text-[8px] font-medium text-gray-700">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                    Production
                  </div>
                </div>

                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Availability
                  </div>

                  <div className="mt-1.5 flex items-center gap-1.5 text-[8px] font-medium text-gray-700">
                    <ShieldCheck size={10} className="text-green-500" />
                    Operational
                  </div>
                </div>

                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Runs
                  </div>

                  <div className="mt-1.5 text-[8px] font-medium text-gray-700">
                    {model.inferences.toLocaleString()} this month
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Right */}
        <div className="space-y-6">
          {/* Recent activity */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">
                  Recent inference runs
                </h2>

                <p className="mt-1 text-[8px] text-gray-400">
                  Latest activity using this model.
                </p>
              </div>

              <button
                type="button"
                onClick={() =>
                  navigate("/app/inference/history")
                }
                className="text-[8px] font-medium text-gray-400 hover:text-gray-700"
              >
                View all
              </button>
            </div>

            <div className="divide-y divide-gray-100">
              {recentRuns.map((run) => (
                <button
                  key={run.id}
                  type="button"
                  onClick={() =>
                    navigate(`/app/inference/${run.id}`)
                  }
                  className="group flex w-full items-center gap-3 p-4 text-left hover:bg-gray-50"
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-50">
                    <Activity size={12} className="text-gray-400" />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-[9px] font-semibold text-gray-700">
                        {run.patient}
                      </span>

                      <span className="shrink-0 font-mono text-[7px] text-gray-400">
                        {run.confidence}
                      </span>
                    </div>

                    <div className="mt-1 flex items-center gap-2">
                      <span className="truncate text-[8px] text-gray-400">
                        {run.result}
                      </span>

                      <span className="text-gray-200">•</span>

                      <span className="shrink-0 text-[7px] text-gray-400">
                        {run.time}
                      </span>
                    </div>
                  </div>

                  <ArrowUpRight
                    size={11}
                    className="shrink-0 text-gray-300 opacity-0 transition group-hover:opacity-100"
                  />
                </button>
              ))}
            </div>
          </section>

          {/* Versions */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">
                  Version history
                </h2>

                <p className="mt-1 text-[8px] text-gray-400">
                  Previous model releases.
                </p>
              </div>

              <GitBranch size={13} className="text-gray-300" />
            </div>

            <div className="divide-y divide-gray-100">
              {versions.map((version, index) => (
                <div
                  key={version.version}
                  className="flex gap-3 p-4"
                >
                  <div className="relative flex w-4 shrink-0 justify-center">
                    <div
                      className={`mt-1.5 h-2 w-2 rounded-full ${index === 0
                        ? "bg-green-500"
                        : "bg-gray-300"
                        }`}
                    />

                    {index < versions.length - 1 && (
                      <div className="absolute top-3 h-full w-px bg-gray-100" />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-[9px] font-semibold text-gray-700">
                        {version.version}
                      </span>

                      {version.status === "Current" && (
                        <span className="rounded-md bg-green-50 px-2 py-1 text-[7px] font-semibold text-green-700">
                          Current
                        </span>
                      )}
                    </div>

                    <div className="mt-1 text-[8px] text-gray-400">
                      {version.changes}
                    </div>

                    <div className="mt-1.5 text-[7px] text-gray-300">
                      {version.date}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <button
              type="button"
              className="flex w-full items-center justify-center gap-1.5 border-t border-gray-100 py-3 text-[8px] font-medium text-gray-400 hover:bg-gray-50 hover:text-gray-700"
            >
              View complete history
              <ChevronDown size={10} />
            </button>
          </section>
        </div>
      </div>
    </div>
  );
}