import {
  Activity,
  ArrowUpRight,
  Brain,
  Cpu,
  GitBranch,
  Image as ImageIcon,
  Layers3,
  MoreHorizontal,
  Plus,
  RefreshCw,
  Server,
  Table2,
} from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";

const models = [
  {
    name: "Respiratory Assessment",
    id: "respiratory-assessment",
    version: "v2.4.1",
    task: "Classification",
    modality: "Multimodal",
    status: "Production",
    accuracy: "94.7%",
    inferences: "1,284",
    updated: "29 Aug 2026",
    description:
      "Combines chest imaging and structured clinical observations for respiratory assessment.",
  },
  {
    name: "Chest X-Ray Classifier",
    id: "chest-xray-classifier",
    version: "v3.1.0",
    task: "Classification",
    modality: "Image",
    status: "Production",
    accuracy: "96.2%",
    inferences: "3,842",
    updated: "27 Aug 2026",
    description:
      "Detects common radiographic patterns and abnormalities from chest X-ray images.",
  },
  {
    name: "Cardiovascular Risk",
    id: "cardiovascular-risk",
    version: "v1.8.2",
    task: "Risk prediction",
    modality: "Tabular",
    status: "Production",
    accuracy: "91.7%",
    inferences: "2,091",
    updated: "24 Aug 2026",
    description:
      "Estimates cardiovascular risk from structured patient and clinical measurements.",
  },
  {
    name: "Skin Lesion Analysis",
    id: "skin-lesion-analysis",
    version: "v2.2.0",
    task: "Classification",
    modality: "Image",
    status: "Review",
    accuracy: "89.3%",
    inferences: "746",
    updated: "19 Aug 2026",
    description:
      "Classifies visual characteristics of skin lesions to assist clinical review.",
  },
  {
    name: "General Clinical Assessment",
    id: "general-clinical-assessment",
    version: "v2.4.1",
    task: "Classification",
    modality: "Multimodal",
    status: "Staging",
    accuracy: "88.9%",
    inferences: "412",
    updated: "17 Aug 2026",
    description:
      "Experimental multimodal model for general clinical risk assessment.",
  },
];

function ModalityIcon({ modality }: { modality: string }) {
  if (modality === "Image") {
    return <ImageIcon size={14} />;
  }

  if (modality === "Tabular") {
    return <Table2 size={14} />;
  }

  return <Layers3 size={14} />;
}

function StatusBadge({ status }: { status: string }) {
  const production = status === "Production";
  const review = status === "Review";

  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[9px] font-semibold",
        production
          ? "bg-green-50 text-green-700"
          : review
            ? "bg-amber-50 text-amber-700"
            : "bg-blue-50 text-blue-700",
      ].join(" ")}
    >
      <span
        className={[
          "h-1.5 w-1.5 rounded-full",
          production
            ? "bg-green-500"
            : review
              ? "bg-amber-500"
              : "bg-blue-500",
        ].join(" ")}
      />

      {status}
    </span>
  );
}

export function ModelRegistry() {
  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      <PageHeader
        eyebrow="AI Inference Engine"
        title="Model registry"
        description="Manage deployed models, versions, modalities, and inference performance."
        action={
          <button className="flex h-9 items-center justify-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[10px] font-semibold text-white transition hover:bg-gray-800">
            <Plus size={14} />
            Register model
          </button>
        }
      />

      {/* Registry metrics */}
      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {[
          {
            label: "Registered models",
            value: "12",
            detail: "3 added this month",
            icon: Brain,
          },
          {
            label: "Production",
            value: "7",
            detail: "All systems operational",
            icon: Server,
          },
          {
            label: "Total inferences",
            value: "8,375",
            detail: "+14.2% this month",
            icon: Activity,
          },
          {
            label: "Avg. latency",
            value: "1.84s",
            detail: "−8.4% from last month",
            icon: Cpu,
          },
        ].map(({ label, value, detail, icon: Icon }) => (
          <div
            key={label}
            className="rounded-xl border border-gray-200 bg-white p-4 sm:p-5"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-400">{label}</span>

              <Icon size={14} className="text-gray-300" />
            </div>

            <div className="mt-3 text-[21px] font-semibold tracking-tight text-gray-950">
              {value}
            </div>

            <div className="mt-1 text-[9px] text-gray-400">{detail}</div>
          </div>
        ))}
      </div>

      {/* Models */}
      <section className="rounded-xl border border-gray-200 bg-white">
        <div className="flex flex-col gap-3 border-b border-gray-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-[13px] font-semibold text-gray-950">
              Registered models
            </h2>

            <p className="mt-0.5 text-[10px] text-gray-400">
              Available models and deployment states
            </p>
          </div>

          <button className="flex h-8 items-center gap-2 self-start rounded-lg border border-gray-200 px-3 text-[10px] font-medium text-gray-500 hover:bg-gray-50">
            <RefreshCw size={12} />
            Sync registry
          </button>
        </div>

        <div className="grid gap-3 p-4 sm:p-5 lg:grid-cols-2">
          {models.map((model) => (
            <article
              key={model.id}
              className="group rounded-xl border border-gray-200 p-4 transition hover:border-gray-300 hover:shadow-sm"
            >
              {/* Top */}
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-start gap-3">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gray-950 text-white">
                    <Brain size={16} />
                  </div>

                  <div className="min-w-0">
                    <div className="truncate text-[12px] font-semibold text-gray-900">
                      {model.name}
                    </div>

                    <div className="mt-0.5 truncate font-mono text-[9px] text-gray-400">
                      {model.id}
                    </div>
                  </div>
                </div>

                <button className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-gray-300 hover:bg-gray-50 hover:text-gray-600">
                  <MoreHorizontal size={15} />
                </button>
              </div>

              {/* Description */}
              <p className="mt-4 min-h-9 text-[10px] leading-4 text-gray-500">
                {model.description}
              </p>

              {/* Tags */}
              <div className="mt-4 flex flex-wrap gap-1.5">
                <StatusBadge status={model.status} />

                <span className="inline-flex items-center gap-1.5 rounded-md bg-gray-50 px-2 py-1 text-[9px] font-medium text-gray-500">
                  <ModalityIcon modality={model.modality} />
                  {model.modality}
                </span>

                <span className="rounded-md bg-gray-50 px-2 py-1 font-mono text-[9px] font-medium text-gray-500">
                  {model.version}
                </span>
              </div>

              {/* Metrics */}
              <div className="mt-5 grid grid-cols-3 border-t border-gray-100 pt-4">
                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Performance
                  </div>

                  <div className="mt-1 text-[11px] font-semibold text-gray-800">
                    {model.accuracy}
                  </div>
                </div>

                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Inferences
                  </div>

                  <div className="mt-1 text-[11px] font-semibold text-gray-800">
                    {model.inferences}
                  </div>
                </div>

                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Updated
                  </div>

                  <div className="mt-1 text-[10px] text-gray-600">
                    {model.updated}
                  </div>
                </div>
              </div>

              {/* Action */}
              <button className="mt-4 flex w-full items-center justify-between border-t border-gray-100 pt-3 text-[10px] font-semibold text-gray-500 transition group-hover:text-gray-900">
                View model details
                <ArrowUpRight size={13} />
              </button>
            </article>
          ))}
        </div>
      </section>

      {/* Architecture / capability panel */}
      <section className="mt-5 overflow-hidden rounded-xl border border-gray-200 bg-gray-950 text-white">
        <div className="grid lg:grid-cols-[1fr_1.3fr]">
          <div className="p-6 sm:p-7">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white/10">
              <GitBranch size={16} />
            </div>

            <h2 className="mt-5 text-[17px] font-semibold tracking-tight">
              One inference layer.
              <br />
              Multiple modalities.
            </h2>

            <p className="mt-3 max-w-md text-[10px] leading-5 text-gray-400">
              The inference engine provides a unified interface for tabular,
              image, and multimodal clinical models while keeping model
              execution isolated from the patient management layer.
            </p>
          </div>

          <div className="grid grid-cols-1 border-t border-white/10 sm:grid-cols-3 lg:border-l lg:border-t-0">
            {[
              {
                icon: Table2,
                title: "Tabular",
                detail: "Structured clinical data",
              },
              {
                icon: ImageIcon,
                title: "Image",
                detail: "Medical imaging inputs",
              },
              {
                icon: Layers3,
                title: "Multimodal",
                detail: "Combined clinical inputs",
              },
            ].map(({ icon: Icon, title, detail }) => (
              <div
                key={title}
                className="border-b border-white/10 p-6 last:border-b-0 sm:border-b-0 sm:border-r sm:last:border-r-0"
              >
                <Icon size={17} className="text-gray-400" />

                <div className="mt-8 text-[12px] font-semibold">
                  {title}
                </div>

                <div className="mt-1 text-[9px] leading-4 text-gray-500">
                  {detail}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}