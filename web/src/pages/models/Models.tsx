import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  ChevronRight,
  Clock3,
  Database,
  Image as ImageIcon,
  Layers3,
  Plus,
  Search,
  Server,
  Table2,
  Zap,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

type ModelStatus = "Production" | "Staging" | "Inactive";
type Modality = "Multimodal" | "Image" | "Tabular";

interface Model {
  id: string;
  name: string;
  description: string;
  version: string;
  status: ModelStatus;
  modality: Modality;
  task: string;
  accuracy: number;
  latency: string;
  inferences: number;
  lastUpdated: string;
}

const models: Model[] = [
  {
    id: "mdl-respiratory",
    name: "Respiratory Assessment",
    description:
      "Multimodal respiratory condition assessment using clinical observations and chest imaging.",
    version: "v2.4.1",
    status: "Production",
    modality: "Multimodal",
    task: "Classification",
    accuracy: 94.7,
    latency: "2.84s",
    inferences: 428,
    lastUpdated: "29 Aug 2026",
  },
  {
    id: "mdl-cxr",
    name: "Chest X-Ray Classifier",
    description:
      "Image-based classification model for detecting abnormalities in chest radiographs.",
    version: "v3.1.0",
    status: "Production",
    modality: "Image",
    task: "Classification",
    accuracy: 96.2,
    latency: "1.42s",
    inferences: 617,
    lastUpdated: "28 Aug 2026",
  },
  {
    id: "mdl-cardio",
    name: "Cardiovascular Risk",
    description:
      "Risk stratification model using structured patient clinical and demographic data.",
    version: "v1.8.3",
    status: "Production",
    modality: "Tabular",
    task: "Risk prediction",
    accuracy: 88.4,
    latency: "0.82s",
    inferences: 239,
    lastUpdated: "26 Aug 2026",
  },
  {
    id: "mdl-derma",
    name: "Dermatology Assistant",
    description:
      "Experimental image classification model for dermatological observations.",
    version: "v0.9.2",
    status: "Staging",
    modality: "Image",
    task: "Classification",
    accuracy: 91.3,
    latency: "1.76s",
    inferences: 87,
    lastUpdated: "24 Aug 2026",
  },
  {
    id: "mdl-lab",
    name: "Laboratory Risk Engine",
    description:
      "Experimental model for identifying elevated clinical risk from laboratory measurements.",
    version: "v0.4.0",
    status: "Staging",
    modality: "Tabular",
    task: "Risk prediction",
    accuracy: 86.9,
    latency: "0.63s",
    inferences: 42,
    lastUpdated: "20 Aug 2026",
  },
];

const modalityIcons = {
  Multimodal: Layers3,
  Image: ImageIcon,
  Tabular: Table2,
};

function StatusBadge({ status }: { status: ModelStatus }) {
  if (status === "Production") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-green-50 px-2 py-1 text-[8px] font-semibold text-green-700">
        <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
        Production
      </span>
    );
  }

  if (status === "Staging") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-amber-50 px-2 py-1 text-[8px] font-semibold text-amber-700">
        <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
        Staging
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 rounded-md bg-gray-100 px-2 py-1 text-[8px] font-semibold text-gray-500">
      <span className="h-1.5 w-1.5 rounded-full bg-gray-400" />
      Inactive
    </span>
  );
}

function ModalityBadge({ modality }: { modality: Modality }) {
  const Icon = modalityIcons[modality];

  return (
    <span className="inline-flex items-center gap-1.5 text-[8px] font-medium text-gray-500">
      <Icon size={11} className="text-gray-400" />
      {modality}
    </span>
  );
}

export function Models() {
  const navigate = useNavigate();

  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<"All" | ModelStatus>("All");
  const [modality, setModality] = useState<"All" | Modality>("All");

  const filteredModels = useMemo(() => {
    const query = search.trim().toLowerCase();

    return models.filter((model) => {
      const matchesSearch =
        !query ||
        model.name.toLowerCase().includes(query) ||
        model.description.toLowerCase().includes(query) ||
        model.id.toLowerCase().includes(query);

      const matchesStatus =
        status === "All" || model.status === status;

      const matchesModality =
        modality === "All" || model.modality === modality;

      return matchesSearch && matchesStatus && matchesModality;
    });
  }, [search, status, modality]);

  const productionModels = models.filter(
    (model) => model.status === "Production",
  ).length;

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Header */}
      <div className="mb-7 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-blue-600">
            <BrainCircuit size={11} />
            AI Infrastructure
          </div>

          <h1 className="mt-2 text-[25px] font-semibold tracking-tight text-gray-950 sm:text-[29px]">
            Models
          </h1>

          <p className="mt-1 max-w-xl text-[10px] leading-5 text-gray-400">
            Manage deployed inference models, versions, modalities,
            and model performance.
          </p>
        </div>

        <button
          type="button"
          className="flex h-9 items-center justify-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[10px] font-semibold text-white hover:bg-gray-800"
        >
          <Plus size={13} />
          Register model
        </button>
      </div>

      {/* Overview */}
      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="flex items-center justify-between">
            <span className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
              Total models
            </span>

            <BrainCircuit size={13} className="text-gray-300" />
          </div>

          <div className="mt-3 text-[22px] font-semibold tracking-tight text-gray-950">
            {models.length}
          </div>

          <div className="mt-1 text-[8px] text-gray-400">
            Across all environments
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="flex items-center justify-between">
            <span className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
              Production
            </span>

            <Server size={13} className="text-gray-300" />
          </div>

          <div className="mt-3 text-[22px] font-semibold tracking-tight text-gray-950">
            {productionModels}
          </div>

          <div className="mt-1 text-[8px] text-green-600">
            All systems operational
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="flex items-center justify-between">
            <span className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
              Total runs
            </span>

            <Activity size={13} className="text-gray-300" />
          </div>

          <div className="mt-3 text-[22px] font-semibold tracking-tight text-gray-950">
            1,413
          </div>

          <div className="mt-1 text-[8px] text-gray-400">
            Last 30 days
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="flex items-center justify-between">
            <span className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
              Avg. latency
            </span>

            <Zap size={13} className="text-gray-300" />
          </div>

          <div className="mt-3 text-[22px] font-semibold tracking-tight text-gray-950">
            1.84s
          </div>

          <div className="mt-1 text-[8px] text-gray-400">
            Across production models
          </div>
        </div>
      </div>

      {/* Production highlight */}
      <section className="mb-6 overflow-hidden rounded-xl border border-gray-200 bg-gray-950 text-white">
        <div className="grid lg:grid-cols-[1fr_auto]">
          <div className="p-5 sm:p-6">
            <div className="flex items-center gap-2 text-[8px] font-semibold uppercase tracking-[0.14em] text-gray-400">
              <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
              Production model
            </div>

            <div className="mt-3 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="text-[19px] font-semibold tracking-tight">
                  Respiratory Assessment
                </h2>

                <p className="mt-1 max-w-xl text-[9px] leading-5 text-gray-400">
                  Multimodal model combining structured clinical data
                  with medical imaging for respiratory assessment.
                </p>
              </div>

              <button
                type="button"
                onClick={() =>
                  navigate("/app/models/mdl-respiratory")
                }
                className="flex shrink-0 items-center gap-2 text-[9px] font-semibold text-gray-300 hover:text-white"
              >
                View model
                <ArrowUpRight size={12} />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-3 border-t border-gray-800 sm:grid-cols-3 lg:border-l lg:border-t-0">
            <div className="border-r border-gray-800 px-4 py-5 lg:px-6">
              <div className="text-[8px] uppercase tracking-wide text-gray-500">
                Confidence
              </div>

              <div className="mt-2 text-[17px] font-semibold">
                94.7%
              </div>
            </div>

            <div className="border-r border-gray-800 px-4 py-5 lg:px-6">
              <div className="text-[8px] uppercase tracking-wide text-gray-500">
                Latency
              </div>

              <div className="mt-2 text-[17px] font-semibold">
                2.84s
              </div>
            </div>

            <div className="px-4 py-5 lg:px-6">
              <div className="text-[8px] uppercase tracking-wide text-gray-500">
                Runs
              </div>

              <div className="mt-2 text-[17px] font-semibold">
                428
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Toolbar */}
      <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
        <div className="flex flex-col gap-3 border-b border-gray-100 p-4 lg:flex-row lg:items-center">
          <div className="relative min-w-0 flex-1 lg:max-w-100">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300"
            />

            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search models..."
              className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 pl-9 pr-3 text-[9px] text-gray-700 outline-none placeholder:text-gray-300 focus:border-gray-300 focus:bg-white"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <div className="flex rounded-lg border border-gray-200 p-0.5">
              {(["All", "Production", "Staging"] as const).map(
                (item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setStatus(item)}
                    className={[
                      "rounded-md px-2.5 py-1.5 text-[8px] font-medium transition",
                      status === item
                        ? "bg-gray-900 text-white"
                        : "text-gray-500 hover:bg-gray-50",
                    ].join(" ")}
                  >
                    {item}
                  </button>
                ),
              )}
            </div>

            <div className="flex rounded-lg border border-gray-200 p-0.5">
              {(
                ["All", "Multimodal", "Image", "Tabular"] as const
              ).map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setModality(item)}
                  className={[
                    "rounded-md px-2.5 py-1.5 text-[8px] font-medium transition",
                    modality === item
                      ? "bg-gray-900 text-white"
                      : "text-gray-500 hover:bg-gray-50",
                  ].join(" ")}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Model list */}
        <div className="divide-y divide-gray-100">
          {filteredModels.map((model) => {
            const Icon = modalityIcons[model.modality];

            return (
              <button
                key={model.id}
                type="button"
                onClick={() =>
                  navigate(`/app/models/${model.id}`)
                }
                className="group block w-full p-5 text-left transition hover:bg-gray-50/70"
              >
                <div className="flex flex-col gap-5 xl:flex-row xl:items-center">
                  {/* Model identity */}
                  <div className="flex min-w-0 flex-1 items-start gap-3">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-gray-200 bg-gray-50">
                      <Icon size={16} className="text-gray-500" />
                    </div>

                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-[11px] font-semibold text-gray-900">
                          {model.name}
                        </h3>

                        <StatusBadge status={model.status} />
                      </div>

                      <p className="mt-1 max-w-2xl text-[8px] leading-4 text-gray-400">
                        {model.description}
                      </p>

                      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
                        <span className="font-mono text-[8px] text-gray-400">
                          {model.version}
                        </span>

                        <span className="text-gray-200">•</span>

                        <ModalityBadge modality={model.modality} />

                        <span className="text-gray-200">•</span>

                        <span className="text-[8px] text-gray-400">
                          {model.task}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-3 gap-5 border-t border-gray-100 pt-4 sm:grid-cols-4 xl:w-115 xl:border-t-0 xl:pt-0">
                    <div>
                      <div className="text-[7px] uppercase tracking-wide text-gray-400">
                        Accuracy
                      </div>

                      <div className="mt-1.5 text-[10px] font-semibold text-gray-800">
                        {model.accuracy}%
                      </div>

                      <div className="mt-1.5 h-1 w-16 overflow-hidden rounded-full bg-gray-100">
                        <div
                          className="h-full rounded-full bg-gray-800"
                          style={{
                            width: `${model.accuracy}%`,
                          }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="text-[7px] uppercase tracking-wide text-gray-400">
                        Latency
                      </div>

                      <div className="mt-1.5 flex items-center gap-1 text-[10px] font-semibold text-gray-800">
                        <Clock3 size={10} className="text-gray-400" />
                        {model.latency}
                      </div>
                    </div>

                    <div>
                      <div className="text-[7px] uppercase tracking-wide text-gray-400">
                        Inferences
                      </div>

                      <div className="mt-1.5 text-[10px] font-semibold text-gray-800">
                        {model.inferences.toLocaleString()}
                      </div>
                    </div>

                    <div className="hidden sm:block">
                      <div className="text-[7px] uppercase tracking-wide text-gray-400">
                        Updated
                      </div>

                      <div className="mt-1.5 text-[8px] text-gray-600">
                        {model.lastUpdated}
                      </div>
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className="hidden xl:flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-gray-300 transition group-hover:bg-gray-100 group-hover:text-gray-700">
                    <ChevronRight size={14} />
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {filteredModels.length === 0 && (
          <div className="flex flex-col items-center justify-center px-5 py-16 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gray-100">
              <Search size={16} className="text-gray-400" />
            </div>

            <div className="mt-4 text-[11px] font-semibold text-gray-700">
              No models found
            </div>

            <p className="mt-1 text-[9px] text-gray-400">
              Try changing your search or filters.
            </p>
          </div>
        )}

        <div className="flex items-center justify-between border-t border-gray-100 px-5 py-3">
          <span className="text-[8px] text-gray-400">
            {filteredModels.length} models
          </span>

          <span className="flex items-center gap-1.5 text-[8px] text-gray-400">
            <Database size={10} />
            Model registry connected
          </span>
        </div>
      </section>
    </div>
  );
}