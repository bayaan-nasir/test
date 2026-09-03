import {
  Activity,
  ArrowLeft,
  ArrowUpRight,
  Code2,
  Database,
  FileImage,
  FileSpreadsheet,
  Image as ImageIcon,
  Play,
  Server,
  ShieldCheck,
  Table2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import {
  getModelRegistry,
  getModelStats,
  type ModelRegistryEntry,
  type ModelUsageStat,
} from "../../api/inference";

function formatDateTime(value?: string | null) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "—";
  return parsed.toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function Metric({ label, value, description }: { label: string; value: string; description: string }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">{label}</div>
      <div className="mt-2 text-[20px] font-semibold tracking-tight text-gray-950">{value}</div>
      <div className="mt-1 text-[8px] text-gray-400">{description}</div>
    </div>
  );
}

export function ModelDetails() {
  const navigate = useNavigate();
  const { modelId } = useParams();

  const [model, setModel] = useState<ModelRegistryEntry | null>(null);
  const [stat, setStat] = useState<ModelUsageStat | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!modelId) {
      setError("No model specified.");
      setIsLoading(false);
      return;
    }
    const id: string = modelId;
    let active = true;

    async function load() {
      setIsLoading(true);
      setError(null);
      try {
        const [registry, stats] = await Promise.all([getModelRegistry(), getModelStats()]);
        if (!active) return;

        const found = registry.models.find((m) => m.model_id === modelId);
        if (!found) {
          setError("This model could not be found in the registry.");
          return;
        }

        setModel(found);
        setStat(stats[id] ?? null);
      } catch {
        if (active) setError("Unable to load this model. The ML service may be unavailable.");
      } finally {
        if (active) setIsLoading(false);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, [modelId]);

  if (isLoading) {
    return (
      <div className="flex min-h-100 items-center justify-center">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-gray-900" />
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
        <button
          type="button"
          onClick={() => navigate("/app/models")}
          className="mb-5 flex items-center gap-2 text-[9px] font-medium text-gray-400 transition hover:text-gray-800"
        >
          <ArrowLeft size={12} />
          Back to models
        </button>
        <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
          <p className="text-[11px] font-semibold text-gray-700">
            {error ?? "This model could not be found."}
          </p>
        </div>
      </div>
    );
  }

  const isImage = model.modality === "image";
  const ModalityIcon = isImage ? ImageIcon : Table2;

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      <button
        type="button"
        onClick={() => navigate("/app/models")}
        className="mb-5 flex items-center gap-2 text-[9px] font-medium text-gray-400 transition hover:text-gray-800"
      >
        <ArrowLeft size={12} />
        Back to models
      </button>

      <section className="mb-6 rounded-xl border border-gray-200 bg-white">
        <div className="p-5 sm:p-6">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="flex min-w-0 gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gray-950 text-white">
                <ModalityIcon size={20} />
              </div>

              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h1 className="text-[20px] font-semibold tracking-tight text-gray-950 sm:text-[23px]">
                    {model.model_name}
                  </h1>
                  <span
                    className={[
                      "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[8px] font-semibold",
                      model.production_ready ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700",
                    ].join(" ")}
                  >
                    <span
                      className={[
                        "h-1.5 w-1.5 rounded-full",
                        model.production_ready ? "bg-green-500" : "bg-amber-500",
                      ].join(" ")}
                    />
                    {model.production_ready ? "Production" : "Staging"}
                  </span>
                </div>

                <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="font-mono text-[8px] text-gray-400">{model.model_id}</span>
                  <span className="text-gray-200">•</span>
                  <span className="text-[8px] text-gray-500">{model.version}</span>
                  <span className="text-gray-200">•</span>
                  <span className="text-[8px] text-gray-500">{model.framework.join(", ")}</span>
                </div>

                <p className="mt-3 max-w-3xl text-[9px] leading-5 text-gray-400">
                  {model.task} — {isImage ? "image-based" : "structured tabular"} classification.
                </p>
              </div>
            </div>

            <div className="flex shrink-0 flex-wrap gap-2">
              <button
                type="button"
                onClick={() => navigate("/app/inference")}
                className="flex h-9 items-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[9px] font-semibold text-white hover:bg-gray-800"
              >
                <Play size={11} />
                Run inference
              </button>
            </div>
          </div>
        </div>
      </section>

      <div className="mb-6">
        <div className="mb-3">
          <h2 className="text-[12px] font-semibold text-gray-900">Usage</h2>
          <p className="mt-0.5 text-[8px] text-gray-400">
            Derived from real completed inferences — not a benchmark evaluation.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          <Metric
            label="Total runs"
            value={String(stat?.total_runs ?? 0)}
            description="Completed inferences using this model"
          />
          <Metric
            label="Avg. confidence"
            value={
              stat?.avg_confidence_pct !== null && stat?.avg_confidence_pct !== undefined
                ? `${stat.avg_confidence_pct}%`
                : "—"
            }
            description="Mean model confidence across runs, not accuracy"
          />
          <Metric
            label="Avg. request time"
            value={
              stat?.avg_latency_seconds !== null && stat?.avg_latency_seconds !== undefined
                ? `${stat.avg_latency_seconds}s`
                : "—"
            }
            description="Whole request duration, not per-model timing"
          />
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.45fr_1fr]">
        <div className="space-y-6">
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 p-5">
              <div className="flex items-center gap-2">
                <Database size={14} className="text-gray-400" />
                <h2 className="text-[11px] font-semibold text-gray-900">Input schema</h2>
              </div>
              <p className="mt-1 text-[8px] text-gray-400">
                Fields accepted by this model, from the live ML service registry.
              </p>
            </div>

            {isImage ? (
              <div className="flex items-center gap-3 p-4">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gray-50">
                  <FileImage size={14} className="text-gray-500" />
                </div>
                <div className="flex-1">
                  <div className="text-[9px] font-semibold text-gray-700">Medical image</div>
                  <div className="mt-1 text-[8px] text-gray-400">
                    JPEG or PNG upload, routed by image type "{model.image_type_value}"
                  </div>
                </div>
                <span className="rounded-md bg-green-50 px-2 py-1 text-[7px] font-semibold text-green-700">
                  Required
                </span>
              </div>
            ) : (
              <div className="divide-y divide-gray-100">
                {(model.fields ?? []).map((field) => (
                  <div key={field.name} className="flex items-center gap-3 p-4">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-50">
                      <FileSpreadsheet size={13} className="text-gray-500" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="font-mono text-[9px] font-semibold text-gray-700">{field.name}</div>
                      {field.description && (
                        <div className="mt-0.5 text-[8px] text-gray-400">{field.description}</div>
                      )}
                    </div>
                    <span
                      className={[
                        "shrink-0 rounded-md px-2 py-1 text-[7px] font-semibold",
                        field.required ? "bg-green-50 text-green-700" : "bg-gray-50 text-gray-400",
                      ].join(" ")}
                    >
                      {field.required ? "Required" : "Optional"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 p-5">
              <div className="flex items-center gap-2">
                <Server size={14} className="text-gray-400" />
                <h2 className="text-[11px] font-semibold text-gray-900">Deployment</h2>
              </div>
            </div>

            <div className="space-y-4 p-5">
              {Object.entries(model.endpoints ?? {}).map(([label, path]) => (
                <div key={label}>
                  <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                    {label} endpoint
                  </div>
                  <div className="mt-2 flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 p-3">
                    <Code2 size={12} className="shrink-0 text-gray-400" />
                    <code className="min-w-0 flex-1 truncate font-mono text-[8px] text-gray-600">{path}</code>
                  </div>
                </div>
              ))}

              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">Environment</div>
                  <div className="mt-1.5 flex items-center gap-1.5 text-[8px] font-medium text-gray-700">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                    {model.production_ready ? "Production" : "Staging"}
                  </div>
                </div>
                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">Last run</div>
                  <div className="mt-1.5 flex items-center gap-1.5 text-[8px] font-medium text-gray-700">
                    <ShieldCheck size={10} className="text-green-500" />
                    {formatDateTime(stat?.last_run_at)}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>

        <div className="space-y-6">
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">Recent inference runs</h2>
                <p className="mt-1 text-[8px] text-gray-400">Latest completed runs that used this model.</p>
              </div>
              <button
                type="button"
                onClick={() => navigate("/app/inference/history")}
                className="text-[8px] font-medium text-gray-400 hover:text-gray-700"
              >
                View all
              </button>
            </div>

            {(stat?.recent_runs?.length ?? 0) === 0 ? (
              <div className="p-5 text-center text-[9px] text-gray-400">
                No completed runs yet for this model.
              </div>
            ) : (
              <div className="divide-y divide-gray-100">
                {stat!.recent_runs.map((run) => (
                  <button
                    key={run.inference_id}
                    type="button"
                    onClick={() => navigate(`/app/inference/${run.inference_id}`)}
                    className="group flex w-full items-center gap-3 p-4 text-left hover:bg-gray-50"
                  >
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-50">
                      <Activity size={12} className="text-gray-400" />
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-[9px] font-semibold text-gray-700">
                          {run.patient_name}
                        </span>
                        <span className="shrink-0 font-mono text-[7px] text-gray-400">{run.confidence_pct}</span>
                      </div>

                      <div className="mt-1 flex items-center gap-2">
                        <span className="truncate text-[8px] text-gray-400">{run.predicted_class}</span>
                        <span className="text-gray-200">•</span>
                        <span className="shrink-0 text-[7px] text-gray-400">
                          {formatDateTime(run.created_at)}
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
            )}
          </section>
        </div>
      </div>
    </div>
  );
}