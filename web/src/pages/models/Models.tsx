import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  ChevronRight,
  Clock3,
  Cpu,
  Image as ImageIcon,
  Search,
  Server,
  Table2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  getModelRegistry,
  getModelStats,
  type ModelRegistryEntry,
  type ModelStatsResponse,
} from "../../api/inference";

type ModalityFilter = "All" | "tabular" | "image";
type StatusFilter = "All" | "Production" | "Staging";

function formatDate(value?: string | null) {
  if (!value) return "Never run";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "Never run";
  return parsed.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

function modalityLabel(modality: string) {
  return modality === "image" ? "Image" : "Tabular";
}

const modalityIcons: Record<string, typeof Table2> = {
  tabular: Table2,
  image: ImageIcon,
};

export function Models() {
  const navigate = useNavigate();

  const [registryModels, setRegistryModels] = useState<ModelRegistryEntry[]>([]);
  const [stats, setStats] = useState<ModelStatsResponse>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<StatusFilter>("All");
  const [modality, setModality] = useState<ModalityFilter>("All");

  useEffect(() => {
    let active = true;

    async function load() {
      setIsLoading(true);
      setError(null);
      try {
        const [registry, usage] = await Promise.all([getModelRegistry(), getModelStats()]);
        if (!active) return;
        setRegistryModels(registry.models);
        setStats(usage);
      } catch {
        if (active) setError("Unable to load the model registry. The ML service may be unavailable.");
      } finally {
        if (active) setIsLoading(false);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  const filteredModels = useMemo(() => {
    const query = search.trim().toLowerCase();

    return registryModels.filter((model) => {
      const statusLabel = model.production_ready ? "Production" : "Staging";
      const matchesSearch =
        !query ||
        model.model_name.toLowerCase().includes(query) ||
        model.model_id.toLowerCase().includes(query);
      const matchesStatus = status === "All" || status === statusLabel;
      const matchesModality = modality === "All" || modality === model.modality;

      return matchesSearch && matchesStatus && matchesModality;
    });
  }, [registryModels, search, status, modality]);

  const totalRuns = useMemo(
    () => Object.values(stats).reduce((sum, s) => sum + s.total_runs, 0),
    [stats],
  );

  const productionCount = registryModels.filter((m) => m.production_ready).length;

  const avgLatency = useMemo(() => {
    const values = Object.values(stats)
      .map((s) => s.avg_latency_seconds)
      .filter((v): v is number => v !== null);
    if (values.length === 0) return null;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }, [stats]);

  const mostActiveModel = useMemo(() => {
    let best: { modelId: string; runs: number } | null = null;
    for (const [modelId, s] of Object.entries(stats)) {
      if (s.total_runs > 0 && (!best || s.total_runs > best.runs)) {
        best = { modelId, runs: s.total_runs };
      }
    }
    if (!best) return null;
    const model = registryModels.find((m) => m.model_id === best!.modelId);
    return model ? { model, stat: stats[best.modelId] } : null;
  }, [stats, registryModels]);

  if (isLoading) {
    return (
      <div className="flex min-h-100 items-center justify-center">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-gray-900" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      <div className="mb-7 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-blue-600">
            <BrainCircuit size={11} />
            AI Infrastructure
          </div>
          <h1 className="mt-2 text-[25px] font-semibold tracking-tight text-gray-950 sm:text-[29px]">Models</h1>
          <p className="mt-1 max-w-xl text-[10px] leading-5 text-gray-400">
            Live model registry and real usage statistics derived from completed inferences.
          </p>
        </div>
      </div>

      {error && (
        <div className="mb-5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-[9px] text-red-700">
          {error}
        </div>
      )}

      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          {
            label: "Registered models",
            value: String(registryModels.length),
            detail: "From the live ML service registry",
            icon: BrainCircuit,
          },
          {
            label: "Production",
            value: String(productionCount),
            detail: `${registryModels.length - productionCount} in staging`,
            icon: Server,
          },
          {
            label: "Total inferences",
            value: totalRuns.toLocaleString(),
            detail: "Across all completed runs",
            icon: Activity,
          },
          {
            label: "Avg. request time",
            value: avgLatency !== null ? `${avgLatency.toFixed(2)}s` : "—",
            detail: "Across models with recorded runs",
            icon: Cpu,
          },
        ].map(({ label, value, detail, icon: Icon }) => (
          <div key={label} className="rounded-xl border border-gray-200 bg-white p-4 sm:p-5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-400">{label}</span>
              <Icon size={14} className="text-gray-300" />
            </div>
            <div className="mt-3 text-[21px] font-semibold tracking-tight text-gray-950">{value}</div>
            <div className="mt-1 text-[9px] text-gray-400">{detail}</div>
          </div>
        ))}
      </div>

      {mostActiveModel ? (
        <section className="mb-6 overflow-hidden rounded-xl border border-gray-200 bg-gray-950 text-white">
          <div className="grid lg:grid-cols-[1fr_auto]">
            <div className="p-5 sm:p-6">
              <div className="flex items-center gap-2 text-[8px] font-semibold uppercase tracking-[0.14em] text-gray-400">
                <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
                Most active model
              </div>
              <div className="mt-3 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <h2 className="text-[19px] font-semibold tracking-tight">{mostActiveModel.model.model_name}</h2>
                  <p className="mt-1 max-w-xl text-[9px] leading-5 text-gray-400">
                    {mostActiveModel.model.task} · {modalityLabel(mostActiveModel.model.modality)}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => navigate(`/app/models/${mostActiveModel.model.model_id}`)}
                  className="flex shrink-0 items-center gap-2 text-[9px] font-semibold text-gray-300 hover:text-white"
                >
                  View model
                  <ArrowUpRight size={12} />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-3 border-t border-gray-800 lg:border-l lg:border-t-0">
              <div className="border-r border-gray-800 px-4 py-5 lg:px-6">
                <div className="text-[8px] uppercase tracking-wide text-gray-500">Runs</div>
                <div className="mt-2 text-[17px] font-semibold">{mostActiveModel.stat.total_runs}</div>
              </div>
              <div className="border-r border-gray-800 px-4 py-5 lg:px-6">
                <div className="text-[8px] uppercase tracking-wide text-gray-500">Avg. confidence</div>
                <div className="mt-2 text-[17px] font-semibold">
                  {mostActiveModel.stat.avg_confidence_pct !== null
                    ? `${mostActiveModel.stat.avg_confidence_pct}%`
                    : "—"}
                </div>
              </div>
              <div className="px-4 py-5 lg:px-6">
                <div className="text-[8px] uppercase tracking-wide text-gray-500">Last run</div>
                <div className="mt-2 text-[13px] font-semibold">{formatDate(mostActiveModel.stat.last_run_at)}</div>
              </div>
            </div>
          </div>
        </section>
      ) : (
        <section className="mb-6 rounded-xl border border-gray-200 bg-gray-50/50 p-6 text-center">
          <p className="text-[10px] text-gray-400">
            No completed inferences yet — usage statistics will appear here once clinicians start running the
            models.
          </p>
        </section>
      )}

      <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
        <div className="flex flex-col gap-3 border-b border-gray-100 p-4 lg:flex-row lg:items-center">
          <div className="relative min-w-0 flex-1 lg:max-w-100">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search models..."
              className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 pl-9 pr-3 text-[9px] text-gray-700 outline-none placeholder:text-gray-300 focus:border-gray-300 focus:bg-white"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <div className="flex rounded-lg border border-gray-200 p-0.5">
              {(["All", "Production", "Staging"] as const).map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setStatus(item)}
                  className={[
                    "rounded-md px-2.5 py-1.5 text-[8px] font-medium transition",
                    status === item ? "bg-gray-900 text-white" : "text-gray-500 hover:bg-gray-50",
                  ].join(" ")}
                >
                  {item}
                </button>
              ))}
            </div>

            <div className="flex rounded-lg border border-gray-200 p-0.5">
              {(["All", "tabular", "image"] as const).map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setModality(item)}
                  className={[
                    "rounded-md px-2.5 py-1.5 text-[8px] font-medium transition",
                    modality === item ? "bg-gray-900 text-white" : "text-gray-500 hover:bg-gray-50",
                  ].join(" ")}
                >
                  {item === "All" ? "All" : modalityLabel(item)}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="divide-y divide-gray-100">
          {filteredModels.map((model) => {
            const Icon = modalityIcons[model.modality] ?? Table2;
            const stat = stats[model.model_id];

            return (
              <button
                key={model.model_id}
                type="button"
                onClick={() => navigate(`/app/models/${model.model_id}`)}
                className="group flex w-full flex-col gap-5 p-5 text-left transition hover:bg-gray-50/70 xl:flex-row xl:items-center"
              >
                <div className="flex min-w-0 flex-1 items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-gray-200 bg-gray-50">
                    <Icon size={16} className="text-gray-500" />
                  </div>

                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-[11px] font-semibold text-gray-900">{model.model_name}</h3>
                      <span
                        className={[
                          "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[9px] font-semibold",
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

                    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
                      <span className="font-mono text-[8px] text-gray-400">{model.model_id}</span>
                      <span className="text-gray-200">•</span>
                      <span className="text-[8px] text-gray-500">{model.task}</span>
                      <span className="text-gray-200">•</span>
                      <span className="text-[8px] text-gray-500">{modalityLabel(model.modality)}</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-5 border-t border-gray-100 pt-4 xl:w-100 xl:border-t-0 xl:pt-0">
                  <div>
                    <div className="text-[7px] uppercase tracking-wide text-gray-400">Runs</div>
                    <div className="mt-1 text-[10px] font-semibold text-gray-800">{stat?.total_runs ?? 0}</div>
                  </div>
                  <div>
                    <div className="text-[7px] uppercase tracking-wide text-gray-400">Avg. confidence</div>
                    <div className="mt-1 text-[10px] font-semibold text-gray-800">
                      {stat?.avg_confidence_pct !== null && stat?.avg_confidence_pct !== undefined
                        ? `${stat.avg_confidence_pct}%`
                        : "—"}
                    </div>
                  </div>
                  <div>
                    <div className="text-[7px] uppercase tracking-wide text-gray-400">Last run</div>
                    <div className="mt-1 text-[9px] text-gray-600">{formatDate(stat?.last_run_at)}</div>
                  </div>
                </div>

                <ChevronRight
                  size={14}
                  className="hidden shrink-0 text-gray-300 transition group-hover:text-gray-600 xl:block"
                />
              </button>
            );
          })}
        </div>

        {filteredModels.length === 0 && (
          <div className="flex flex-col items-center justify-center px-5 py-16 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gray-100">
              <Search size={16} className="text-gray-400" />
            </div>
            <div className="mt-4 text-[11px] font-semibold text-gray-700">No models found</div>
            <p className="mt-1 text-[9px] text-gray-400">Try changing your search or filters.</p>
          </div>
        )}

        <div className="flex items-center justify-between border-t border-gray-100 px-5 py-3">
          <span className="text-[8px] text-gray-400">{filteredModels.length} models</span>
          <span className="flex items-center gap-1.5 text-[8px] text-gray-400">
            <Clock3 size={10} />
            Live from ML service registry
          </span>
        </div>
      </section>
    </div>
  );
}