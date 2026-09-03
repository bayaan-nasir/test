import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Eye,
  Loader2,
  RefreshCw,
  Search,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { getInferences, type Inference } from "../../api/inference";

type StatusFilter = "All" | Inference["status"];
type TypeFilter = "All" | Inference["inference_type"];

function formatDate(value?: string | null) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "—";
  return parsed.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function formatTime(value?: string | null) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "—";
  return parsed.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getDurationSeconds(start?: string | null, end?: string | null) {
  if (!start || !end) return null;
  const startMs = new Date(start).getTime();
  const endMs = new Date(end).getTime();
  if (Number.isNaN(startMs) || Number.isNaN(endMs) || endMs < startMs)
    return null;
  return (endMs - startMs) / 1000;
}

function StatusBadge({ status }: { status: Inference["status"] }) {
  const config = {
    PENDING: {
      label: "Pending",
      classes: "bg-gray-100 text-gray-500",
      Icon: Clock3,
    },
    PROCESSING: {
      label: "Processing",
      classes: "bg-blue-50 text-blue-700",
      Icon: Loader2,
    },
    COMPLETED: {
      label: "Completed",
      classes: "bg-green-50 text-green-700",
      Icon: CheckCircle2,
    },
    FAILED: {
      label: "Failed",
      classes: "bg-red-50 text-red-700",
      Icon: XCircle,
    },
  } as const;

  const { label, classes, Icon } = config[status];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[8px] font-semibold ${classes}`}
    >
      <Icon
        size={9}
        className={status === "PROCESSING" ? "animate-spin" : ""}
      />
      {label}
    </span>
  );
}

const PAGE_SIZE = 10;

export function InferenceHistory() {
  const navigate = useNavigate();

  const [inferences, setInferences] = useState<Inference[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<StatusFilter>("All");
  const [type, setType] = useState<TypeFilter>("All");
  const [showFilters, setShowFilters] = useState(false);
  const [page, setPage] = useState(1);

  const loadInferences = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getInferences();
      setInferences(data);
    } catch {
      setError("Unable to load inference history. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadInferences();
  }, []);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    return inferences.filter((item) => {
      const matchesSearch =
        !query ||
        item.patient_name.toLowerCase().includes(query) ||
        (item.patient_id ?? "").toLowerCase().includes(query) ||
        item.predicted_class.toLowerCase().includes(query) ||
        `inf-${item.id}`.includes(query);

      const matchesStatus = status === "All" || item.status === status;
      const matchesType = type === "All" || item.inference_type === type;

      return matchesSearch && matchesStatus && matchesType;
    });
  }, [inferences, search, status, type]);

  useEffect(() => {
    setPage(1);
  }, [search, status, type]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const stats = useMemo(() => {
    const total = inferences.length;
    const completed = inferences.filter((i) => i.status === "COMPLETED").length;
    const inProgress = inferences.filter(
      (i) => i.status === "PENDING" || i.status === "PROCESSING",
    ).length;
    const durations = inferences
      .map((i) => getDurationSeconds(i.started_at, i.completed_at))
      .filter((d): d is number => d !== null);
    const avgDuration =
      durations.length > 0
        ? durations.reduce((a, b) => a + b, 0) / durations.length
        : null;

    return { total, completed, inProgress, avgDuration };
  }, [inferences]);

  if (isLoading) {
    return (
      <div className="flex min-h-100 items-center justify-center">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-gray-900" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-blue-600">
            <BrainCircuit size={11} />
            AI Inference Engine
          </div>
          <h1 className="mt-2 text-[25px] font-semibold tracking-tight text-gray-950 sm:text-[29px]">
            Inference history
          </h1>
          <p className="mt-1 max-w-xl text-[10px] leading-5 text-gray-400">
            Review previous model runs, predictions, confidence scores, and
            status.
          </p>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={loadInferences}
            className="flex h-9 items-center justify-center gap-2 rounded-lg border border-gray-200 px-3.5 text-[10px] font-medium text-gray-500 hover:bg-gray-50"
          >
            <RefreshCw size={13} />
            Refresh
          </button>
          <button
            type="button"
            onClick={() => navigate("/app/inference")}
            className="flex h-9 items-center justify-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[10px] font-semibold text-white hover:bg-gray-800"
          >
            <Activity size={13} />
            New inference
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-[9px] text-red-700">
          {error}
        </div>
      )}

      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Total inferences
          </div>
          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            {stats.total}
          </div>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Completed
          </div>
          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            {stats.completed}
          </div>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            In progress
          </div>
          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            {stats.inProgress}
          </div>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Avg. processing
          </div>
          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            {stats.avgDuration !== null
              ? `${stats.avgDuration.toFixed(2)}s`
              : "—"}
          </div>
        </div>
      </div>

      <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
        <div className="flex flex-col gap-3 border-b border-gray-100 p-4 lg:flex-row lg:items-center">
          <div className="relative min-w-0 flex-1 lg:max-w-95">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300"
            />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search patient, prediction, INF-ID..."
              className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 pl-9 pr-3 text-[9px] text-gray-700 outline-none placeholder:text-gray-300 focus:border-gray-300 focus:bg-white"
            />
          </div>

          <button
            type="button"
            onClick={() => setShowFilters((v) => !v)}
            className={[
              "flex h-9 items-center gap-2 rounded-lg border px-3 text-[9px] font-medium",
              showFilters || status !== "All" || type !== "All"
                ? "border-gray-300 bg-gray-50 text-gray-900"
                : "border-gray-200 text-gray-500 hover:bg-gray-50",
            ].join(" ")}
          >
            Filters
          </button>
        </div>

        {showFilters && (
          <div className="grid gap-4 border-b border-gray-100 bg-gray-50/50 p-4 sm:grid-cols-2">
            <div>
              <div className="mb-2 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                Status
              </div>
              <div className="flex flex-wrap gap-1.5">
                {(
                  [
                    "All",
                    "PENDING",
                    "PROCESSING",
                    "COMPLETED",
                    "FAILED",
                  ] as const
                ).map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setStatus(item)}
                    className={[
                      "rounded-md px-2.5 py-1.5 text-[8px] font-medium",
                      status === item
                        ? "bg-gray-900 text-white"
                        : "bg-white text-gray-500 ring-1 ring-gray-200 hover:bg-gray-50",
                    ].join(" ")}
                  >
                    {item === "All"
                      ? "All"
                      : item.charAt(0) + item.slice(1).toLowerCase()}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-2 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                Type
              </div>
              <div className="flex flex-wrap gap-1.5">
                {(["All", "SYMPTOMS", "IMAGE"] as const).map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setType(item)}
                    className={[
                      "rounded-md px-2.5 py-1.5 text-[8px] font-medium",
                      type === item
                        ? "bg-gray-900 text-white"
                        : "bg-white text-gray-500 ring-1 ring-gray-200 hover:bg-gray-50",
                    ].join(" ")}
                  >
                    {item === "All"
                      ? "All"
                      : item === "SYMPTOMS"
                        ? "Symptoms"
                        : "Image"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="hidden overflow-x-auto lg:block">
          <table className="w-full min-w-225">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50 text-left">
                <th className="px-5 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Patient
                </th>
                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Type
                </th>
                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Prediction
                </th>
                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Confidence
                </th>
                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Status
                </th>
                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Date
                </th>
                <th className="w-10 px-3 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {paginated.map((item) => (
                <tr
                  key={item.id}
                  onClick={() => navigate(`/app/inference/${item.id}`)}
                  className="group cursor-pointer transition hover:bg-gray-50/70"
                >
                  <td className="px-5 py-3.5">
                    <div className="text-[10px] font-semibold text-gray-800">
                      {item.patient_name}
                    </div>
                    {item.patient_id && (
                      <div className="mt-1 font-mono text-[8px] text-gray-400">
                        {item.patient_id}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3.5 text-[9px] text-gray-600">
                    {item.inference_type === "IMAGE" ? "Image" : "Symptoms"}
                  </td>
                  <td className="px-4 py-3.5">
                    <div className="text-[9px] font-semibold text-gray-800">
                      {item.predicted_class || "—"}
                    </div>
                    <div className="mt-1 font-mono text-[8px] text-gray-400">
                      INF-{item.id}
                    </div>
                  </td>
                  <td className="px-4 py-3.5">
                    {item.confidence == null ? (
                      <span className="text-[9px] text-gray-300">—</span>
                    ) : (
                      <div className="flex items-center gap-2">
                        <div className="h-1 w-14 overflow-hidden rounded-full bg-gray-100">
                          <div
                            className="h-full rounded-full bg-gray-800"
                            style={{ width: `${item.confidence * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-[8px] text-gray-500">
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3.5">
                    <StatusBadge status={item.status} />
                  </td>
                  <td className="px-4 py-3.5">
                    <div className="text-[9px] text-gray-600">
                      {formatDate(item.created_at)}
                    </div>
                    <div className="mt-1 text-[8px] text-gray-400">
                      {formatTime(item.created_at)}
                    </div>
                  </td>
                  <td className="px-3 py-3.5">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/app/inference/${item.id}`);
                      }}
                      className="flex h-7 w-7 items-center justify-center rounded-md text-gray-300 opacity-0 transition group-hover:opacity-100 hover:bg-gray-100 hover:text-gray-700"
                      aria-label="View inference"
                    >
                      <Eye size={13} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="divide-y divide-gray-100 lg:hidden">
          {paginated.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => navigate(`/app/inference/${item.id}`)}
              className="block w-full p-4 text-left transition hover:bg-gray-50"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-[10px] font-semibold text-gray-800">
                    {item.patient_name}
                  </div>
                  {item.patient_id && (
                    <div className="mt-1 font-mono text-[8px] text-gray-400">
                      {item.patient_id}
                    </div>
                  )}
                </div>
                <StatusBadge status={item.status} />
              </div>

              <div className="mt-3 text-[9px] font-medium text-gray-700">
                {item.predicted_class || "—"}
              </div>

              <div className="mt-3 grid grid-cols-3 gap-3 border-t border-gray-100 pt-3">
                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Confidence
                  </div>
                  <div className="mt-1 font-mono text-[9px] text-gray-700">
                    {item.confidence == null
                      ? "—"
                      : `${(item.confidence * 100).toFixed(1)}%`}
                  </div>
                </div>
                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Date
                  </div>
                  <div className="mt-1 text-[8px] text-gray-700">
                    {formatDate(item.created_at)}
                  </div>
                </div>
                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Type
                  </div>
                  <div className="mt-1 text-[8px] text-gray-700">
                    {item.inference_type === "IMAGE" ? "Image" : "Symptoms"}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>

        {filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center px-5 py-16 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gray-100">
              <Search size={16} className="text-gray-400" />
            </div>
            <div className="mt-4 text-[11px] font-semibold text-gray-700">
              No inferences found
            </div>
            <p className="mt-1 max-w-xs text-[9px] leading-4 text-gray-400">
              Try changing your search or removing one of the active filters.
            </p>
          </div>
        )}

        {filtered.length > 0 && (
          <div className="flex flex-col gap-2 border-t border-gray-100 px-5 py-3 sm:flex-row sm:items-center sm:justify-between">
            <span className="text-[8px] text-gray-400">
              Showing {(page - 1) * PAGE_SIZE + 1}-
              {Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length}
            </span>

            {totalPages > 1 && (
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="h-7 rounded-md border border-gray-200 px-2.5 text-[8px] text-gray-500 hover:enabled:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300"
                >
                  Previous
                </button>
                <span className="px-2 text-[8px] text-gray-500">
                  Page {page} of {totalPages}
                </span>
                <button
                  type="button"
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  className="h-7 rounded-md border border-gray-200 px-2.5 text-[8px] text-gray-500 hover:enabled:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
