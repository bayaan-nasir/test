import {
  Activity,
  ArrowUpDown,
  BrainCircuit,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Eye,
  Filter,
  Image as ImageIcon,
  Layers3,
  Search,
  Table2,
  XCircle,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

type Status = "Completed" | "Review required" | "Failed";
type Modality = "Multimodal" | "Image" | "Tabular";

interface Inference {
  id: string;
  patientId: string;
  patient: string;
  model: string;
  modality: Modality;
  result: string;
  confidence: number;
  status: Status;
  date: string;
  time: string;
  duration: string;
}

const inferences: Inference[] = [
  {
    id: "INF-7F82A1",
    patientId: "PT-10482",
    patient: "Ama Mensah",
    model: "Respiratory Assessment",
    modality: "Multimodal",
    result: "Pneumonia",
    confidence: 94.7,
    status: "Review required",
    date: "29 Aug 2026",
    time: "14:32",
    duration: "2.84s",
  },
  {
    id: "INF-6D91C4",
    patientId: "PT-10371",
    patient: "Kwame Asante",
    model: "Chest X-Ray Classifier",
    modality: "Image",
    result: "Normal",
    confidence: 96.2,
    status: "Completed",
    date: "29 Aug 2026",
    time: "13:18",
    duration: "1.42s",
  },
  {
    id: "INF-5A27E9",
    patientId: "PT-10298",
    patient: "Abena Owusu",
    model: "Cardiovascular Risk",
    modality: "Tabular",
    result: "Low risk",
    confidence: 88.4,
    status: "Completed",
    date: "29 Aug 2026",
    time: "11:46",
    duration: "0.82s",
  },
  {
    id: "INF-4C83B2",
    patientId: "PT-10411",
    patient: "Kofi Boateng",
    model: "Respiratory Assessment",
    modality: "Multimodal",
    result: "Other abnormality",
    confidence: 79.1,
    status: "Review required",
    date: "28 Aug 2026",
    time: "16:07",
    duration: "3.16s",
  },
  {
    id: "INF-3B61D8",
    patientId: "PT-10187",
    patient: "Akosua Addo",
    model: "Chest X-Ray Classifier",
    modality: "Image",
    result: "Abnormal finding",
    confidence: 91.4,
    status: "Completed",
    date: "28 Aug 2026",
    time: "12:34",
    duration: "1.63s",
  },
  {
    id: "INF-2E48F5",
    patientId: "PT-10042",
    patient: "Yaw Mensima",
    model: "Cardiovascular Risk",
    modality: "Tabular",
    result: "Moderate risk",
    confidence: 73.8,
    status: "Completed",
    date: "27 Aug 2026",
    time: "15:21",
    duration: "0.91s",
  },
  {
    id: "INF-1A39C7",
    patientId: "PT-10304",
    patient: "Esi Ofori",
    model: "Respiratory Assessment",
    modality: "Multimodal",
    result: "Pneumonia",
    confidence: 89.6,
    status: "Completed",
    date: "27 Aug 2026",
    time: "09:52",
    duration: "2.71s",
  },
  {
    id: "INF-0D72A4",
    patientId: "PT-10221",
    patient: "Nana Adjei",
    model: "Chest X-Ray Classifier",
    modality: "Image",
    result: "Processing failed",
    confidence: 0,
    status: "Failed",
    date: "26 Aug 2026",
    time: "17:03",
    duration: "0.38s",
  },
];

const modalityIcons = {
  Multimodal: Layers3,
  Image: ImageIcon,
  Tabular: Table2,
};

function StatusBadge({ status }: { status: Status }) {
  if (status === "Completed") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-green-50 px-2 py-1 text-[8px] font-semibold text-green-700">
        <CheckCircle2 size={9} />
        Completed
      </span>
    );
  }

  if (status === "Review required") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-amber-50 px-2 py-1 text-[8px] font-semibold text-amber-700">
        <Clock3 size={9} />
        Review required
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 rounded-md bg-red-50 px-2 py-1 text-[8px] font-semibold text-red-700">
      <XCircle size={9} />
      Failed
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

export function InferenceHistory() {
  const navigate = useNavigate();

  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<"All" | Status>("All");
  const [modality, setModality] = useState<"All" | Modality>("All");
  const [showFilters, setShowFilters] = useState(false);
  const [sortDescending, setSortDescending] = useState(true);

  const filteredInferences = useMemo(() => {
    const query = search.trim().toLowerCase();

    const filtered = inferences.filter((item) => {
      const matchesSearch =
        !query ||
        item.patient.toLowerCase().includes(query) ||
        item.patientId.toLowerCase().includes(query) ||
        item.model.toLowerCase().includes(query) ||
        item.result.toLowerCase().includes(query) ||
        item.id.toLowerCase().includes(query);

      const matchesStatus =
        status === "All" || item.status === status;

      const matchesModality =
        modality === "All" || item.modality === modality;

      return matchesSearch && matchesStatus && matchesModality;
    });

    return sortDescending ? filtered : [...filtered].reverse();
  }, [search, status, modality, sortDescending]);

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Header */}
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
            Review previous model runs, predictions, confidence scores,
            and clinical review status.
          </p>
        </div>

        <button
          type="button"
          onClick={() => navigate("/app/inference")}
          className="flex h-9 items-center justify-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[10px] font-semibold text-white hover:bg-gray-800"
        >
          <Activity size={13} />
          New inference
        </button>
      </div>

      {/* Stats */}
      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Total inferences
          </div>

          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            1,284
          </div>

          <div className="mt-1 text-[8px] text-gray-400">
            Last 30 days
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Completed
          </div>

          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            1,247
          </div>

          <div className="mt-1 text-[8px] text-green-600">
            97.1% success rate
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Awaiting review
          </div>

          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            23
          </div>

          <div className="mt-1 text-[8px] text-amber-600">
            Requires attention
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Avg. processing
          </div>

          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            1.84s
          </div>

          <div className="mt-1 text-[8px] text-gray-400">
            Across all models
          </div>
        </div>
      </div>

      {/* Table */}
      <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
        {/* Toolbar */}
        <div className="flex flex-col gap-3 border-b border-gray-100 p-4 lg:flex-row lg:items-center">
          <div className="relative min-w-0 flex-1 lg:max-w-95">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300"
            />

            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search patient, model, result..."
              className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 pl-9 pr-8 text-[9px] text-gray-700 outline-none placeholder:text-gray-300 focus:border-gray-300 focus:bg-white"
            />

            {search && (
              <button
                type="button"
                onClick={() => setSearch("")}
                className="absolute right-2 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded text-gray-400 hover:bg-gray-100"
              >
                <XCircle size={11} />
              </button>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setShowFilters((value) => !value)}
              className={[
                "flex h-9 items-center gap-2 rounded-lg border px-3 text-[9px] font-medium",
                showFilters ||
                  status !== "All" ||
                  modality !== "All"
                  ? "border-gray-300 bg-gray-50 text-gray-900"
                  : "border-gray-200 text-gray-500 hover:bg-gray-50",
              ].join(" ")}
            >
              <Filter size={12} />
              Filters

              {(status !== "All" || modality !== "All") && (
                <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-gray-900 px-1 text-[7px] text-white">
                  {(status !== "All" ? 1 : 0) +
                    (modality !== "All" ? 1 : 0)}
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() => setSortDescending((value) => !value)}
              className="flex h-9 items-center gap-2 rounded-lg border border-gray-200 px-3 text-[9px] font-medium text-gray-500 hover:bg-gray-50"
            >
              <ArrowUpDown size={12} />
              Date
            </button>

            <button
              type="button"
              className="flex h-9 items-center gap-2 rounded-lg border border-gray-200 px-3 text-[9px] font-medium text-gray-500 hover:bg-gray-50"
            >
              <CalendarDays size={12} />
              Date range
              <ChevronDown size={11} />
            </button>
          </div>
        </div>

        {/* Filters */}
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
                    "Completed",
                    "Review required",
                    "Failed",
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
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-2 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                Modality
              </div>

              <div className="flex flex-wrap gap-1.5">
                {(
                  ["All", "Multimodal", "Image", "Tabular"] as const
                ).map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setModality(item)}
                    className={[
                      "rounded-md px-2.5 py-1.5 text-[8px] font-medium",
                      modality === item
                        ? "bg-gray-900 text-white"
                        : "bg-white text-gray-500 ring-1 ring-gray-200 hover:bg-gray-50",
                    ].join(" ")}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Desktop table */}
        <div className="hidden overflow-x-auto lg:block">
          <table className="w-full min-w-225">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50 text-left">
                <th className="px-5 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Patient
                </th>

                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Model
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
              {filteredInferences.map((item) => (
                <tr
                  key={item.id}
                  onClick={() =>
                    navigate(`/app/inference/${item.id}`)
                  }
                  className="group cursor-pointer transition hover:bg-gray-50/70"
                >
                  <td className="px-5 py-3.5">
                    <div className="text-[10px] font-semibold text-gray-800">
                      {item.patient}
                    </div>

                    <div className="mt-1 font-mono text-[8px] text-gray-400">
                      {item.patientId}
                    </div>
                  </td>

                  <td className="px-4 py-3.5">
                    <div className="text-[9px] font-medium text-gray-700">
                      {item.model}
                    </div>

                    <div className="mt-1">
                      <ModalityBadge modality={item.modality} />
                    </div>
                  </td>

                  <td className="px-4 py-3.5">
                    <div className="text-[9px] font-semibold text-gray-800">
                      {item.result}
                    </div>

                    <div className="mt-1 font-mono text-[8px] text-gray-400">
                      {item.id}
                    </div>
                  </td>

                  <td className="px-4 py-3.5">
                    {item.status === "Failed" ? (
                      <span className="text-[9px] text-gray-300">
                        —
                      </span>
                    ) : (
                      <div className="flex items-center gap-2">
                        <div className="h-1 w-14 overflow-hidden rounded-full bg-gray-100">
                          <div
                            className="h-full rounded-full bg-gray-800"
                            style={{
                              width: `${item.confidence}%`,
                            }}
                          />
                        </div>

                        <span className="font-mono text-[8px] text-gray-500">
                          {item.confidence.toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </td>

                  <td className="px-4 py-3.5">
                    <StatusBadge status={item.status} />
                  </td>

                  <td className="px-4 py-3.5">
                    <div className="text-[9px] text-gray-600">
                      {item.date}
                    </div>

                    <div className="mt-1 text-[8px] text-gray-400">
                      {item.time} · {item.duration}
                    </div>
                  </td>

                  <td className="px-3 py-3.5">
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
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

        {/* Mobile cards */}
        <div className="divide-y divide-gray-100 lg:hidden">
          {filteredInferences.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() =>
                navigate(`/app/inference/${item.id}`)
              }
              className="block w-full p-4 text-left transition hover:bg-gray-50"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-[10px] font-semibold text-gray-800">
                    {item.patient}
                  </div>

                  <div className="mt-1 font-mono text-[8px] text-gray-400">
                    {item.patientId}
                  </div>
                </div>

                <StatusBadge status={item.status} />
              </div>

              <div className="mt-4 flex items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gray-100">
                  {(() => {
                    const Icon = modalityIcons[item.modality];

                    return (
                      <Icon size={12} className="text-gray-500" />
                    );
                  })()}
                </div>

                <div>
                  <div className="text-[9px] font-medium text-gray-700">
                    {item.model}
                  </div>

                  <div className="mt-0.5 text-[8px] text-gray-400">
                    {item.result}
                  </div>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-3 border-t border-gray-100 pt-3">
                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Confidence
                  </div>

                  <div className="mt-1 font-mono text-[9px] text-gray-700">
                    {item.status === "Failed"
                      ? "—"
                      : `${item.confidence.toFixed(1)}%`}
                  </div>
                </div>

                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Date
                  </div>

                  <div className="mt-1 text-[8px] text-gray-700">
                    {item.date}
                  </div>
                </div>

                <div>
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Duration
                  </div>

                  <div className="mt-1 font-mono text-[8px] text-gray-700">
                    {item.duration}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Empty state */}
        {filteredInferences.length === 0 && (
          <div className="flex flex-col items-center justify-center px-5 py-16 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gray-100">
              <Search size={16} className="text-gray-400" />
            </div>

            <div className="mt-4 text-[11px] font-semibold text-gray-700">
              No inferences found
            </div>

            <p className="mt-1 max-w-xs text-[9px] leading-4 text-gray-400">
              Try changing your search or removing one of the active
              filters.
            </p>
          </div>
        )}

        {/* Footer */}
        {filteredInferences.length > 0 && (
          <div className="flex flex-col gap-2 border-t border-gray-100 px-5 py-3 sm:flex-row sm:items-center sm:justify-between">
            <span className="text-[8px] text-gray-400">
              Showing {filteredInferences.length} of 1,284 inferences
            </span>

            <div className="flex items-center gap-1">
              <button
                type="button"
                disabled
                className="h-7 rounded-md border border-gray-200 px-2.5 text-[8px] text-gray-300"
              >
                Previous
              </button>

              <button
                type="button"
                className="h-7 rounded-md bg-gray-900 px-2.5 text-[8px] font-medium text-white"
              >
                1
              </button>

              <button
                type="button"
                className="h-7 rounded-md border border-gray-200 px-2.5 text-[8px] text-gray-500 hover:bg-gray-50"
              >
                2
              </button>

              <button
                type="button"
                className="h-7 rounded-md border border-gray-200 px-2.5 text-[8px] text-gray-500 hover:bg-gray-50"
              >
                3
              </button>

              <button
                type="button"
                className="h-7 rounded-md border border-gray-200 px-2.5 text-[8px] text-gray-500 hover:bg-gray-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}