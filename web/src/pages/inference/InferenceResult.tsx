import {
  AlertTriangle,
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  FileText,
  Info,
  Loader2,
  ShieldCheck,
  Sparkles,
  UserRound,
  XCircle,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  getInference,
  type Inference,
  type InferenceModelResult,
} from "../../api/inference";

const POLL_INTERVAL_MS = 3000;

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

function getMediaUrl(path?: string | null) {
  if (!path) return null;

  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  const baseUrl = "http://localhost:8001";

  return `${baseUrl.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}

function formatDuration(
  start?: string | null,
  end?: string | null,
) {
  if (!start || !end) return null;

  const startMs = new Date(start).getTime();
  const endMs = new Date(end).getTime();

  if (
    Number.isNaN(startMs) ||
    Number.isNaN(endMs) ||
    endMs < startMs
  ) {
    return null;
  }

  return `${((endMs - startMs) / 1000).toFixed(2)} seconds`;
}

function StatusBadge({
  status,
}: {
  status: Inference["status"];
}) {
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
      className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-[8px] font-semibold ${classes}`}
    >
      <Icon
        size={10}
        className={
          status === "PROCESSING" ? "animate-spin" : ""
        }
      />
      {label}
    </span>
  );
}

function TriageBadge({
  triage,
}: {
  triage: string;
}) {
  const styles: Record<string, string> = {
    high: "bg-red-50 text-red-700",
    medium: "bg-amber-50 text-amber-700",
    low: "bg-gray-100 text-gray-600",
  };

  const label = triage
    ? `${triage.charAt(0).toUpperCase()}${triage.slice(1)} priority`
    : "Unknown";

  return (
    <span
      className={`rounded-md px-2 py-1 text-[8px] font-semibold ${styles[triage] ??
        "bg-gray-100 text-gray-600"
        }`}
    >
      {label}
    </span>
  );
}

function ConfidenceBar({
  value,
  selected = false,
}: {
  value: number;
  selected?: boolean;
}) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
      <div
        className={[
          "h-full rounded-full transition-all",
          selected ? "bg-gray-900" : "bg-gray-300",
        ].join(" ")}
        style={{
          width: `${Math.min(
            100,
            Math.max(0, value),
          )}%`,
        }}
      />
    </div>
  );
}

function MarkdownContent({
  content,
}: {
  content: string;
}) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => (
          <p className="text-[9px] leading-5 text-gray-500">
            {children}
          </p>
        ),

        strong: ({ children }) => (
          <strong className="font-semibold text-gray-700">
            {children}
          </strong>
        ),

        em: ({ children }) => (
          <em className="italic">{children}</em>
        ),

        ul: ({ children }) => (
          <ul className="mt-2 list-disc space-y-1 pl-4 text-[9px] leading-5 text-gray-500">
            {children}
          </ul>
        ),

        ol: ({ children }) => (
          <ol className="mt-2 list-decimal space-y-1 pl-4 text-[9px] leading-5 text-gray-500">
            {children}
          </ol>
        ),

        li: ({ children }) => (
          <li className="pl-0.5">{children}</li>
        ),

        h1: ({ children }) => (
          <h1 className="mt-4 text-[14px] font-semibold text-gray-900 first:mt-0">
            {children}
          </h1>
        ),

        h2: ({ children }) => (
          <h2 className="mt-4 text-[13px] font-semibold text-gray-900 first:mt-0">
            {children}
          </h2>
        ),

        h3: ({ children }) => (
          <h3 className="mt-4 text-[10px] font-semibold text-gray-800 first:mt-0">
            {children}
          </h3>
        ),

        hr: () => (
          <hr className="my-4 border-gray-200" />
        ),

        blockquote: ({ children }) => (
          <blockquote className="my-2 border-l-2 border-gray-300 pl-3 text-[9px] italic text-gray-500">
            {children}
          </blockquote>
        ),

        code: ({ children }) => (
          <code className="rounded bg-gray-100 px-1 py-0.5 font-mono text-[8px] text-gray-700">
            {children}
          </code>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

export function InferenceResult() {
  const navigate = useNavigate();
  const { inferenceId } = useParams();

  const [inference, setInference] =
    useState<Inference | null>(null);

  const [isLoading, setIsLoading] =
    useState(true);

  const [error, setError] =
    useState<string | null>(null);

  const pollRef =
    useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const numericId = Number(inferenceId);

    if (!inferenceId || Number.isNaN(numericId)) {
      setError("Invalid inference ID.");
      setIsLoading(false);
      return;
    }

    let active = true;

    async function load(showSpinner: boolean) {
      if (showSpinner) {
        setIsLoading(true);
      }

      try {
        const data = await getInference(numericId);

        if (!active) return;

        setInference(data);
        setError(null);

        if (
          data.status === "PENDING" ||
          data.status === "PROCESSING"
        ) {
          if (!pollRef.current) {
            pollRef.current = setInterval(
              () => load(false),
              POLL_INTERVAL_MS,
            );
          }
        } else if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch (err) {
        if (!active) return;

        const maybeError =
          err as {
            response?: {
              status?: number;
            };
          };

        if (maybeError.response?.status === 404) {
          setError(
            "This inference could not be found.",
          );
        } else if (
          maybeError.response?.status === 403
        ) {
          setError(
            "You do not have access to this inference.",
          );
        } else {
          setError(
            "Unable to load this inference. Please try again.",
          );
        }
      } finally {
        if (active) {
          setIsLoading(false);
        }
      }
    }

    load(true);

    return () => {
      active = false;

      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [inferenceId]);

  if (isLoading) {
    return (
      <div className="flex min-h-100 items-center justify-center">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-gray-900" />
      </div>
    );
  }

  if (error || !inference) {
    return (
      <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
        <button
          type="button"
          onClick={() =>
            navigate("/app/inference/history")
          }
          className="mb-5 flex items-center gap-2 text-[11px] font-medium text-gray-500 hover:text-gray-900"
        >
          <ArrowLeft size={14} />
          Back to inference history
        </button>

        <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
          <p className="text-[11px] font-semibold text-gray-700">
            {error ??
              "This inference could not be found."}
          </p>
        </div>
      </div>
    );
  }

  const results =
    inference.response_payload?.results ?? [];

  const topResult =
    inference.response_payload?.top_result;

  const symptomEntries = Object.entries(
    inference.request_payload?.symptoms ?? {},
  );

  const duration = formatDuration(
    inference.started_at,
    inference.completed_at,
  );

  const heatmapResults = results.filter(
    (result) =>
      Boolean(
        result.explainability?.heatmap_url,
      ),
  );

  const topHeatmapUrl = getMediaUrl(
    topResult?.explainability?.heatmap_url,
  );

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Back button */}
      <button
        type="button"
        onClick={() =>
          inference.patient_id
            ? navigate(
              `/app/patients/${inference.patient_id}`,
            )
            : navigate(
              "/app/inference/history",
            )
        }
        className="mb-5 flex items-center gap-2 text-[11px] font-medium text-gray-500 hover:text-gray-900"
      >
        <ArrowLeft size={14} />

        {inference.patient_id
          ? "Back to patient"
          : "Back to inference history"}
      </button>

      {/* Header */}
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge
              status={inference.status}
            />

            <span className="font-mono text-[8px] text-gray-400">
              INF-{inference.id}
            </span>
          </div>

          <h1 className="mt-2 text-[25px] font-semibold tracking-tight text-gray-950 sm:text-[29px]">
            AI inference result
          </h1>

          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] text-gray-400">
            <span>
              {inference.inference_type ===
                "IMAGE"
                ? "Image-based"
                : "Symptoms-based"}
            </span>

            <span className="text-gray-200">
              •
            </span>

            <span>
              {formatDateTime(
                inference.created_at,
              )}
            </span>
          </div>
        </div>

        <button
          type="button"
          onClick={() =>
            navigate(
              inference.patient_id
                ? `/app/inference?patient=${inference.patient_id}`
                : "/app/inference",
            )
          }
          className="flex h-9 items-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[10px] font-semibold text-white hover:bg-gray-800"
        >
          <Sparkles size={13} />
          New inference
        </button>
      </div>

      {/* Patient */}
      <section className="mb-5 flex flex-col gap-3 rounded-xl border border-gray-200 bg-white px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gray-100 text-gray-400">
            <UserRound size={14} />
          </div>

          <div>
            <div className="text-[11px] font-semibold text-gray-900">
              {inference.patient_name}
            </div>

            {inference.patient_id && (
              <div className="mt-0.5 font-mono text-[8px] text-gray-400">
                {inference.patient_id}
              </div>
            )}
          </div>
        </div>

        {inference.patient_id && (
          <button
            type="button"
            onClick={() =>
              navigate(
                `/app/patients/${inference.patient_id}`,
              )
            }
            className="flex items-center gap-1.5 text-[9px] font-semibold text-gray-500 hover:text-gray-900"
          >
            <UserRound size={12} />
            View patient record
          </button>
        )}
      </section>

      {/* Processing */}
      {(inference.status === "PENDING" ||
        inference.status === "PROCESSING") && (
          <section className="mb-5 flex items-center gap-3 rounded-xl border border-blue-100 bg-blue-50/50 p-5">
            <Loader2
              size={16}
              className="animate-spin text-blue-600"
            />

            <div>
              <div className="text-[10px] font-semibold text-blue-900">
                {inference.status === "PENDING"
                  ? "Waiting to start"
                  : "Processing"}
              </div>

              <p className="mt-0.5 text-[8px] text-blue-700">
                This page will update automatically
                once the result is ready.
              </p>
            </div>
          </section>
        )}

      {/* Failed */}
      {inference.status === "FAILED" && (
        <section className="mb-5 flex items-start gap-3 rounded-xl border border-red-100 bg-red-50/50 p-5">
          <AlertTriangle
            size={16}
            className="mt-0.5 shrink-0 text-red-600"
          />

          <div>
            <div className="text-[10px] font-semibold text-red-900">
              Inference failed
            </div>

            <p className="mt-1 text-[8px] leading-4 text-red-700">
              {inference.error_message ||
                "The inference could not be completed."}
            </p>
          </div>
        </section>
      )}

      {/* Completed */}
      {inference.status === "COMPLETED" && (
        <div className="grid gap-5 lg:grid-cols-[1fr_350px]">
          <main className="space-y-5">

            {/* Primary assessment */}
            <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
              <div className="border-b border-gray-100 px-5 py-4">
                <div className="flex items-center gap-2">
                  <BrainCircuit
                    size={15}
                    className="text-gray-400"
                  />

                  <h2 className="text-[12px] font-semibold text-gray-950">
                    Primary assessment
                  </h2>
                </div>
              </div>

              <div className="p-5 sm:p-6">
                <div className="grid gap-6 md:grid-cols-[1fr_180px] md:items-center">
                  <div>
                    <div className="text-[9px] font-medium uppercase tracking-[0.12em] text-gray-400">
                      Predicted finding
                    </div>

                    <div className="mt-2 text-[28px] font-semibold tracking-tight text-gray-950">
                      {inference.predicted_class ||
                        "No result"}
                    </div>

                    <div className="mt-2 flex items-center gap-2">
                      <TriageBadge
                        triage={
                          inference.overall_triage
                        }
                      />

                      {topResult && (
                        <span className="text-[9px] text-gray-400">
                          {topResult.disease} ·{" "}
                          {topResult.model_used}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="text-center md:text-right">
                    <div className="text-[29px] font-semibold tracking-tight text-gray-950">
                      {inference.confidence !=
                        null
                        ? `${(
                          inference.confidence *
                          100
                        ).toFixed(1)}%`
                        : "—"}
                    </div>

                    <div className="text-[8px] uppercase tracking-wide text-gray-400">
                      confidence
                    </div>
                  </div>
                </div>

                {/* Clinical summary */}
                {inference.clinical_summary && (
                  <div className="mt-7 rounded-lg bg-gray-50 p-4">
                    <div className="flex items-start gap-2.5">
                      <Info
                        size={13}
                        className="mt-0.5 shrink-0 text-gray-400"
                      />

                      <div className="min-w-0 flex-1">
                        <div className="text-[9px] font-semibold text-gray-700">
                          Clinical summary
                        </div>

                        <div className="mt-1">
                          <MarkdownContent
                            content={
                              inference.clinical_summary
                            }
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </section>

            {/* Primary Grad-CAM */}
            {topHeatmapUrl && (
              <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
                <div className="border-b border-gray-100 px-5 py-4">
                  <div className="flex items-center gap-2">
                    <BrainCircuit
                      size={14}
                      className="text-gray-400"
                    />

                    <div>
                      <h2 className="text-[12px] font-semibold text-gray-950">
                        Primary model explainability
                      </h2>

                      <p className="mt-0.5 text-[9px] text-gray-400">
                        Grad-CAM visualisation for the
                        primary prediction
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-5">
                  <div className="grid gap-5 md:grid-cols-[minmax(0,1fr)_220px] md:items-center">
                    <div className="overflow-hidden rounded-lg border border-gray-200 bg-gray-950">
                      <img
                        src={topHeatmapUrl}
                        alt={`Grad-CAM heatmap for ${topResult?.disease ?? "primary"} prediction`}
                        className="block max-h-130 w-full object-contain"
                        loading="lazy"
                      />
                    </div>

                    <div>
                      <div className="text-[8px] font-medium uppercase tracking-[0.12em] text-gray-400">
                        Primary result
                      </div>

                      <div className="mt-2 text-[18px] font-semibold tracking-tight text-gray-950">
                        {topResult?.predicted_class ??
                          inference.predicted_class ??
                          "Unknown"}
                      </div>

                      {topResult && (
                        <>
                          <div className="mt-2">
                            <TriageBadge
                              triage={
                                topResult.triage
                              }
                            />
                          </div>

                          <div className="mt-4 space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-[8px] text-gray-400">
                                Confidence
                              </span>

                              <span className="font-mono text-[9px] font-semibold text-gray-700">
                                {
                                  topResult.confidence_pct
                                }
                              </span>
                            </div>

                            <ConfidenceBar
                              value={
                                topResult.confidence *
                                100
                              }
                              selected
                            />
                          </div>

                          <div className="mt-4">
                            <div className="text-[8px] text-gray-400">
                              Model
                            </div>

                            <div className="mt-1 text-[9px] font-medium text-gray-700">
                              {
                                topResult.model_used
                              }
                            </div>
                          </div>
                        </>
                      )}

                      <div className="mt-4 rounded-lg bg-gray-50 p-3">
                        <p className="text-[8px] leading-4 text-gray-500">
                          Grad-CAM highlights image
                          regions that contributed most
                          strongly to the model's
                          prediction. It is an
                          explainability aid and should not
                          be interpreted as a definitive
                          anatomical diagnosis.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* Model results */}
            {results.length > 0 && (
              <section className="rounded-xl border border-gray-200 bg-white">
                <div className="border-b border-gray-100 px-5 py-4">
                  <h2 className="text-[12px] font-semibold text-gray-950">
                    Model results
                  </h2>

                  <p className="mt-0.5 text-[9px] text-gray-400">
                    Every model the router selected for
                    this submission
                  </p>
                </div>

                <div className="space-y-5 p-5">
                  {results.map(
                    (
                      result: InferenceModelResult,
                      index: number,
                    ) => (
                      <div
                        key={`${result.disease}-${index}`}
                      >
                        <div className="mb-2 flex items-center justify-between gap-3">
                          <div className="flex items-center gap-2">
                            {index === 0 && (
                              <CheckCircle2
                                size={12}
                                className="text-gray-900"
                              />
                            )}

                            <span className="text-[10px] font-medium text-gray-700">
                              {result.disease} —{" "}
                              {
                                result.predicted_class
                              }
                            </span>
                          </div>

                          <span className="font-mono text-[9px] text-gray-500">
                            {
                              result.confidence_pct
                            }
                          </span>
                        </div>

                        <ConfidenceBar
                          value={
                            result.confidence * 100
                          }
                          selected={index === 0}
                        />

                        <div className="mt-1.5 flex items-center gap-2 text-[7px] text-gray-400">
                          <span>
                            {result.model_used}
                          </span>

                          <span>•</span>

                          <TriageBadge
                            triage={result.triage}
                          />
                        </div>
                      </div>
                    ),
                  )}
                </div>
              </section>
            )}

            {/* All Grad-CAM heatmaps */}
            {heatmapResults.length > 0 && (
              <section className="rounded-xl border border-gray-200 bg-white">
                <div className="border-b border-gray-100 px-5 py-4">
                  <div className="flex items-center gap-2">
                    <BrainCircuit
                      size={14}
                      className="text-gray-400"
                    />

                    <div>
                      <h2 className="text-[12px] font-semibold text-gray-950">
                        Model explainability
                      </h2>

                      <p className="mt-0.5 text-[9px] text-gray-400">
                        Grad-CAM heatmaps showing the
                        regions that contributed to each
                        model prediction
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid gap-5 p-5 md:grid-cols-2 xl:grid-cols-3">
                  {heatmapResults.map(
                    (
                      result: InferenceModelResult,
                      index: number,
                    ) => {
                      const heatmapUrl =
                        getMediaUrl(
                          result.explainability
                            ?.heatmap_url,
                        );

                      if (!heatmapUrl) {
                        return null;
                      }

                      return (
                        <div
                          key={`${result.disease}-heatmap-${index}`}
                          className="overflow-hidden rounded-lg border border-gray-200 bg-gray-50"
                        >
                          {/* Heatmap header */}
                          <div className="border-b border-gray-200 bg-white px-4 py-3">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <div className="text-[10px] font-semibold text-gray-900">
                                  {result.disease}
                                </div>

                                <div className="mt-0.5 text-[8px] text-gray-400">
                                  {
                                    result.model_used
                                  }
                                </div>
                              </div>

                              <TriageBadge
                                triage={
                                  result.triage
                                }
                              />
                            </div>

                            <div className="mt-2 flex items-center justify-between gap-3">
                              <span className="text-[8px] uppercase tracking-wide text-gray-400">
                                Prediction
                              </span>

                              <span className="text-right text-[9px] font-semibold text-gray-700">
                                {
                                  result.predicted_class
                                }
                              </span>
                            </div>
                          </div>

                          {/* Heatmap */}
                          <div className="aspect-square bg-gray-950">
                            <img
                              src={heatmapUrl}
                              alt={`Grad-CAM heatmap for ${result.disease} prediction`}
                              className="h-full w-full object-contain"
                              loading="lazy"
                            />
                          </div>

                          {/* Heatmap footer */}
                          <div className="border-t border-gray-200 bg-white px-4 py-3">
                            <div className="flex items-center justify-between">
                              <span className="text-[8px] text-gray-400">
                                Model confidence
                              </span>

                              <span className="font-mono text-[9px] font-semibold text-gray-700">
                                {
                                  result.confidence_pct
                                }
                              </span>
                            </div>

                            <div className="mt-2">
                              <ConfidenceBar
                                value={
                                  result.confidence *
                                  100
                                }
                                selected={
                                  result ===
                                  topResult
                                }
                              />
                            </div>
                          </div>
                        </div>
                      );
                    },
                  )}
                </div>
              </section>
            )}

            {/* Submitted clinical data */}
            {symptomEntries.length > 0 && (
              <section className="rounded-xl border border-gray-200 bg-white">
                <div className="border-b border-gray-100 px-5 py-4">
                  <div className="flex items-center gap-2">
                    <FileText
                      size={14}
                      className="text-gray-400"
                    />

                    <h2 className="text-[12px] font-semibold text-gray-950">
                      Submitted clinical data
                    </h2>
                  </div>
                </div>

                <div className="grid gap-x-6 gap-y-3 p-5 sm:grid-cols-2 md:grid-cols-3">
                  {symptomEntries.map(
                    ([key, value]) => (
                      <div
                        key={key}
                        className="flex items-center justify-between gap-3 border-b border-gray-50 pb-2"
                      >
                        <span className="text-[8px] capitalize text-gray-400">
                          {key.replace(/_/g, " ")}
                        </span>

                        <span className="font-mono text-[9px] font-medium text-gray-700">
                          {String(value)}
                        </span>
                      </div>
                    ),
                  )}
                </div>
              </section>
            )}

            {/* Clinical notes */}
            {inference.request_payload
              ?.clinical_notes && (
                <section className="rounded-xl border border-gray-200 bg-white p-5">
                  <div className="text-[9px] font-semibold text-gray-700">
                    Clinical notes
                  </div>

                  <p className="mt-2 text-[9px] leading-5 text-gray-500">
                    {
                      inference.request_payload
                        .clinical_notes
                    }
                  </p>
                </section>
              )}

            {/* Metadata */}
            <section className="rounded-xl border border-gray-200 bg-white">
              <div className="border-b border-gray-100 px-5 py-4">
                <h2 className="text-[12px] font-semibold text-gray-950">
                  Inference metadata
                </h2>
              </div>

              <div className="grid gap-x-8 gap-y-4 p-5 sm:grid-cols-2 lg:grid-cols-3">
                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Inference ID
                  </div>

                  <div className="mt-1 font-mono text-[9px] font-medium text-gray-700">
                    INF-{inference.id}
                  </div>
                </div>

                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Models run
                  </div>

                  <div className="mt-1 text-[9px] font-medium text-gray-700">
                    {(
                      inference.response_payload
                        ?.models_run ?? []
                    ).join(", ") || "—"}
                  </div>
                </div>

                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Processing time
                  </div>

                  <div className="mt-1 flex items-center gap-1.5 text-[9px] font-medium text-gray-700">
                    <Clock3
                      size={11}
                      className="text-gray-400"
                    />

                    {duration ?? "—"}
                  </div>
                </div>

                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Started
                  </div>

                  <div className="mt-1 text-[9px] font-medium text-gray-700">
                    {formatDateTime(
                      inference.started_at,
                    )}
                  </div>
                </div>

                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Completed
                  </div>

                  <div className="mt-1 text-[9px] font-medium text-gray-700">
                    {formatDateTime(
                      inference.completed_at,
                    )}
                  </div>
                </div>

                <div>
                  <div className="text-[8px] uppercase tracking-wide text-gray-400">
                    Status
                  </div>

                  <div className="mt-1 flex items-center gap-1.5 text-[9px] font-medium text-green-700">
                    <CheckCircle2 size={11} />
                    Completed
                  </div>
                </div>
              </div>
            </section>
          </main>

          {/* Sidebar */}
          <aside className="space-y-5">
            {/* Models skipped */}
            <section className="rounded-xl border border-gray-200 bg-white">
              <div className="border-b border-gray-100 px-5 py-4">
                <h2 className="text-[12px] font-semibold text-gray-950">
                  Models skipped
                </h2>
              </div>

              <div className="p-5">
                {(
                  inference.response_payload
                    ?.models_skipped ?? []
                ).length === 0 ? (
                  <p className="text-[8px] text-gray-400">
                    All applicable models ran.
                  </p>
                ) : (
                  <ul className="space-y-2">
                    {(
                      inference.response_payload
                        ?.models_skipped ?? []
                    ).map((entry) => (
                      <li
                        key={entry}
                        className="text-[8px] leading-4 text-gray-500"
                      >
                        {entry}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </section>

            {/* Disclaimer */}
            <section className="rounded-xl border border-gray-200 bg-gray-50/60 p-4">
              <div className="flex items-start gap-2.5">
                <ShieldCheck
                  size={14}
                  className="mt-0.5 shrink-0 text-gray-500"
                />

                <div>
                  <div className="text-[9px] font-semibold text-gray-700">
                    Decision support only
                  </div>

                  <p className="mt-1 text-[8px] leading-4 text-gray-500">
                    {inference.response_payload
                      ?.disclaimer ??
                      "This prediction does not constitute a medical diagnosis. A qualified healthcare professional must interpret the result in clinical context."}
                  </p>
                </div>
              </div>
            </section>
          </aside>
        </div>
      )}
    </div>
  );
}