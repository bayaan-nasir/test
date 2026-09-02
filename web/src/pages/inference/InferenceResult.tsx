import {
  AlertTriangle,
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  FileImage,
  FileText,
  ImageIcon,
  Info,
  Layers3,
  ShieldCheck,
  Sparkles,
  Table2,
  UserRound,
} from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";

const result = {
  id: "INF-7F82A1",
  patientId: "PT-10482",
  patientName: "Ama Mensah",
  model: "Respiratory Assessment",
  version: "v2.4.1",
  status: "Completed",
  timestamp: "29 Aug 2026 · 14:32",
  duration: "2.84 seconds",
  primaryFinding: "Pneumonia",
  confidence: 94.7,
  risk: "High probability",
};

const predictions = [
  {
    label: "Pneumonia",
    confidence: 94.7,
    selected: true,
  },
  {
    label: "Other abnormality",
    confidence: 3.1,
    selected: false,
  },
  {
    label: "Normal",
    confidence: 2.2,
    selected: false,
  },
];

const observations = [
  ["Heart rate", "74 bpm"],
  ["Temperature", "36.7 °C"],
  ["SpO₂", "98%"],
  ["Respiratory rate", "16 /min"],
  ["Systolic BP", "118 mmHg"],
  ["Diastolic BP", "76 mmHg"],
];

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
        style={{ width: `${value}%` }}
      />
    </div>
  );
}

export function InferenceResult() {
  const navigate = useNavigate();
  const { inferenceId } = useParams();

  const id = inferenceId ?? result.id;

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Back */}
      <button
        type="button"
        onClick={() =>
          navigate(`/app/patients/${result.patientId}`)
        }
        className="mb-5 flex items-center gap-2 text-[11px] font-medium text-gray-500 hover:text-gray-900"
      >
        <ArrowLeft size={14} />
        Back to patient
      </button>

      {/* Header */}
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="flex items-center gap-1.5 rounded-md bg-green-50 px-2 py-1 text-[8px] font-semibold text-green-700">
              <CheckCircle2 size={10} />
              {result.status}
            </span>

            <span className="font-mono text-[8px] text-gray-400">
              {id}
            </span>
          </div>

          <h1 className="mt-2 text-[25px] font-semibold tracking-tight text-gray-950 sm:text-[29px]">
            AI inference result
          </h1>

          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] text-gray-400">
            <span>{result.model}</span>
            <span className="text-gray-200">•</span>
            <span>{result.version}</span>
            <span className="text-gray-200">•</span>
            <span>{result.timestamp}</span>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            className="flex h-9 items-center gap-2 rounded-lg border border-gray-200 px-3.5 text-[10px] font-semibold text-gray-600 hover:bg-gray-50"
          >
            <FileText size={13} />
            Export report
          </button>

          <button
            type="button"
            onClick={() =>
              navigate(
                `/app/inference?patient=${result.patientId}`,
              )
            }
            className="flex h-9 items-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[10px] font-semibold text-white hover:bg-gray-800"
          >
            <Sparkles size={13} />
            New inference
          </button>
        </div>
      </div>

      {/* Patient context */}
      <section className="mb-5 flex flex-col gap-3 rounded-xl border border-gray-200 bg-white px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gray-100 text-[9px] font-semibold text-gray-700">
            AM
          </div>

          <div>
            <div className="text-[11px] font-semibold text-gray-900">
              {result.patientName}
            </div>

            <div className="mt-0.5 flex items-center gap-2 text-[8px] text-gray-400">
              <span className="font-mono">{result.patientId}</span>
              <span>•</span>
              <span>34 years</span>
              <span>•</span>
              <span>Female</span>
            </div>
          </div>
        </div>

        <button
          type="button"
          onClick={() =>
            navigate(`/app/patients/${result.patientId}`)
          }
          className="flex items-center gap-1.5 text-[9px] font-semibold text-gray-500 hover:text-gray-900"
        >
          <UserRound size={12} />
          View patient record
        </button>
      </section>

      {/* Main */}
      <div className="grid gap-5 lg:grid-cols-[1fr_350px]">
        <main className="space-y-5">
          {/* Primary result */}
          <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 px-5 py-4">
              <div className="flex items-center gap-2">
                <BrainCircuit size={15} className="text-gray-400" />

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
                    {result.primaryFinding}
                  </div>

                  <div className="mt-2 flex items-center gap-2">
                    <span className="rounded-md bg-amber-50 px-2 py-1 text-[8px] font-semibold text-amber-700">
                      {result.risk}
                    </span>

                    <span className="text-[9px] text-gray-400">
                      Model confidence
                    </span>
                  </div>
                </div>

                <div className="flex justify-center md:justify-end">
                  <div className="relative flex h-36.25 w-36.25 items-center justify-center rounded-full border-10 border-gray-100">
                    <div
                      className="absolute -inset-2.5 rounded-full border-10 border-transparent"
                      style={{
                        clipPath:
                          "polygon(0 0, 100% 0, 100% 100%, 0 100%)",
                        borderTopColor: "#111827",
                        borderRightColor: "#111827",
                        transform: "rotate(35deg)",
                      }}
                    />

                    <div className="text-center">
                      <div className="text-[29px] font-semibold tracking-tight text-gray-950">
                        {result.confidence}%
                      </div>

                      <div className="text-[8px] uppercase tracking-wide text-gray-400">
                        confidence
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-7 rounded-lg bg-gray-50 p-4">
                <div className="flex items-start gap-2.5">
                  <Info
                    size={13}
                    className="mt-0.5 shrink-0 text-gray-400"
                  />

                  <div>
                    <div className="text-[9px] font-semibold text-gray-700">
                      Interpretation
                    </div>

                    <p className="mt-1 text-[9px] leading-5 text-gray-500">
                      The model identified a high probability of pneumonia
                      based on the supplied clinical observations and
                      medical image. This output is an AI-assisted
                      prediction and should be evaluated alongside the
                      patient's clinical presentation.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Prediction distribution */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 px-5 py-4">
              <h2 className="text-[12px] font-semibold text-gray-950">
                Prediction distribution
              </h2>

              <p className="mt-0.5 text-[9px] text-gray-400">
                Model output across candidate classes
              </p>
            </div>

            <div className="space-y-5 p-5">
              {predictions.map((prediction) => (
                <div key={prediction.label}>
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {prediction.selected && (
                        <CheckCircle2
                          size={12}
                          className="text-gray-900"
                        />
                      )}

                      <span className="text-[10px] font-medium text-gray-700">
                        {prediction.label}
                      </span>
                    </div>

                    <span className="font-mono text-[9px] text-gray-500">
                      {prediction.confidence.toFixed(1)}%
                    </span>
                  </div>

                  <ConfidenceBar
                    value={prediction.confidence}
                    selected={prediction.selected}
                  />
                </div>
              ))}
            </div>
          </section>

          {/* Input data */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 px-5 py-4">
              <div className="flex items-center gap-2">
                <Layers3 size={14} className="text-gray-400" />

                <h2 className="text-[12px] font-semibold text-gray-950">
                  Inference inputs
                </h2>
              </div>
            </div>

            <div className="grid gap-5 p-5 md:grid-cols-2">
              {/* Tabular */}
              <div>
                <div className="mb-3 flex items-center gap-2">
                  <Table2 size={13} className="text-gray-400" />

                  <span className="text-[9px] font-semibold text-gray-700">
                    Clinical data
                  </span>
                </div>

                <div className="divide-y divide-gray-100 rounded-lg border border-gray-200">
                  {observations.map(([label, value]) => (
                    <div
                      key={label}
                      className="flex items-center justify-between px-3 py-2.5"
                    >
                      <span className="text-[8px] text-gray-400">
                        {label}
                      </span>

                      <span className="font-mono text-[9px] font-medium text-gray-700">
                        {value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Image */}
              <div>
                <div className="mb-3 flex items-center gap-2">
                  <FileImage size={13} className="text-gray-400" />

                  <span className="text-[9px] font-semibold text-gray-700">
                    Medical image
                  </span>
                </div>

                <div className="relative flex aspect-4/3 items-center justify-center overflow-hidden rounded-lg bg-gray-900">
                  <div className="absolute inset-0 opacity-20">
                    <div className="absolute left-1/4 top-1/2 h-32 w-20 -translate-y-1/2 rounded-[50%] border border-white" />
                    <div className="absolute right-1/4 top-1/2 h-32 w-20 -translate-y-1/2 rounded-[50%] border border-white" />
                  </div>

                  <div className="relative text-center">
                    <ImageIcon
                      size={28}
                      className="mx-auto text-gray-500"
                    />

                    <div className="mt-2 text-[8px] text-gray-500">
                      Chest X-ray
                    </div>
                  </div>

                  <div className="absolute bottom-2 left-2 rounded bg-black/60 px-2 py-1 font-mono text-[7px] text-gray-400">
                    chest_xray_01842.png
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Technical details */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 px-5 py-4">
              <h2 className="text-[12px] font-semibold text-gray-950">
                Inference metadata
              </h2>
            </div>

            <div className="grid gap-x-8 gap-y-4 p-5 sm:grid-cols-2 lg:grid-cols-3">
              <div>
                <div className="text-[8px] uppercase tracking-wide text-gray-400">
                  Engine
                </div>

                <div className="mt-1 text-[9px] font-medium text-gray-700">
                  Multimodal Inference Engine
                </div>
              </div>

              <div>
                <div className="text-[8px] uppercase tracking-wide text-gray-400">
                  Model version
                </div>

                <div className="mt-1 font-mono text-[9px] font-medium text-gray-700">
                  {result.version}
                </div>
              </div>

              <div>
                <div className="text-[8px] uppercase tracking-wide text-gray-400">
                  Processing time
                </div>

                <div className="mt-1 flex items-center gap-1.5 text-[9px] font-medium text-gray-700">
                  <Clock3 size={11} className="text-gray-400" />
                  {result.duration}
                </div>
              </div>

              <div>
                <div className="text-[8px] uppercase tracking-wide text-gray-400">
                  Input modalities
                </div>

                <div className="mt-1 text-[9px] font-medium text-gray-700">
                  Tabular + Image
                </div>
              </div>

              <div>
                <div className="text-[8px] uppercase tracking-wide text-gray-400">
                  Inference ID
                </div>

                <div className="mt-1 font-mono text-[9px] font-medium text-gray-700">
                  {id}
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

        {/* Right rail */}
        <aside className="space-y-5">
          {/* Review */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 px-5 py-4">
              <h2 className="text-[12px] font-semibold text-gray-950">
                Clinical review
              </h2>
            </div>

            <div className="p-5">
              <div className="rounded-lg border border-amber-100 bg-amber-50/50 p-3.5">
                <div className="flex items-start gap-2.5">
                  <AlertTriangle
                    size={14}
                    className="mt-0.5 shrink-0 text-amber-600"
                  />

                  <div>
                    <div className="text-[10px] font-semibold text-amber-900">
                      Review required
                    </div>

                    <p className="mt-1 text-[8px] leading-4 text-amber-800/70">
                      This result has not yet been reviewed by a clinician.
                    </p>
                  </div>
                </div>
              </div>

              <button
                type="button"
                className="mt-4 flex h-9 w-full items-center justify-center gap-2 rounded-lg bg-gray-950 text-[9px] font-semibold text-white hover:bg-gray-800"
              >
                <CheckCircle2 size={12} />
                Mark as reviewed
              </button>
            </div>
          </section>

          {/* Model */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 px-5 py-4">
              <h2 className="text-[12px] font-semibold text-gray-950">
                Model information
              </h2>
            </div>

            <div className="p-5">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gray-950 text-white">
                  <BrainCircuit size={15} />
                </div>

                <div>
                  <div className="text-[10px] font-semibold text-gray-800">
                    {result.model}
                  </div>

                  <div className="mt-0.5 font-mono text-[8px] text-gray-400">
                    {result.version}
                  </div>
                </div>
              </div>

              <div className="mt-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[8px] text-gray-400">
                    Modality
                  </span>

                  <span className="text-[9px] font-medium text-gray-700">
                    Multimodal
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-[8px] text-gray-400">
                    Model status
                  </span>

                  <span className="flex items-center gap-1 text-[9px] font-medium text-green-700">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                    Production
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-[8px] text-gray-400">
                    Calibration
                  </span>

                  <span className="text-[9px] font-medium text-gray-700">
                    Verified
                  </span>
                </div>
              </div>
            </div>
          </section>

          {/* Safety */}
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
                  This prediction does not constitute a medical diagnosis.
                  A qualified healthcare professional must interpret the
                  result in clinical context.
                </p>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}