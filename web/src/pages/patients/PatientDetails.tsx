import {
  Activity,
  ArrowLeft,
  ArrowUpRight,
  CalendarDays,
  ChevronRight,
  CircleUserRound,
  Clock3,
  FileImage,
  FileText,
  HeartPulse,
  Mail,
  MapPin,
  MoreHorizontal,
  Phone,
  Play,
  Plus,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";

interface PatientData {
  id: string;
  name: string;
  initials: string;
  age: number;
  sex: string;
  dateOfBirth: string;
  phone: string;
  email: string;
  location: string;
  status: "Active" | "Needs review" | "Inactive";
  registered: string;
  lastVisit: string;
  bloodGroup: string;
  allergies: string;
  emergencyContact: string;
}

const patientData: Record<string, PatientData> = {
  "PT-10482": {
    id: "PT-10482",
    name: "Ama Mensah",
    initials: "AM",
    age: 34,
    sex: "Female",
    dateOfBirth: "14 March 1992",
    phone: "+233 24 381 9204",
    email: "ama.mensah@example.com",
    location: "Kumasi, Ghana",
    status: "Needs review",
    registered: "12 January 2025",
    lastVisit: "29 August 2026",
    bloodGroup: "O+",
    allergies: "None recorded",
    emergencyContact: "Kwesi Mensah · +233 20 441 2901",
  },
};

const inferenceHistory = [
  {
    id: "INF-7F82A1",
    model: "Respiratory Assessment",
    version: "v2.4.1",
    result: "Pneumonia",
    confidence: "94.7%",
    date: "29 Aug 2026 · 14:32",
    status: "Review required",
    modality: "Multimodal",
  },
  {
    id: "INF-6A91D4",
    model: "Chest X-Ray Classifier",
    version: "v3.1.0",
    result: "Abnormal finding",
    confidence: "91.8%",
    date: "29 Aug 2026 · 14:31",
    status: "Completed",
    modality: "Image",
  },
  {
    id: "INF-3D72B8",
    model: "Respiratory Assessment",
    version: "v2.3.2",
    result: "Normal",
    confidence: "67.2%",
    date: "14 Jul 2026 · 10:18",
    status: "Completed",
    modality: "Multimodal",
  },
];

const timeline = [
  {
    date: "29 Aug 2026",
    time: "14:32",
    title: "AI inference completed",
    description:
      "Respiratory Assessment identified pneumonia with 94.7% confidence.",
    type: "ai",
  },
  {
    date: "29 Aug 2026",
    time: "14:10",
    title: "Clinical consultation",
    description:
      "Patient presented with cough, fever and increased respiratory rate.",
    type: "clinical",
  },
  {
    date: "29 Aug 2026",
    time: "13:54",
    title: "Chest X-ray uploaded",
    description: "Chest radiograph added to the patient's clinical record.",
    type: "image",
  },
  {
    date: "12 Jan 2025",
    time: "09:21",
    title: "Patient registered",
    description: "Patient record created in the clinical management system.",
    type: "registration",
  },
];

const documents = [
  {
    name: "Chest X-ray — 29 Aug 2026",
    type: "Medical image",
    size: "2.4 MB",
    icon: FileImage,
  },
  {
    name: "Clinical consultation — 29 Aug 2026",
    type: "Clinical note",
    size: "18 KB",
    icon: FileText,
  },
  {
    name: "Laboratory results — 29 Aug 2026",
    type: "Laboratory report",
    size: "184 KB",
    icon: FileText,
  },
];

function StatusBadge({ status }: { status: PatientData["status"] }) {
  const styles = {
    Active: "bg-green-50 text-green-700",
    "Needs review": "bg-amber-50 text-amber-700",
    Inactive: "bg-gray-100 text-gray-500",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[8px] font-semibold ${styles[status]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          status === "Active"
            ? "bg-green-500"
            : status === "Needs review"
              ? "bg-amber-500"
              : "bg-gray-400"
        }`}
      />
      {status}
    </span>
  );
}

function TimelineIcon({ type }: { type: string }) {
  const icons = {
    ai: Activity,
    clinical: Stethoscope,
    image: FileImage,
    registration: CircleUserRound,
  };

  const Icon = icons[type as keyof typeof icons] ?? Activity;

  return (
    <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-gray-200 bg-white text-gray-400">
      <Icon size={13} />
    </div>
  );
}

export function PatientDetails() {
  const navigate = useNavigate();
  const { patientId } = useParams();

  const patient =
    patientData[patientId ?? "PT-10482"] ?? patientData["PT-10482"];

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      {/* Back */}
      <button
        type="button"
        onClick={() => navigate("/app/patients")}
        className="mb-5 flex items-center gap-2 text-[9px] font-medium text-gray-400 transition hover:text-gray-800"
      >
        <ArrowLeft size={12} />
        Back to patients
      </button>

      {/* Patient header */}
      <section className="mb-6 overflow-hidden rounded-xl border border-gray-200 bg-white">
        <div className="p-5 sm:p-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div className="flex min-w-0 gap-4">
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[13px] font-semibold text-gray-500">
                {patient.initials}
              </div>

              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h1 className="text-[20px] font-semibold tracking-tight text-gray-950 sm:text-[23px]">
                    {patient.name}
                  </h1>

                  <StatusBadge status={patient.status} />
                </div>

                <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="font-mono text-[8px] text-gray-400">
                    {patient.id}
                  </span>

                  <span className="text-gray-200">•</span>

                  <span className="text-[8px] text-gray-500">
                    {patient.age} years
                  </span>

                  <span className="text-gray-200">•</span>

                  <span className="text-[8px] text-gray-500">
                    {patient.sex}
                  </span>
                </div>

                <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
                  <span className="flex items-center gap-1.5 text-[8px] text-gray-400">
                    <Phone size={10} />
                    {patient.phone}
                  </span>

                  <span className="flex items-center gap-1.5 text-[8px] text-gray-400">
                    <Mail size={10} />
                    {patient.email}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex shrink-0 flex-wrap gap-2">
              <button
                type="button"
                onClick={() => navigate("/app/inference")}
                className="flex h-9 items-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[9px] font-semibold text-white hover:bg-gray-800"
              >
                <Play size={11} />
                Run AI inference
              </button>

              <button
                type="button"
                className="flex h-9 items-center gap-2 rounded-lg border border-gray-200 px-3 text-[9px] font-medium text-gray-600 hover:bg-gray-50"
              >
                Edit patient
              </button>

              <button
                type="button"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-50"
              >
                <MoreHorizontal size={13} />
              </button>
            </div>
          </div>
        </div>

        <div className="grid border-t border-gray-100 sm:grid-cols-2 lg:grid-cols-4">
          <div className="border-b border-gray-100 px-5 py-4 sm:border-r lg:border-b-0">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Date of birth
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <CalendarDays size={11} className="text-gray-400" />
              {patient.dateOfBirth}
            </div>
          </div>

          <div className="border-b border-gray-100 px-5 py-4 lg:border-b-0 lg:border-r">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Location
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <MapPin size={11} className="text-gray-400" />
              {patient.location}
            </div>
          </div>

          <div className="border-b border-gray-100 px-5 py-4 sm:border-r lg:border-b-0">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Blood group
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <HeartPulse size={11} className="text-gray-400" />
              {patient.bloodGroup}
            </div>
          </div>

          <div className="px-5 py-4">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Registered
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <Clock3 size={11} className="text-gray-400" />
              {patient.registered}
            </div>
          </div>
        </div>
      </section>

      {/* Review banner */}
      {patient.status === "Needs review" && (
        <div className="mb-6 flex flex-col gap-3 rounded-xl border border-amber-200 bg-amber-50/50 p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-700">
              <Activity size={13} />
            </div>

            <div>
              <div className="text-[9px] font-semibold text-amber-900">
                AI assessment requires clinical review
              </div>

              <p className="mt-1 text-[8px] leading-4 text-amber-700">
                The latest inference identified pneumonia with 94.7% confidence.
                Review the assessment before taking clinical action.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => navigate(`/app/inference/${inferenceHistory[0].id}`)}
            className="flex h-8 shrink-0 items-center justify-center gap-1.5 rounded-lg border border-amber-200 bg-white px-3 text-[8px] font-semibold text-amber-800 hover:bg-amber-50"
          >
            Review assessment
            <ArrowUpRight size={10} />
          </button>
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
        {/* Left */}
        <div className="space-y-6">
          {/* Clinical summary */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">
                  Clinical summary
                </h2>

                <p className="mt-1 text-[8px] text-gray-400">
                  Key information from the patient's record.
                </p>
              </div>

              <button
                type="button"
                className="text-[8px] font-medium text-gray-400 hover:text-gray-700"
              >
                Edit
              </button>
            </div>

            <div className="grid sm:grid-cols-2">
              <div className="border-b border-gray-100 p-4 sm:border-r">
                <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                  Allergies
                </div>

                <div className="mt-2 text-[9px] font-medium text-gray-700">
                  {patient.allergies}
                </div>
              </div>

              <div className="border-b border-gray-100 p-4">
                <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                  Emergency contact
                </div>

                <div className="mt-2 text-[9px] font-medium text-gray-700">
                  {patient.emergencyContact}
                </div>
              </div>

              <div className="p-4 sm:border-r">
                <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                  Last visit
                </div>

                <div className="mt-2 text-[9px] font-medium text-gray-700">
                  {patient.lastVisit}
                </div>
              </div>

              <div className="p-4">
                <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                  Record status
                </div>

                <div className="mt-2">
                  <StatusBadge status={patient.status} />
                </div>
              </div>
            </div>
          </section>

          {/* Timeline */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">
                  Clinical timeline
                </h2>

                <p className="mt-1 text-[8px] text-gray-400">
                  Recent events in the patient's record.
                </p>
              </div>

              <button
                type="button"
                className="flex items-center gap-1.5 text-[8px] font-medium text-gray-400 hover:text-gray-700"
              >
                <Plus size={11} />
                Add event
              </button>
            </div>

            <div className="p-5">
              <div className="relative">
                <div className="absolute bottom-5 left-4 top-5 w-px bg-gray-100" />

                <div className="space-y-6">
                  {timeline.map((event) => (
                    <div
                      key={`${event.date}-${event.time}-${event.title}`}
                      className="relative flex gap-3"
                    >
                      <TimelineIcon type={event.type} />

                      <div className="min-w-0 flex-1 pt-0.5">
                        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                          <div className="text-[9px] font-semibold text-gray-700">
                            {event.title}
                          </div>

                          <div className="font-mono text-[7px] text-gray-300">
                            {event.date} · {event.time}
                          </div>
                        </div>

                        <p className="mt-1.5 max-w-xl text-[8px] leading-4 text-gray-400">
                          {event.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <button
              type="button"
              className="flex w-full items-center justify-center gap-1.5 border-t border-gray-100 py-3 text-[8px] font-medium text-gray-400 hover:bg-gray-50 hover:text-gray-700"
            >
              View complete timeline
              <ChevronRight size={10} />
            </button>
          </section>

          {/* Documents */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">
                  Documents & imaging
                </h2>

                <p className="mt-1 text-[8px] text-gray-400">
                  Files attached to this patient's record.
                </p>
              </div>

              <button
                type="button"
                className="flex items-center gap-1.5 text-[8px] font-medium text-gray-400 hover:text-gray-700"
              >
                <Plus size={11} />
                Upload
              </button>
            </div>

            <div className="divide-y divide-gray-100">
              {documents.map((document) => {
                const Icon = document.icon;

                return (
                  <button
                    key={document.name}
                    type="button"
                    className="flex w-full items-center gap-3 p-4 text-left hover:bg-gray-50"
                  >
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-50">
                      <Icon size={13} className="text-gray-400" />
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[9px] font-medium text-gray-700">
                        {document.name}
                      </div>

                      <div className="mt-1 text-[7px] text-gray-400">
                        {document.type} · {document.size}
                      </div>
                    </div>

                    <ArrowUpRight
                      size={11}
                      className="shrink-0 text-gray-300"
                    />
                  </button>
                );
              })}
            </div>
          </section>
        </div>

        {/* Right */}
        <div className="space-y-6">
          {/* Current AI assessment */}
          <section className="overflow-hidden rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 p-5">
              <div className="flex items-center gap-2">
                <Activity size={13} className="text-gray-400" />

                <h2 className="text-[11px] font-semibold text-gray-900">
                  Current AI assessment
                </h2>
              </div>

              <p className="mt-1 text-[8px] text-gray-400">
                Latest result across the patient's inference history.
              </p>
            </div>

            <div className="p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                    Primary prediction
                  </div>

                  <div className="mt-2 text-[20px] font-semibold tracking-tight text-gray-950">
                    Pneumonia
                  </div>

                  <div className="mt-1 text-[8px] text-gray-400">
                    Respiratory Assessment · v2.4.1
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[20px] font-semibold tracking-tight text-gray-950">
                    94.7%
                  </div>

                  <div className="mt-1 text-[7px] text-gray-400">
                    confidence
                  </div>
                </div>
              </div>

              <div className="mt-5">
                <div className="h-1.5 overflow-hidden rounded-full bg-gray-100">
                  <div
                    className="h-full rounded-full bg-gray-900"
                    style={{ width: "94.7%" }}
                  />
                </div>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-2">
                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Modality
                  </div>

                  <div className="mt-1.5 text-[8px] font-medium text-gray-700">
                    Multimodal
                  </div>
                </div>

                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="text-[7px] uppercase tracking-wide text-gray-400">
                    Inference
                  </div>

                  <div className="mt-1.5 font-mono text-[8px] font-medium text-gray-700">
                    {inferenceHistory[0].id}
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={() =>
                  navigate(`/app/inference/${inferenceHistory[0].id}`)
                }
                className="mt-4 flex h-8 w-full items-center justify-center gap-2 rounded-lg border border-gray-200 text-[8px] font-semibold text-gray-600 hover:bg-gray-50"
              >
                View full assessment
                <ArrowUpRight size={10} />
              </button>
            </div>
          </section>

          {/* Inference history */}
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">
                  AI inference history
                </h2>

                <p className="mt-1 text-[8px] text-gray-400">
                  Previous assessments for this patient.
                </p>
              </div>

              <span className="rounded-md bg-gray-50 px-2 py-1 text-[7px] font-medium text-gray-400">
                {inferenceHistory.length} runs
              </span>
            </div>

            <div className="divide-y divide-gray-100">
              {inferenceHistory.map((inference) => (
                <button
                  key={inference.id}
                  type="button"
                  onClick={() => navigate(`/app/inference/${inference.id}`)}
                  className="group w-full p-4 text-left hover:bg-gray-50"
                >
                  <div className="flex items-start gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-50">
                      <Activity size={12} className="text-gray-400" />
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-[9px] font-semibold text-gray-700">
                          {inference.result}
                        </span>

                        <span className="font-mono text-[8px] font-semibold text-gray-600">
                          {inference.confidence}
                        </span>
                      </div>

                      <div className="mt-1 text-[8px] text-gray-400">
                        {inference.model}
                      </div>

                      <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
                        <span className="font-mono text-[7px] text-gray-300">
                          {inference.id}
                        </span>

                        <span className="text-gray-200">•</span>

                        <span className="text-[7px] text-gray-300">
                          {inference.modality}
                        </span>

                        <span className="text-gray-200">•</span>

                        <span className="text-[7px] text-gray-300">
                          {inference.date}
                        </span>
                      </div>

                      {inference.status === "Review required" && (
                        <div className="mt-2 inline-flex rounded-md bg-amber-50 px-2 py-1 text-[7px] font-semibold text-amber-700">
                          Review required
                        </div>
                      )}
                    </div>

                    <ChevronRight
                      size={11}
                      className="mt-1 shrink-0 text-gray-300 opacity-0 transition group-hover:opacity-100"
                    />
                  </div>
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={() => navigate("/app/inference/history")}
              className="flex w-full items-center justify-center gap-1.5 border-t border-gray-100 py-3 text-[8px] font-medium text-gray-400 hover:bg-gray-50 hover:text-gray-700"
            >
              View all inference history
              <ArrowUpRight size={10} />
            </button>
          </section>

          {/* Privacy */}
          <div className="flex gap-3 rounded-xl border border-gray-200 bg-gray-50/50 p-4">
            <ShieldCheck size={14} className="mt-0.5 shrink-0 text-gray-400" />

            <div>
              <div className="text-[8px] font-semibold text-gray-700">
                Protected clinical record
              </div>

              <p className="mt-1 text-[7px] leading-4 text-gray-400">
                Patient information and AI assessments are protected by the
                system's access controls and audit mechanisms.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
