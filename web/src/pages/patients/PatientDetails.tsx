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
  Mail,
  MoreHorizontal,
  Phone,
  Play,
  Plus,
  Check,
  Copy,
  Pencil,
  Trash2,
  ShieldCheck,
  Stethoscope,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import {
  deletePatient,
  getPatient,
  updatePatient,
  type Patient as ApiPatient,
  type UpdatePatientPayload,
} from "../../api/patients";

// ── Demo-only content ──────────────────────────────────────────────────
// There is no backend endpoint yet that aggregates a patient's inference
// history, timeline, or documents. These stay as illustrative placeholder
// data (matching the rest of the app — Dashboard, PatientRecord, etc. are
// mock too) until that API exists. They are intentionally NOT derived from
// the real patient record below.
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
];
// ────────────────────────────────────────────────────────────────────────

function formatDate(dateValue?: string | null) {
  if (!dateValue) return "Not recorded";
  const parsed = new Date(dateValue);
  if (Number.isNaN(parsed.getTime())) return "Not recorded";

  return parsed.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function calculateAge(dateOfBirth?: string | null) {
  if (!dateOfBirth) return null;

  const birth = new Date(dateOfBirth);
  if (Number.isNaN(birth.getTime())) return null;

  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();

  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age -= 1;
  }

  return age;
}

function formatSex(sex?: string | null) {
  if (!sex) return "Not recorded";
  return sex.charAt(0).toUpperCase() + sex.slice(1).toLowerCase();
}

function getInitials(patient: ApiPatient) {
  const source =
    patient.full_name ??
    `${patient.first_name ?? ""} ${patient.last_name ?? ""}`;

  return source
    .trim()
    .split(/\s   /)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
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

interface EditPatientModalProps {
  patient: ApiPatient;
  onClose: () => void;
  onSaved: (patient: ApiPatient) => void;
}

function EditPatientModal({
  patient,
  onClose,
  onSaved,
}: EditPatientModalProps) {
  const [formData, setFormData] = useState<UpdatePatientPayload>({
    first_name: patient.first_name ?? "",
    last_name: patient.last_name ?? "",
    date_of_birth: patient.date_of_birth ?? "",
    sex: patient.sex ?? "",
    phone_number: patient.phone_number ?? "",
    email: patient.email ?? "",
    notes: patient.notes ?? "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    setError(null);

    if (!formData.first_name?.trim() || !formData.last_name?.trim()) {
      setError("First and last names are required.");
      return;
    }

    setIsSubmitting(true);
    try {
      const updated = await updatePatient(patient.patient_id, formData);
      onSaved(updated);
    } catch (err) {
      const maybeError = err as {
        response?: { data?: { detail?: string; [key: string]: unknown } };
      };
      const message =
        maybeError.response?.data?.detail ??
        "Unable to update patient. Please try again.";
      setError(
        typeof message === "string" ? message : "Unable to update patient.",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="mx-4 w-full max-w-md rounded-xl border border-gray-200 bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-100 p-5">
          <div>
            <h2 className="text-[11px] font-semibold text-gray-900">
              Edit patient
            </h2>
            <p className="mt-1 text-[8px] text-gray-400">
              Update this patient's record.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-6 w-6 items-center justify-center rounded-md text-gray-400 hover:bg-gray-100"
          >
            <X size={14} />
          </button>
        </div>

        <div className="space-y-3 p-5">
          <div className="grid gap-3 sm:grid-cols-2">
            <label>
              <div className="mb-1 text-[8px] font-medium text-gray-600">
                First name *
              </div>
              <input
                type="text"
                value={formData.first_name}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    first_name: e.target.value,
                  }))
                }
                className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
              />
            </label>

            <label>
              <div className="mb-1 text-[8px] font-medium text-gray-600">
                Last name *
              </div>
              <input
                type="text"
                value={formData.last_name}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    last_name: e.target.value,
                  }))
                }
                className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
              />
            </label>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label>
              <div className="mb-1 text-[8px] font-medium text-gray-600">
                Date of birth
              </div>
              <input
                type="date"
                value={formData.date_of_birth ?? ""}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    date_of_birth: e.target.value,
                  }))
                }
                className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
              />
            </label>

            <label>
              <div className="mb-1 text-[8px] font-medium text-gray-600">
                Sex
              </div>
              <select
                value={formData.sex ?? ""}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, sex: e.target.value }))
                }
                className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
              >
                <option value="">Select sex</option>
                <option value="MALE">Male</option>
                <option value="FEMALE">Female</option>
                <option value="OTHER">Other</option>
              </select>
            </label>
          </div>

          <label>
            <div className="mb-1 text-[8px] font-medium text-gray-600">
              Phone number
            </div>
            <input
              type="tel"
              value={formData.phone_number ?? ""}
              onChange={(e) =>
                setFormData((prev) => ({
                  ...prev,
                  phone_number: e.target.value,
                }))
              }
              className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
            />
          </label>

          <label>
            <div className="mb-1 text-[8px] font-medium text-gray-600">
              Email
            </div>
            <input
              type="email"
              value={formData.email ?? ""}
              onChange={(e) =>
                setFormData((prev) => ({ ...prev, email: e.target.value }))
              }
              className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
            />
          </label>

          <label>
            <div className="mb-1 text-[8px] font-medium text-gray-600">
              Clinical notes
            </div>
            <textarea
              value={formData.notes ?? ""}
              onChange={(e) =>
                setFormData((prev) => ({ ...prev, notes: e.target.value }))
              }
              rows={3}
              className="w-full resize-none rounded-lg border border-gray-200 bg-gray-50/50 p-2.5 text-[9px] leading-4 text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
            />
          </label>

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[8px] text-red-700">
              {error}
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-gray-100 p-4">
          <button
            type="button"
            onClick={onClose}
            className="h-9 rounded-lg border border-gray-200 px-4 text-[8px] font-medium text-gray-600 hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={isSubmitting}
            className="h-9 rounded-lg bg-gray-950 px-4 text-[8px] font-semibold text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isSubmitting ? "Saving..." : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}

interface DeletePatientModalProps {
  patient: ApiPatient;
  onClose: () => void;
  onDeleted: () => void;
}

function DeletePatientModal({
  patient,
  onClose,
  onDeleted,
}: DeletePatientModalProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fullName =
    patient.full_name ?? `${patient.first_name} ${patient.last_name}`;

  const handleDelete = async () => {
    setIsDeleting(true);
    setError(null);

    try {
      await deletePatient(patient.patient_id);
      onDeleted();
    } catch (err) {
      const maybeError = err as {
        response?: { status?: number; data?: { detail?: string } };
      };
      const message =
        maybeError.response?.data?.detail ??
        (maybeError.response?.status === 409
          ? "This patient has existing AI inference records and cannot be deleted."
          : "Unable to delete this patient. Please try again.");
      setError(message);
      setIsDeleting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="mx-4 w-full max-w-sm rounded-xl border border-gray-200 bg-white shadow-xl">
        <div className="p-5">
          <h2 className="text-[11px] font-semibold text-gray-900">
            Delete patient
          </h2>

          <p className="mt-2 text-[9px] leading-4 text-gray-500">
            Are you sure you want to delete{" "}
            <span className="font-semibold text-gray-700">{fullName}</span> (
            {patient.patient_id})? This action cannot be undone.
          </p>

          {error && (
            <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[8px] text-red-700">
              {error}
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-gray-100 p-4">
          <button
            type="button"
            onClick={onClose}
            disabled={isDeleting}
            className="h-9 rounded-lg border border-gray-200 px-4 text-[8px] font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-50"
          >
            Cancel
          </button>

          <button
            type="button"
            onClick={handleDelete}
            disabled={isDeleting}
            className="h-9 rounded-lg bg-red-600 px-4 text-[8px] font-semibold text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isDeleting ? "Deleting..." : "Delete patient"}
          </button>
        </div>
      </div>
    </div>
  );
}

export function PatientDetails() {
  const navigate = useNavigate();
  const { patientId } = useParams();

  const [patient, setPatient] = useState<ApiPatient | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showActionsMenu, setShowActionsMenu] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [copied, setCopied] = useState(false);

  const actionsMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleOutsideClick(event: MouseEvent) {
      if (
        actionsMenuRef.current &&
        !actionsMenuRef.current.contains(event.target as Node)
      ) {
        setShowActionsMenu(false);
      }
    }
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, []);

  useEffect(() => {
    if (!patientId) {
      setError("No patient specified.");
      setIsLoading(false);
      return;
    }

    let active = true;

    async function loadPatient() {
      setIsLoading(true);
      setError(null);

      try {
        const data = await getPatient(patientId!);
        if (active) {
          setPatient(data);
        }
      } catch (err) {
        if (!active) return;

        const maybeError = err as { response?: { status?: number } };
        if (maybeError.response?.status === 404) {
          setError("This patient could not be found.");
        } else if (maybeError.response?.status === 403) {
          setError("You do not have access to this patient's record.");
        } else {
          setError("Unable to load this patient. Please try again.");
        }
      } finally {
        if (active) {
          setIsLoading(false);
        }
      }
    }

    loadPatient();

    return () => {
      active = false;
    };
  }, [patientId]);

  if (isLoading) {
    return (
      <div className="flex min-h-100 items-center justify-center">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-gray-900" />
      </div>
    );
  }

  if (error || !patient) {
    return (
      <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
        <button
          type="button"
          onClick={() => navigate("/app/patients")}
          className="mb-5 flex items-center gap-2 text-[9px] font-medium text-gray-400 transition hover:text-gray-800"
        >
          <ArrowLeft size={12} />
          Back to patients
        </button>

        <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
          <p className="text-[11px] font-semibold text-gray-700">
            {error ?? "This patient could not be found."}
          </p>
        </div>
      </div>
    );
  }

  const fullName =
    patient.full_name ?? `${patient.first_name} ${patient.last_name}`;
  const age = calculateAge(patient.date_of_birth);

  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(patient.patient_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API unavailable — menu just stays open, no crash.
    }
  };

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
                {getInitials(patient)}
              </div>

              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h1 className="text-[20px] font-semibold tracking-tight text-gray-950 sm:text-[23px]">
                    {fullName}
                  </h1>
                </div>

                <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="font-mono text-[8px] text-gray-400">
                    {patient.patient_id}
                  </span>

                  <span className="text-gray-200">•</span>

                  <span className="text-[8px] text-gray-500">
                    {age !== null ? `${age} years` : "Age unknown"}
                  </span>

                  <span className="text-gray-200">•</span>

                  <span className="text-[8px] text-gray-500">
                    {formatSex(patient.sex)}
                  </span>
                </div>

                <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
                  <span className="flex items-center gap-1.5 text-[8px] text-gray-400">
                    <Phone size={10} />
                    {patient.phone_number || "No phone on file"}
                  </span>

                  <span className="flex items-center gap-1.5 text-[8px] text-gray-400">
                    <Mail size={10} />
                    {patient.email || "No email on file"}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex shrink-0 flex-wrap gap-2">
              <button
                type="button"
                onClick={() =>
                  navigate(`/app/inference?patient=${patient.patient_id}`)
                }
                className="flex h-9 items-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[9px] font-semibold text-white hover:bg-gray-800"
              >
                <Play size={11} />
                Run AI inference
              </button>

              <div className="relative" ref={actionsMenuRef}>
                <button
                  type="button"
                  onClick={() => setShowActionsMenu((value) => !value)}
                  aria-label="More actions"
                  aria-expanded={showActionsMenu}
                  className="flex h-9 w-9 items-center justify-center rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-50"
                >
                  <MoreHorizontal size={13} />
                </button>

                {showActionsMenu && (
                  <div className="absolute right-0 top-[calc(100%   8px)] z-30 w-48 overflow-hidden rounded-lg border border-gray-200 bg-white p-1.5 shadow-lg">
                    <button
                      type="button"
                      onClick={handleCopyId}
                      className="flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-[9px] text-gray-600 hover:bg-gray-50"
                    >
                      {copied ? (
                        <Check size={12} className="text-green-600" />
                      ) : (
                        <Copy size={12} className="text-gray-400" />
                      )}
                      {copied ? "Copied!" : "Copy patient ID"}
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setShowActionsMenu(false);
                        setShowEditModal(true);
                      }}
                      className="flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-[9px] text-gray-600 hover:bg-gray-50"
                    >
                      <Pencil size={12} className="text-gray-400" />
                      Edit patient
                    </button>

                    <div className="my-1 h-px bg-gray-100" />

                    <button
                      type="button"
                      onClick={() => {
                        setShowActionsMenu(false);
                        setShowDeleteModal(true);
                      }}
                      className="flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-[9px] text-red-600 hover:bg-red-50"
                    >
                      <Trash2 size={12} />
                      Delete patient
                    </button>
                  </div>
                )}
              </div>
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
              {formatDate(patient.date_of_birth)}
            </div>
          </div>

          <div className="border-b border-gray-100 px-5 py-4 lg:border-b-0 lg:border-r">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Sex
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <CircleUserRound size={11} className="text-gray-400" />
              {formatSex(patient.sex)}
            </div>
          </div>

          <div className="border-b border-gray-100 px-5 py-4 sm:border-r lg:border-b-0">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Contact number
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <Phone size={11} className="text-gray-400" />
              {patient.phone_number || "Not recorded"}
            </div>
          </div>

          <div className="px-5 py-4">
            <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
              Registered
            </div>

            <div className="mt-2 flex items-center gap-2 text-[9px] font-medium text-gray-700">
              <Clock3 size={11} className="text-gray-400" />
              {formatDate(patient.created_at)}
            </div>
          </div>
        </div>
      </section>

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
                onClick={() => setShowEditModal(true)}
                className="text-[8px] font-medium text-gray-400 hover:text-gray-700"
              >
                Edit
              </button>
            </div>

            <div className="grid sm:grid-cols-2">
              <div className="border-b border-gray-100 p-4 sm:border-r">
                <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                  Notes
                </div>

                <div className="mt-2 text-[9px] font-medium text-gray-700">
                  {patient.notes?.trim() || "No clinical notes recorded"}
                </div>
              </div>

              <div className="border-b border-gray-100 p-4">
                <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                  Email
                </div>

                <div className="mt-2 text-[9px] font-medium text-gray-700">
                  {patient.email || "Not recorded"}
                </div>
              </div>

              <div className="p-4 sm:border-r">
                <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                  Last updated
                </div>

                <div className="mt-2 text-[9px] font-medium text-gray-700">
                  {formatDate(patient.updated_at)}
                </div>
              </div>

              <div className="p-4">
                <div className="text-[7px] font-semibold uppercase tracking-wide text-gray-400">
                  Record status
                </div>

                <div className="mt-2">
                  <span className="inline-flex items-center gap-1.5 rounded-md bg-green-50 px-2 py-1 text-[8px] font-semibold text-green-700">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                    Active
                  </span>
                </div>
              </div>
            </div>
          </section>

          {/* Timeline — demo content, see note at top of file */}
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
          </section>

          {/* Documents — demo content, see note at top of file */}
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

        {/* Right — AI history, demo content, see note at top of file */}
        <div className="space-y-6">
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

      {showEditModal && (
        <EditPatientModal
          patient={patient}
          onClose={() => setShowEditModal(false)}
          onSaved={(updated) => {
            setPatient(updated);
            setShowEditModal(false);
          }}
        />
      )}

      {showDeleteModal && (
        <DeletePatientModal
          patient={patient}
          onClose={() => setShowDeleteModal(false)}
          onDeleted={() => navigate("/app/patients")}
        />
      )}
    </div>
  );
}
