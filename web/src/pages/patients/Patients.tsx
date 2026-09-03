import {
  Activity,
  ChevronRight,
  CircleUserRound,
  Plus,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  createPatient,
  getPatients,
  type CreatePatientPayload,
  type Patient as ApiPatient,
} from "../../api/patients";

type PatientStatus = "Active" | "Needs review" | "Inactive";

function formatDate(dateValue?: string | null) {
  if (!dateValue) return "—";
  const parsedDate = new Date(dateValue);
  if (Number.isNaN(parsedDate.getTime())) return "—";

  return parsedDate.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function getPageNumbers(
  current: number,
  total: number,
): (number | "ellipsis")[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }

  const pages: (number | "ellipsis")[] = [1];

  if (current > 3) {
    pages.push("ellipsis");
  }

  const start = Math.max(2, current - 1);
  const end = Math.min(total - 1, current + 1);

  for (let i = start; i <= end; i += 1) {
    pages.push(i);
  }

  if (current < total - 2) {
    pages.push("ellipsis");
  }

  pages.push(total);

  return pages;
}

const PATIENTS_PER_PAGE = 10;

function StatusBadge({ status }: { status: PatientStatus }) {
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

export function Patients() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState<ApiPatient[]>([]);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<"All" | PatientStatus>("All");
  const [showFilters, setShowFilters] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formData, setFormData] = useState<CreatePatientPayload>({
    first_name: "",
    last_name: "",
    date_of_birth: "",
    sex: "",
    phone_number: "",
    email: "",
    notes: "",
  });
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadPatients() {
      try {
        const data = await getPatients();
        if (active) {
          setPatients(data);
        }
      } catch {
        if (active) {
          setError("Unable to load patients. Please try again.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadPatients();

    return () => {
      active = false;
    };
  }, []);

  const filteredPatients = useMemo(() => {
    const query = search.trim().toLowerCase();

    return patients.filter((patient) => {
      const fullName =
        patient.full_name ?? `${patient.first_name} ${patient.last_name}`;
      const matchesSearch =
        !query ||
        fullName.toLowerCase().includes(query) ||
        patient.patient_id.toLowerCase().includes(query) ||
        (patient.phone_number ?? "").toLowerCase().includes(query);

      const matchesStatus = status === "All" || status === "Active";

      return matchesSearch && matchesStatus;
    });
  }, [patients, search, status]);

  const totalPages = Math.max(
    1,
    Math.ceil(filteredPatients.length / PATIENTS_PER_PAGE),
  );

  const paginatedPatients = useMemo(
    () =>
      filteredPatients.slice(
        (page - 1) * PATIENTS_PER_PAGE,
        page * PATIENTS_PER_PAGE,
      ),
    [filteredPatients, page],
  );

  // Reset to page 1 whenever the filters change the result set
  useEffect(() => {
    setPage(1);
  }, [search, status]);

  // Clamp page if it becomes out of range (e.g. after a filter shrinks the list)
  useEffect(() => {
    setPage((current) => Math.min(current, totalPages));
  }, [totalPages]);

  if (loading) {
    return <div className="p-8 text-sm text-gray-500">Loading patients…</div>;
  }

  if (error) {
    return <div className="p-8 text-sm text-red-600">{error}</div>;
  }

  const handleAddPatient = async () => {
    setFormError(null);

    if (!formData.first_name.trim() || !formData.last_name.trim()) {
      setFormError("First and last names are required.");
      return;
    }

    if (!formData.date_of_birth) {
      setFormError("Date of birth is required.");
      return;
    }

    if (!formData.sex) {
      setFormError("Sex is required.");
      return;
    }

    setIsSubmitting(true);
    try {
      const payload: CreatePatientPayload = {
        first_name: formData.first_name.trim(),
        last_name: formData.last_name.trim(),
        date_of_birth: formData.date_of_birth,
        sex: formData.sex,
        ...(formData.phone_number?.trim() && {
          phone_number: formData.phone_number.trim(),
        }),
        ...(formData.email?.trim() && { email: formData.email.trim() }),
        ...(formData.notes?.trim() && { notes: formData.notes.trim() }),
      };

      const newPatient = await createPatient(payload);
      setPatients((current) => [...current, newPatient]);
      setShowAddModal(false);
      setFormData({
        first_name: "",
        last_name: "",
        date_of_birth: "",
        sex: "",
        phone_number: "",
        email: "",
        notes: "",
      });
    } catch (err) {
      const maybeError = err as {
        response?: {
          data?: {
            detail?: string;
            non_field_errors?: unknown;
            [key: string]: unknown;
          };
        };
      };
      const message =
        maybeError.response?.data?.detail ??
        "Unable to create patient. Please try again.";
      setFormError(
        typeof message === "string"
          ? message
          : "Unable to create patient. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      <div className="mb-7 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-blue-600">
            <CircleUserRound size={11} />
            Patient Management
          </div>

          <h1 className="mt-2 text-[25px] font-semibold tracking-tight text-gray-950 sm:text-[29px]">
            Patients
          </h1>

          <p className="mt-1 max-w-xl text-[10px] leading-5 text-gray-400">
            Manage patient records and access their clinical and AI inference
            history.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setShowAddModal(true)}
          className="flex h-9 items-center justify-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[10px] font-semibold text-white hover:bg-gray-800"
        >
          <Plus size={13} />
          Add patient
        </button>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Total patients
          </div>

          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            {patients.length}
          </div>

          <div className="mt-1 text-[8px] text-gray-400">
            Registered patients
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Active
          </div>

          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            {patients.length}
          </div>

          <div className="mt-1 text-[8px] text-green-600">
            Available in the system
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Requires review
          </div>

          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            0
          </div>

          <div className="mt-1 text-[8px] text-amber-600">
            AI results awaiting review
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
            Last updated
          </div>

          <div className="mt-2 text-[21px] font-semibold tracking-tight text-gray-950">
            {formatDate(patients[0]?.updated_at ?? patients[0]?.created_at) ||
              "—"}
          </div>

          <div className="mt-1 text-[8px] text-gray-400">
            Latest patient activity
          </div>
        </div>
      </div>

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
              placeholder="Search name, patient ID, phone..."
              className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 pl-9 pr-3 text-[9px] text-gray-700 outline-none placeholder:text-gray-300 focus:border-gray-300 focus:bg-white"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setShowFilters((value) => !value)}
              className={[
                "flex h-9 items-center gap-2 rounded-lg border px-3 text-[9px] font-medium",
                showFilters || status !== "All"
                  ? "border-gray-300 bg-gray-50 text-gray-900"
                  : "border-gray-200 text-gray-500 hover:bg-gray-50",
              ].join(" ")}
            >
              <SlidersHorizontal size={12} />
              Filters
            </button>
          </div>
        </div>

        {showFilters && (
          <div className="border-b border-gray-100 bg-gray-50/50 p-4">
            <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
              Patient status
            </div>

            <div className="mt-2 flex flex-wrap gap-1.5">
              {(["All", "Active", "Needs review", "Inactive"] as const).map(
                (item) => (
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
                ),
              )}
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
                  Contact
                </th>

                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Status
                </th>

                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Last updated
                </th>

                <th className="px-4 py-3 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                  Latest AI result
                </th>

                <th className="w-10 px-3 py-3" />
              </tr>
            </thead>

            <tbody className="divide-y divide-gray-100">
              {paginatedPatients.map((patient) => {
                const fullName =
                  patient.full_name ??
                  `${patient.first_name} ${patient.last_name}`;

                return (
                  <tr
                    key={patient.patient_id}
                    onClick={() =>
                      navigate(`/app/patients/${patient.patient_id}`)
                    }
                    className="group cursor-pointer hover:bg-gray-50/70"
                  >
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[8px] font-semibold text-gray-500">
                          {fullName
                            .split(" ")
                            .map((part) => part[0])
                            .join("")
                            .slice(0, 2)}
                        </div>

                        <div>
                          <div className="text-[10px] font-semibold text-gray-800">
                            {fullName}
                          </div>
                          <div className="mt-1 font-mono text-[8px] text-gray-400">
                            {patient.patient_id} · {patient.sex ?? "Unknown"}
                          </div>
                        </div>
                      </div>
                    </td>

                    <td className="px-4 py-3.5">
                      <span className="text-[9px] text-gray-600">
                        {patient.phone_number || "No phone on file"}
                      </span>
                    </td>

                    <td className="px-4 py-3.5">
                      <StatusBadge status={"Active"} />
                    </td>

                    <td className="px-4 py-3.5">
                      <span className="text-[9px] text-gray-600">
                        {formatDate(patient.updated_at ?? patient.created_at)}
                      </span>
                    </td>

                    <td className="px-4 py-3.5">
                      <div className="flex items-center gap-2">
                        <Activity size={11} className="text-gray-300" />
                        <div>
                          <div className="text-[9px] font-medium text-gray-700">
                            No inference yet
                          </div>
                          <div className="mt-1 font-mono text-[7px] text-gray-400">
                            —
                          </div>
                        </div>
                      </div>
                    </td>

                    <td className="px-3 py-3.5">
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          navigate(`/app/patients/${patient.patient_id}`);
                        }}
                        className="flex h-7 w-7 items-center justify-center rounded-md text-gray-300 opacity-0 transition group-hover:opacity-100 hover:bg-gray-100 hover:text-gray-700"
                        aria-label="View patient"
                      >
                        <ChevronRight size={13} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="divide-y divide-gray-100 lg:hidden">
          {paginatedPatients.map((patient) => {
            const fullName =
              patient.full_name ?? `${patient.first_name} ${patient.last_name}`;

            return (
              <button
                key={patient.patient_id}
                type="button"
                onClick={() => navigate(`/app/patients/${patient.patient_id}`)}
                className="block w-full p-4 text-left hover:bg-gray-50"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[8px] font-semibold text-gray-500">
                      {fullName
                        .split(" ")
                        .map((part) => part[0])
                        .join("")
                        .slice(0, 2)}
                    </div>

                    <div className="min-w-0">
                      <div className="truncate text-[10px] font-semibold text-gray-800">
                        {fullName}
                      </div>
                      <div className="mt-1 font-mono text-[7px] text-gray-400">
                        {patient.patient_id} · {patient.sex ?? "Unknown"}
                      </div>
                    </div>
                  </div>

                  <StatusBadge status={"Active"} />
                </div>

                <div className="mt-4 rounded-lg bg-gray-50 p-3">
                  <div className="flex items-center gap-2">
                    <Activity size={11} className="text-gray-400" />
                    <span className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
                      Latest AI result
                    </span>
                  </div>

                  <div className="mt-2 flex items-center justify-between gap-3">
                    <span className="text-[9px] font-semibold text-gray-700">
                      No inference yet
                    </span>
                    <span className="font-mono text-[8px] text-gray-500">
                      —
                    </span>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between">
                  <span className="text-[8px] text-gray-400">
                    Last update:{" "}
                    {formatDate(patient.updated_at ?? patient.created_at)}
                  </span>
                  <ChevronRight size={12} className="text-gray-300" />
                </div>
              </button>
            );
          })}
        </div>

        {paginatedPatients.length === 0 && (
          <div className="flex flex-col items-center justify-center px-5 py-16 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gray-100">
              <Search size={16} className="text-gray-400" />
            </div>

            <div className="mt-4 text-[11px] font-semibold text-gray-700">
              No patients found
            </div>

            <p className="mt-1 text-[9px] text-gray-400">
              Try changing your search or filters.
            </p>
          </div>
        )}

        <div className="flex flex-col gap-3 border-t border-gray-100 px-5 py-3 sm:flex-row sm:items-center sm:justify-between">
          <span className="text-[8px] text-gray-400">
            {filteredPatients.length === 0
              ? "Showing 0 patients"
              : `Showing ${(page - 1) * PATIENTS_PER_PAGE + 1}-${Math.min(
                  page * PATIENTS_PER_PAGE,
                  filteredPatients.length,
                )} of ${filteredPatients.length} patients`}
          </span>

          {totalPages > 1 && (
            <div className="flex items-center gap-1">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                className="rounded-md border border-gray-200 px-2.5 py-1.5 text-[8px] text-gray-500 hover:enabled:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300"
              >
                Previous
              </button>

              {getPageNumbers(page, totalPages).map((item, index) =>
                item === "ellipsis" ? (
                  <span
                    key={`ellipsis-${index}`}
                    className="px-1.5 text-[8px] text-gray-300"
                  >
                    …
                  </span>
                ) : (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setPage(item)}
                    className={[
                      "rounded-md px-2.5 py-1.5 text-[8px] font-medium",
                      item === page
                        ? "bg-gray-900 text-white"
                        : "border border-gray-200 text-gray-500 hover:bg-gray-50",
                    ].join(" ")}
                  >
                    {item}
                  </button>
                ),
              )}

              <button
                type="button"
                disabled={page >= totalPages}
                onClick={() =>
                  setPage((current) => Math.min(totalPages, current + 1))
                }
                className="rounded-md border border-gray-200 px-2.5 py-1.5 text-[8px] text-gray-500 hover:enabled:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300"
              >
                Next
              </button>
            </div>
          )}
        </div>
      </section>

      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="mx-4 w-full max-w-md rounded-xl border border-gray-200 bg-white shadow-xl">
            <div className="flex items-center justify-between border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">
                  Add new patient
                </h2>
                <p className="mt-1 text-[8px] text-gray-400">
                  Create a new patient record in the system.
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setShowAddModal(false);
                  setFormError(null);
                }}
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
                    placeholder="e.g., Ama"
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
                    placeholder="e.g., Mensah"
                    className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
                  />
                </label>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <label>
                  <div className="mb-1 text-[8px] font-medium text-gray-600">
                    Date of birth *
                  </div>
                  <input
                    type="date"
                    value={formData.date_of_birth || ""}
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
                    Sex *
                  </div>
                  <select
                    value={formData.sex || ""}
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

              <div>
                <label>
                  <div className="mb-1 text-[8px] font-medium text-gray-600">
                    Phone number
                  </div>
                  <input
                    type="tel"
                    value={formData.phone_number || ""}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        phone_number: e.target.value,
                      }))
                    }
                    placeholder="e.g., +233 24 381 9204"
                    className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
                  />
                </label>
              </div>

              <div>
                <label>
                  <div className="mb-1 text-[8px] font-medium text-gray-600">
                    Email
                  </div>
                  <input
                    type="email"
                    value={formData.email || ""}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        email: e.target.value,
                      }))
                    }
                    placeholder="e.g., ama.mensah@example.com"
                    className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
                  />
                </label>
              </div>

              <div>
                <label>
                  <div className="mb-1 text-[8px] font-medium text-gray-600">
                    Clinical notes
                  </div>
                  <textarea
                    value={formData.notes || ""}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        notes: e.target.value,
                      }))
                    }
                    placeholder="Add any relevant clinical notes..."
                    rows={3}
                    className="w-full resize-none rounded-lg border border-gray-200 bg-gray-50/50 p-2.5 text-[9px] leading-4 text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
                  />
                </label>
              </div>

              {formError && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[8px] text-red-700">
                  {formError}
                </div>
              )}
            </div>

            <div className="flex items-center justify-end gap-2 border-t border-gray-100 p-4">
              <button
                type="button"
                onClick={() => {
                  setShowAddModal(false);
                  setFormError(null);
                }}
                className="h-9 rounded-lg border border-gray-200 px-4 text-[8px] font-medium text-gray-600 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleAddPatient}
                disabled={isSubmitting}
                className="h-9 rounded-lg bg-gray-950 px-4 text-[8px] font-semibold text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isSubmitting ? "Creating..." : "Add patient"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
