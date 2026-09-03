import {
  Activity,
  ArrowLeft,
  Check,
  ChevronDown,
  FileImage,
  Info,
  Play,
  RotateCcw,
  Search,
  ShieldCheck,
  Upload,
  UserRound,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import api from "../../api/client";
import {
  getModelRegistry,
  type ModelField,
  type ModelRegistryEntry,
} from "../../api/inference";
import { getPatients, type Patient as ApiPatient } from "../../api/patients";

type Modality = "tabular" | "image" | "multimodal";
type ImageType = "xray" | "skin" | "blood_smear" | "histology";

const AUTO_MODEL_ID = "__auto__";

interface ClinicalField {
  key: string;
  label: string;
  value: string;
  type: "integer" | "float" | "string";
  required: boolean;
  options: { value: string; label: string }[];
}

const imageTypes: { value: ImageType; label: string; description: string }[] = [
  { value: "xray", label: "Chest X-Ray", description: "Chest radiography" },
  { value: "skin", label: "Skin", description: "Clinical skin photograph" },
  { value: "blood_smear", label: "Blood Smear", description: "Peripheral blood smear" },
  { value: "histology", label: "Histology", description: "Tissue histology image" },
];

function fieldLabel(field: ModelField) {
  if (field.description) return field.description;
  return field.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function flattenOptions(field: ModelField) {
  return (field.options ?? []).map((opt) => {
    const [value, label] = Object.entries(opt)[0];
    return { value, label };
  });
}

function toClinicalField(field: ModelField): ClinicalField {
  return {
    key: field.name,
    label: fieldLabel(field),
    value: "",
    type: field.type === "integer" || field.type === "float" ? field.type : "string",
    required: field.required,
    options: flattenOptions(field),
  };
}

function buildFieldsForModel(tabularModels: ModelRegistryEntry[], modelId: string): ClinicalField[] {
  if (modelId === AUTO_MODEL_ID) {
    const seen = new Map<string, ClinicalField>();
    for (const model of tabularModels) {
      for (const field of model.fields ?? []) {
        if (!seen.has(field.name)) {
          seen.set(field.name, toClinicalField(field));
        }
      }
    }
    return Array.from(seen.values());
  }

  const model = tabularModels.find((m) => m.model_id === modelId);
  return (model?.fields ?? []).map(toClinicalField);
}

function parseClinicalValue(value: string) {
  if (value.trim() === "") {
    return "";
  }

  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : value;
}

function calculateAge(dateOfBirth?: string) {
  if (!dateOfBirth) {
    return "Age unknown";
  }

  const birth = new Date(dateOfBirth);

  if (Number.isNaN(birth.getTime())) {
    return "Age unknown";
  }

  const today = new Date();

  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();

  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age -= 1;
  }

  return `${age} years`;
}

function getInitials(patient: ApiPatient) {
  return `${patient.first_name?.[0] ?? ""}${patient.last_name?.[0] ?? ""}`;
}

export function InferenceWorkspace() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const [modality, setModality] = useState<Modality>("tabular");

  const [registryModels, setRegistryModels] = useState<ModelRegistryEntry[]>([]);
  const [isLoadingRegistry, setIsLoadingRegistry] = useState(true);
  const [registryError, setRegistryError] = useState<string | null>(null);

  const [selectedTabularModelId, setSelectedTabularModelId] = useState(AUTO_MODEL_ID);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const [fields, setFields] = useState<ClinicalField[]>([]);

  const [notes, setNotes] = useState("");

  const [patients, setPatients] = useState<ApiPatient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState("");

  const [patientSearch, setPatientSearch] = useState("");
  const [showPatientMenu, setShowPatientMenu] = useState(false);

  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [selectedImageName, setSelectedImageName] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageType, setImageType] = useState<ImageType>("xray");

  const [isLoadingPatients, setIsLoadingPatients] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const patientMenuRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  const tabularModels = useMemo(
    () => registryModels.filter((m) => m.modality === "tabular"),
    [registryModels],
  );

  const selectedTabularModel = useMemo(
    () => tabularModels.find((m) => m.model_id === selectedTabularModelId) ?? null,
    [tabularModels, selectedTabularModelId],
  );

  const missingRequiredFields = useMemo(() => {
    if (selectedTabularModelId === AUTO_MODEL_ID) return [];
    return fields.filter((field) => field.required && field.value.trim() === "");
  }, [fields, selectedTabularModelId]);

  const selectedPatient = useMemo(
    () => patients.find((patient) => patient.patient_id === selectedPatientId) ?? null,
    [patients, selectedPatientId],
  );

  const filteredPatients = useMemo(() => {
    const query = patientSearch.trim().toLowerCase();

    if (!query) {
      return patients;
    }

    return patients.filter((patient) => {
      const name = patient.full_name ?? `${patient.first_name} ${patient.last_name}`;

      return (
        name.toLowerCase().includes(query) ||
        patient.patient_id.toLowerCase().includes(query) ||
        patient.email?.toLowerCase().includes(query)
      );
    });
  }, [patients, patientSearch]);

  useEffect(() => {
    let active = true;

    async function loadRegistry() {
      setIsLoadingRegistry(true);
      setRegistryError(null);

      try {
        const registry = await getModelRegistry();
        if (active) setRegistryModels(registry.models);
      } catch {
        if (active) {
          setRegistryError(
            "Unable to load the model registry. The ML service may be unavailable.",
          );
        }
      } finally {
        if (active) setIsLoadingRegistry(false);
      }
    }

    loadRegistry();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (isLoadingRegistry) return;
    setFields(buildFieldsForModel(tabularModels, selectedTabularModelId));
  }, [isLoadingRegistry, tabularModels, selectedTabularModelId]);

  useEffect(() => {
    let active = true;

    async function loadPatients() {
      try {
        const data = await getPatients();

        if (!active) {
          return;
        }

        setPatients(data);

        const patientParam = searchParams.get("patient");

        if (patientParam && data.some((patient) => patient.patient_id === patientParam)) {
          setSelectedPatientId(patientParam);
        }
      } catch {
        if (active) {
          setError(
            "Unable to load patient list. You can still run an anonymous inference.",
          );
        }
      } finally {
        if (active) {
          setIsLoadingPatients(false);
        }
      }
    }

    loadPatients();

    return () => {
      active = false;
    };
  }, [searchParams]);

  useEffect(() => {
    function handleOutsideClick(event: MouseEvent) {
      if (patientMenuRef.current && !patientMenuRef.current.contains(event.target as Node)) {
        setShowPatientMenu(false);
      }
      if (modelMenuRef.current && !modelMenuRef.current.contains(event.target as Node)) {
        setShowModelMenu(false);
      }
    }

    document.addEventListener("mousedown", handleOutsideClick);

    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview);
      }
    };
  }, [imagePreview]);

  const updateField = (key: string, value: string) => {
    setFields((current) =>
      current.map((field) => (field.key === key ? { ...field, value } : field)),
    );
  };

  const canRun =
    (modality === "tabular" &&
      !isLoadingRegistry &&
      fields.length > 0 &&
      missingRequiredFields.length === 0) ||
    (modality === "image" && !!selectedImageFile) ||
    (modality === "multimodal" &&
      !isLoadingRegistry &&
      fields.length > 0 &&
      missingRequiredFields.length === 0 &&
      !!selectedImageFile);

  const buildSymptomsPayload = () =>
    Object.fromEntries(
      fields
        .filter((field) => field.value.trim() !== "")
        .map((field) => [field.key, parseClinicalValue(field.value)]),
    );

  const handleSelectPatient = (patient: ApiPatient) => {
    setSelectedPatientId(patient.patient_id);
    setPatientSearch("");
    setShowPatientMenu(false);

    setSearchParams({
      patient: patient.patient_id,
    });
  };

  const handleAnonymousPatient = () => {
    setSelectedPatientId("");
    setPatientSearch("");
    setShowPatientMenu(false);

    setSearchParams({});
  };

  const handleImageChange = (file: File | null) => {
    if (!file) {
      return;
    }

    if (!["image/jpeg", "image/png", "image/jpg"].includes(file.type)) {
      setError("Only JPEG and PNG images are accepted.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("Image is too large. Maximum size is 10MB.");
      return;
    }

    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }

    setSelectedImageFile(file);
    setSelectedImageName(file.name);
    setImagePreview(URL.createObjectURL(file));
    setError(null);
  };

  const clearImage = () => {
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }

    setImagePreview(null);
    setSelectedImageFile(null);
    setSelectedImageName(null);
  };

  const handleRunInference = async () => {
    if (!canRun) {
      setError("Please complete the required clinical and image inputs.");
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      const symptoms = buildSymptomsPayload();

      if (modality === "tabular") {
        const response = await api.post("/inference/symptoms/", {
          patient_id: selectedPatientId || null,
          clinical_notes: notes,
          symptoms,
        });

        const inferenceId = Number(response.data.id);

        if (!Number.isFinite(inferenceId)) {
          throw new Error("Invalid inference id returned by the server.");
        }

        if (selectedPatientId) {
          setSearchParams({ patient: selectedPatientId });
        } else {
          setSearchParams({});
        }

        navigate(`/app/inference/${inferenceId}`);
        return;
      }

      const formData = new FormData();

      if (selectedPatientId) {
        formData.append("patient_id", selectedPatientId);
      }

      formData.append("clinical_notes", notes);
      formData.append("image_type", imageType);
      formData.append("symptoms", JSON.stringify(symptoms));

      if (selectedImageFile) {
        formData.append("file", selectedImageFile);
      }

      const response = await api.post("/inference/image/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const inferenceId = Number(response.data.id);

      if (!Number.isFinite(inferenceId)) {
        throw new Error("Invalid inference id returned by the server.");
      }

      if (selectedPatientId) {
        setSearchParams({ patient: selectedPatientId });
      } else {
        setSearchParams({});
      }

      navigate(`/app/inference/${inferenceId}`);
    } catch (requestError) {
      const maybeError = requestError as {
        response?: {
          data?: {
            detail?: string;
            non_field_errors?: string | string[];
            symptoms?: string | string[];
            patient_id?: string | string[];
            [key: string]: unknown;
          };
        };
        message?: string;
      };

      const errorPayload = maybeError.response?.data;

      const detail = errorPayload?.detail;
      const nonFieldErrors = errorPayload?.non_field_errors;
      const symptomErrors = errorPayload?.symptoms;
      const patientErrors = errorPayload?.patient_id;

      const message =
        (typeof detail === "string" && detail) ||
        (Array.isArray(nonFieldErrors) ? nonFieldErrors.join(" ") : nonFieldErrors) ||
        (Array.isArray(symptomErrors) ? symptomErrors.join(" ") : symptomErrors) ||
        (Array.isArray(patientErrors) ? patientErrors.join(" ") : patientErrors) ||
        maybeError.message ||
        "Unable to submit inference. Please verify the inputs and try again.";

      setError(
        typeof message === "string"
          ? message
          : "Unable to submit inference. Please verify the inputs and try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setSelectedTabularModelId(AUTO_MODEL_ID);
    setFields(buildFieldsForModel(tabularModels, AUTO_MODEL_ID));
    setNotes("");
    clearImage();
    setImageType("xray");
    setError(null);
  };

  return (
    <div className="mx-auto max-w-360 px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
      <div className="mb-6">
        <button
          type="button"
          onClick={() => navigate("/app/patients")}
          className="mb-5 flex items-center gap-2 text-[9px] font-medium text-gray-400 hover:text-gray-800"
        >
          <ArrowLeft size={12} />
          Back
        </button>

        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-blue-600">
              <Activity size={11} />
              AI Inference
            </div>

            <h1 className="mt-2 text-[25px] font-semibold tracking-tight text-gray-950 sm:text-[29px]">
              New inference
            </h1>

            <p className="mt-1 max-w-xl text-[10px] leading-5 text-gray-400">
              Provide clinical inputs and imaging data for AI-assisted assessment.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-green-50">
              <ShieldCheck size={12} className="text-green-600" />
            </div>

            <div>
              <div className="text-[8px] font-semibold text-gray-700">Inference engine ready</div>
              <div className="mt-0.5 text-[7px] text-gray-400">All systems operational</div>
            </div>
          </div>
        </div>
      </div>

      <section className="mb-5 grid gap-4 lg:grid-cols-[1fr_1.4fr]">
        {/* Patient */}
        <div ref={patientMenuRef} className="relative rounded-xl border border-gray-200 bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">Patient</div>
              <div className="mt-1 text-[7px] text-gray-300">Optional</div>
            </div>

            {selectedPatient && (
              <button
                type="button"
                onClick={handleAnonymousPatient}
                className="text-[8px] font-medium text-gray-400 hover:text-gray-800"
              >
                Clear
              </button>
            )}
          </div>

          {selectedPatient ? (
            <button
              type="button"
              onClick={() => setShowPatientMenu((value) => !value)}
              className="flex w-full items-center gap-3 text-left"
            >
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[9px] font-semibold text-gray-500">
                {getInitials(selectedPatient)}
              </div>

              <div className="min-w-0 flex-1">
                <div className="text-[10px] font-semibold text-gray-800">
                  {selectedPatient.full_name ?? `${selectedPatient.first_name} ${selectedPatient.last_name}`}
                </div>

                <div className="mt-1 font-mono text-[7px] text-gray-400">
                  {selectedPatient.patient_id} · {selectedPatient.sex ?? "Unknown"} ·{" "}
                  {calculateAge(selectedPatient.date_of_birth)}
                </div>
              </div>

              <ChevronDown size={13} className="shrink-0 text-gray-400" />
            </button>
          ) : (
            <button
              type="button"
              onClick={() => setShowPatientMenu((value) => !value)}
              className="flex w-full items-center gap-3 text-left"
            >
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gray-100">
                <UserRound size={14} className="text-gray-400" />
              </div>

              <div className="min-w-0 flex-1">
                <div className="text-[10px] font-semibold text-gray-700">Anonymous patient</div>
                <div className="mt-1 text-[7px] text-gray-400">No patient record will be attached</div>
              </div>

              <ChevronDown size={13} className="shrink-0 text-gray-400" />
            </button>
          )}

          {showPatientMenu && (
            <div className="absolute left-4 right-4 top-21.5 z-30 overflow-hidden rounded-lg border border-gray-200 bg-white shadow-lg">
              <div className="border-b border-gray-100 p-2">
                <div className="relative">
                  <Search size={11} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300" />

                  <input
                    type="text"
                    value={patientSearch}
                    onChange={(event) => setPatientSearch(event.target.value)}
                    placeholder="Search by name or patient ID..."
                    autoFocus
                    className="h-8 w-full rounded-md border border-gray-200 bg-gray-50 pl-8 pr-3 text-[8px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
                  />
                </div>
              </div>

              <button
                type="button"
                onClick={handleAnonymousPatient}
                className="flex w-full items-center gap-3 border-b border-gray-100 px-3 py-3 text-left hover:bg-gray-50"
              >
                <div className="flex h-7 w-7 items-center justify-center rounded-md bg-gray-100">
                  <UserRound size={11} className="text-gray-400" />
                </div>

                <div>
                  <div className="text-[8px] font-semibold text-gray-700">Anonymous patient</div>
                  <div className="mt-0.5 text-[7px] text-gray-400">Run without attaching a patient</div>
                </div>

                {!selectedPatient && <Check size={11} className="ml-auto text-gray-500" />}
              </button>

              <div className="max-h-64 overflow-y-auto">
                {isLoadingPatients ? (
                  <div className="px-3 py-4 text-[8px] text-gray-400">Loading patients…</div>
                ) : filteredPatients.length === 0 ? (
                  <div className="px-3 py-4 text-[8px] text-gray-400">No matching patients.</div>
                ) : (
                  filteredPatients.map((patient) => {
                    const name = patient.full_name ?? `${patient.first_name} ${patient.last_name}`;
                    const isSelected = patient.patient_id === selectedPatientId;

                    return (
                      <button
                        key={patient.patient_id}
                        type="button"
                        onClick={() => handleSelectPatient(patient)}
                        className="flex w-full items-center gap-3 border-b border-gray-50 px-3 py-3 text-left last:border-0 hover:bg-gray-50"
                      >
                        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[7px] font-semibold text-gray-500">
                          {getInitials(patient)}
                        </div>

                        <div className="min-w-0 flex-1">
                          <div className="truncate text-[8px] font-semibold text-gray-700">{name}</div>

                          <div className="mt-0.5 truncate font-mono text-[6px] text-gray-400">
                            {patient.patient_id} · {calculateAge(patient.date_of_birth)}
                          </div>
                        </div>

                        {isSelected && <Check size={11} className="shrink-0 text-gray-500" />}
                      </button>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </div>

        {/* Tabular model selection */}
        <div ref={modelMenuRef} className="relative rounded-xl border border-gray-200 bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-[8px] font-semibold uppercase tracking-wide text-gray-400">
              {modality === "image" ? "Image routing" : "Tabular model"}
            </div>

            {modality !== "image" && (
              <span className="rounded-md bg-gray-50 px-2 py-1 font-mono text-[7px] text-gray-400">
                {tabularModels.length} available
              </span>
            )}
          </div>

          {modality === "image" ? (
            <p className="text-[9px] leading-5 text-gray-500">
              Image models are selected automatically based on the image type you choose below.
            </p>
          ) : isLoadingRegistry ? (
            <p className="text-[9px] text-gray-400">Loading available models…</p>
          ) : registryError ? (
            <p className="text-[9px] text-red-500">{registryError}</p>
          ) : (
            <>
              <button
                type="button"
                onClick={() => setShowModelMenu((value) => !value)}
                className="flex w-full items-center justify-between text-left"
              >
                <div className="min-w-0">
                  <div className="text-[10px] font-semibold text-gray-800">
                    {selectedTabularModelId === AUTO_MODEL_ID
                      ? "Automatic (route to all applicable models)"
                      : selectedTabularModel?.model_name ?? "Automatic"}
                  </div>

                  <div className="mt-1 truncate text-[8px] text-gray-400">
                    {selectedTabularModelId === AUTO_MODEL_ID
                      ? "Runs every model with enough data provided"
                      : "Only fields for this model are shown below"}
                  </div>
                </div>

                <ChevronDown size={13} className="ml-3 shrink-0 text-gray-400" />
              </button>

              {showModelMenu && (
                <div className="absolute left-4 right-4 top-18.5 z-20 max-h-72 overflow-y-auto rounded-lg border border-gray-200 bg-white shadow-lg">
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedTabularModelId(AUTO_MODEL_ID);
                      setShowModelMenu(false);
                    }}
                    className="flex w-full items-start gap-3 border-b border-gray-100 p-3 text-left hover:bg-gray-50"
                  >
                    <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-gray-100">
                      {selectedTabularModelId === AUTO_MODEL_ID ? (
                        <Check size={11} />
                      ) : (
                        <Activity size={11} />
                      )}
                    </div>

                    <div className="min-w-0">
                      <div className="text-[9px] font-semibold text-gray-700">Automatic</div>
                      <div className="mt-1 text-[7px] text-gray-400">
                        Route to every tabular model with enough data
                      </div>
                    </div>
                  </button>

                  {tabularModels.map((model) => (
                    <button
                      key={model.model_id}
                      type="button"
                      onClick={() => {
                        setSelectedTabularModelId(model.model_id);
                        setShowModelMenu(false);
                      }}
                      className="flex w-full items-start gap-3 border-b border-gray-100 p-3 text-left last:border-0 hover:bg-gray-50"
                    >
                      <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-gray-100">
                        {selectedTabularModelId === model.model_id ? (
                          <Check size={11} />
                        ) : (
                          <Activity size={11} />
                        )}
                      </div>

                      <div className="min-w-0">
                        <div className="text-[9px] font-semibold text-gray-700">{model.model_name}</div>
                        <div className="mt-1 text-[7px] text-gray-400">{model.task}</div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </section>

      <section className="mb-5 rounded-xl border border-gray-200 bg-white p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="text-[9px] font-semibold text-gray-800">Input modality</div>

            <p className="mt-1 text-[8px] text-gray-400">
              Select the type of data you want to send to the unified inference engine.
            </p>
          </div>

          <div className="grid grid-cols-3 rounded-lg bg-gray-100 p-1">
            {[
              ["tabular", "Tabular"],
              ["image", "Image"],
              ["multimodal", "Multimodal"],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setModality(value as Modality)}
                className={[
                  "rounded-md px-3 py-2 text-[8px] font-semibold transition",
                  modality === value ? "bg-white text-gray-900 shadow-sm" : "text-gray-400 hover:text-gray-700",
                ].join(" ")}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-[1fr_1fr]">
        {(modality === "tabular" || modality === "multimodal") && (
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 p-5">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-[11px] font-semibold text-gray-900">Clinical data</h2>

                  <p className="mt-1 text-[8px] text-gray-400">
                    {selectedTabularModelId === AUTO_MODEL_ID
                      ? "Structured variables sent to the unified symptoms endpoint."
                      : `Fields required by ${selectedTabularModel?.model_name ?? "this model"}.`}
                  </p>
                </div>

                <span className="rounded-md bg-gray-50 px-2 py-1 font-mono text-[7px] text-gray-400">
                  {fields.length} variables
                </span>
              </div>
            </div>

            {isLoadingRegistry ? (
              <div className="p-5 text-[9px] text-gray-400">Loading clinical intake fields…</div>
            ) : fields.length === 0 ? (
              <div className="p-5 text-[9px] text-gray-400">No fields available for this model.</div>
            ) : (
              <div className="grid gap-x-4 gap-y-4 p-5 sm:grid-cols-2">
                {fields.map((field) => (
                  <label key={field.key}>
                    <div className="mb-1.5 flex items-center justify-between">
                      <span className="text-[8px] font-medium text-gray-600">
                        {field.label}
                        {field.required && <span className="ml-1 text-red-400">*</span>}
                      </span>
                    </div>

                    {field.options.length > 0 ? (
                      <select
                        value={field.value}
                        onChange={(event) => updateField(field.key, event.target.value)}
                        className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
                      >
                        <option value="">Select...</option>
                        {field.options.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={field.type === "string" ? "text" : "number"}
                        inputMode={
                          field.type === "integer" ? "numeric" : field.type === "float" ? "decimal" : "text"
                        }
                        step={field.type === "integer" ? 1 : field.type === "float" ? "any" : undefined}
                        value={field.value}
                        onChange={(event) => updateField(field.key, event.target.value)}
                        className="h-9 w-full rounded-lg border border-gray-200 bg-gray-50/50 px-3 text-[9px] text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
                      />
                    )}
                  </label>
                ))}
              </div>
            )}

            {missingRequiredFields.length > 0 && (
              <div className="mx-5 mb-5 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[8px] leading-4 text-amber-700">
                Fill in the required fields for {selectedTabularModel?.model_name}:{" "}
                {missingRequiredFields.map((f) => f.label).join(", ")}
              </div>
            )}

            <div className="border-t border-gray-100 p-5">
              <label>
                <div className="mb-1.5 text-[8px] font-medium text-gray-600">Clinical notes</div>

                <textarea
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                  rows={4}
                  className="w-full resize-none rounded-lg border border-gray-200 bg-gray-50/50 p-3 text-[9px] leading-5 text-gray-700 outline-none focus:border-gray-400 focus:bg-white"
                  placeholder="Add relevant clinical context..."
                />
              </label>
            </div>
          </section>
        )}

        {(modality === "image" || modality === "multimodal") && (
          <section className="rounded-xl border border-gray-200 bg-white">
            <div className="border-b border-gray-100 p-5">
              <div>
                <h2 className="text-[11px] font-semibold text-gray-900">Medical imaging</h2>

                <p className="mt-1 text-[8px] text-gray-400">
                  Upload an image and select its clinical type.
                </p>
              </div>
            </div>

            <div className="p-5">
              <div className="mb-4">
                <div className="mb-2 text-[8px] font-medium text-gray-600">Image type</div>

                <div className="grid grid-cols-2 gap-2">
                  {imageTypes.map((item) => (
                    <button
                      key={item.value}
                      type="button"
                      onClick={() => setImageType(item.value)}
                      className={[
                        "rounded-lg border p-3 text-left transition",
                        imageType === item.value ? "border-gray-400 bg-gray-50" : "border-gray-200 hover:bg-gray-50",
                      ].join(" ")}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-[8px] font-semibold text-gray-700">{item.label}</span>

                        {imageType === item.value && <Check size={10} />}
                      </div>

                      <div className="mt-1 text-[7px] text-gray-400">{item.description}</div>
                    </button>
                  ))}
                </div>
              </div>

              {imagePreview ? (
                <div className="relative overflow-hidden rounded-xl border border-gray-200 bg-gray-950">
                  <div className="aspect-4/3">
                    <img
                      src={imagePreview}
                      alt="Medical upload preview"
                      className="h-full w-full object-cover"
                    />
                  </div>

                  <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between bg-gray-950/90 px-3 py-2">
                    <div className="flex items-center gap-2">
                      <FileImage size={11} className="text-gray-400" />

                      <span className="text-[7px] text-gray-300">
                        {selectedImageName ?? "uploaded_image.png"}
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={clearImage}
                      className="flex h-6 w-6 items-center justify-center rounded-md text-gray-400 hover:bg-white/10 hover:text-white"
                    >
                      <X size={11} />
                    </button>
                  </div>
                </div>
              ) : (
                <label className="flex aspect-4/3 w-full cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-gray-300 bg-gray-50/50 text-center hover:border-gray-400 hover:bg-gray-50">
                  <input
                    type="file"
                    accept="image/jpeg,image/png"
                    onChange={(event) => {
                      handleImageChange(event.target.files?.[0] ?? null);
                      event.target.value = "";
                    }}
                    className="hidden"
                  />

                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white shadow-sm">
                    <Upload size={16} className="text-gray-400" />
                  </div>

                  <div className="mt-4 text-[9px] font-semibold text-gray-700">Upload medical image</div>

                  <div className="mt-1 text-[7px] text-gray-400">JPEG or PNG · maximum 10MB</div>
                </label>
              )}

              <div className="mt-4 flex gap-3 rounded-lg bg-gray-50 p-3">
                <Info size={12} className="mt-0.5 shrink-0 text-gray-400" />

                <p className="text-[7px] leading-4 text-gray-400">
                  Ensure the image is clear and correctly oriented. Image quality can affect model
                  performance.
                </p>
              </div>
            </div>
          </section>
        )}
      </div>

      <section className="mt-5 rounded-xl border border-gray-200 bg-white p-4 sm:p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-100">
              <Activity size={13} className="text-gray-500" />
            </div>

            <div>
              <div className="text-[9px] font-semibold text-gray-800">Ready to run</div>

              <p className="mt-1 max-w-lg text-[8px] leading-4 text-gray-400">
                {selectedPatient ? (
                  <>
                    Results will be attached to{" "}
                    <span className="font-medium text-gray-600">
                      {selectedPatient.full_name ?? `${selectedPatient.first_name} ${selectedPatient.last_name}`}
                    </span>
                    .
                  </>
                ) : (
                  <>
                    This inference will run as an{" "}
                    <span className="font-medium text-gray-600">anonymous patient</span> without a patient
                    record.
                  </>
                )}
              </p>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleReset}
              className="flex h-9 items-center justify-center gap-2 rounded-lg border border-gray-200 px-3 text-[8px] font-medium text-gray-500 hover:bg-gray-50"
            >
              <RotateCcw size={11} />
              Reset
            </button>

            <button
              type="button"
              disabled={!canRun || isSubmitting}
              onClick={handleRunInference}
              className="flex h-9 items-center justify-center gap-2 rounded-lg bg-gray-950 px-4 text-[8px] font-semibold text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Play size={11} />
              {isSubmitting ? "Submitting..." : "Run inference"}
            </button>
          </div>
        </div>

        {error ? (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[8px] text-red-700">
            {error}
          </div>
        ) : null}
      </section>

      <div className="mt-4 flex items-center justify-center gap-2 px-4 text-center">
        <UserRound size={10} className="text-gray-300" />

        <p className="text-[7px] leading-4 text-gray-400">
          AI-generated results are decision-support information and must be reviewed by a qualified
          healthcare professional.
        </p>
      </div>
    </div>
  );
}