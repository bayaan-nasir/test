import api from "./client";

export interface Patient {
  id: number;
  patient_id: string;
  first_name: string;
  last_name: string;
  full_name?: string;
  date_of_birth?: string;
  sex?: string;
  phone_number?: string;
  email?: string;
  notes?: string;
  created_by?: number;
  created_at?: string;
  updated_at?: string;
}

export interface AIModel {
  id: number;
  name: string;
  version: string;
  description?: string;
  input_types: string[];
  task?: string;
}

export type InferenceInputType = "TABULAR" | "IMAGE" | "MULTIMODAL";

export interface InferenceEvent {
  id: number;
  event_type: string;
  message: string;
  created_by: number | null;
  created_by_name: string | null;
  created_at: string;
}

export interface InferenceFile {
  id: number;
  file: string;
  file_type: string;
  original_name: string;
  mime_type: string;
  size: number;
  created_at: string;
}

export interface Inference {
  id: number;
  patient: number;
  patient_name: string;
  requested_by: number;
  model: number;
  model_name: string;
  model_version: string;
  input_type: InferenceInputType;
  input_data: Record<string, unknown>;
  output_data: Record<string, unknown>;
  confidence: number | null;
  status: string;
  error_message: string;
  input_files: InferenceFile[];
  events: InferenceEvent[];
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface CreateInferencePayload {
  patient: number;
  model: number;
  input_type: InferenceInputType;
  input_data: Record<string, unknown>;
}

export async function getPatients() {
  const response = await api.get<Patient[]>("/patients/");
  console.log(response.data);

  return response.data;
}

export async function getModels() {
  const response = await api.get<AIModel[]>("/inference/models/");

  return response.data;
}

export async function createInference(payload: CreateInferencePayload) {
  const response = await api.post<Inference>("/inference/", payload);

  return response.data;
}

export async function uploadInferenceFile(inferenceId: number, file: File) {
  const formData = new FormData();

  formData.append("file", file);

  const response = await api.post<InferenceFile>(
    `/inference/${inferenceId}/files/`,
    formData,
  );

  return response.data;
}

export async function submitInference(inferenceId: number) {
  const response = await api.post<Inference>(
    `/inference/${inferenceId}/submit/`,
  );

  return response.data;
}

export async function getInference(inferenceId: number) {
  const response = await api.get<Inference>(`/inference/${inferenceId}/`);

  return response.data;
}

export interface CreatePatientPayload {
  first_name: string;
  last_name: string;
  date_of_birth: string;
  sex: string;
  phone_number?: string;
  email?: string;
  notes?: string;
}

export async function createPatient(payload: CreatePatientPayload) {
  const response = await api.post<Patient>("/patients/", payload);

  return response.data;
}
