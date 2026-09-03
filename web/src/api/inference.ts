import api from "./client";

export type InferenceType = "SYMPTOMS" | "IMAGE";
export type InferenceRunStatus =
  | "PENDING"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED";
export type TriageLevel = "high" | "medium" | "low";

export interface InferenceModelResult {
  disease: string;
  domain: string;
  predicted_class: string;
  confidence: number;
  confidence_pct: string;
  triage: TriageLevel;
  explainability: Record<string, unknown>;
  model_used: string;
}

export interface InferenceResponsePayload {
  request_id?: string;
  models_run?: string[];
  models_skipped?: string[];
  results?: InferenceModelResult[];
  top_result?: InferenceModelResult;
  overall_triage?: TriageLevel;
  clinical_summary?: string;
  gemini_used?: boolean;
  image_type_used?: string | null;
  clinical_notes?: string | null;
  disclaimer?: string;
}

export interface InferenceRequestPayload {
  symptoms?: Record<string, unknown>;
  clinical_notes?: string;
  image_type?: string;
}

export interface Inference {
  id: number;
  patient_id: string | null;
  patient_name: string;
  inference_type: InferenceType;
  status: InferenceRunStatus;
  overall_triage: string;
  predicted_class: string;
  confidence: number | null;
  clinical_summary: string;
  request_payload: InferenceRequestPayload;
  response_payload: InferenceResponsePayload;
  error_message: string;
  fastapi_request_id: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export async function getInference(inferenceId: number) {
  const response = await api.get<Inference>(`/inference/${inferenceId}/`);
  return response.data;
}

export async function getInferences() {
  const response = await api.get<Inference[]>("/inference/");
  return response.data;
}
