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
	model_used: string;
	explainability?: {
		type: string;
		heatmap_url?: string | null;
		top_features?: unknown;
	};
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

export interface ModelFieldOption {
	[value: string]: string;
}

export interface ModelField {
	name: string;
	type: "integer" | "float" | "string" | "file";
	required: boolean;
	description?: string;
	options?: ModelFieldOption[];
}

export interface ModelRegistryEntry {
	model_id: string;
	model_name: string;
	modality: "tabular" | "image";
	fields?: ModelField[];
}

export interface ModelRegistryResponse {
	total_models: number;
	production_ready_count: number;
	models: ModelRegistryEntry[];
}

export interface ModelField {
	name: string;
	type: "integer" | "float" | "string" | "file";
	required: boolean;
	description?: string;
	options?: ModelFieldOption[];
}

export interface ModelRegistryEntry {
	model_id: string;
	model_name: string;
	modality: "tabular" | "image";
	task: string;
	framework: string[];
	version: string;
	production_ready: boolean;
	endpoints: Record<string, string>;
	fields?: ModelField[];
	image_type_value?: string;
}

export interface ModelRegistryResponse {
	total_models: number;
	production_ready_count: number;
	models: ModelRegistryEntry[];
}

export async function getModelRegistry() {
	const response = await api.get<ModelRegistryResponse>("/inference/models/");
	return response.data;
}

export interface ModelRecentRun {
	inference_id: number;
	patient_name: string;
	predicted_class: string;
	confidence_pct: string;
	triage: TriageLevel;
	created_at: string | null;
}

export interface ModelUsageStat {
	total_runs: number;
	avg_confidence_pct: number | null;
	avg_latency_seconds: number | null;
	last_run_at: string | null;
	recent_runs: ModelRecentRun[];
}

export type ModelStatsResponse = Record<string, ModelUsageStat>;

export async function getModelStats() {
	const response = await api.get<ModelStatsResponse>(
		"/inference/model-stats/",
	);
	return response.data;
}
