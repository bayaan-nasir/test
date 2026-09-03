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

export async function getPatients() {
  const response = await api.get<Patient[]>("/patients/");
  console.log(response.data);

  return response.data;
}

export async function getPatient(patientId: string) {
  const response = await api.get<Patient>(`/patients/${patientId}/`);
  return response.data;
}

export interface UpdatePatientPayload {
  first_name?: string;
  last_name?: string;
  date_of_birth?: string;
  sex?: string;
  phone_number?: string;
  email?: string;
  notes?: string;
}

export async function updatePatient(
  patientId: string,
  payload: UpdatePatientPayload,
) {
  const response = await api.patch<Patient>(`/patients/${patientId}/`, payload);
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

export async function deletePatient(patientId: string) {
  await api.delete(`/patients/${patientId}/`);
}
