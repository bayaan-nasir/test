export type UserRole = "PATIENT" | "CLINICIAN" | "ADMIN";

export type ProfessionalRole =
	| "DOCTOR"
	| "NURSE"
	| "CLINICAL_RESEARCHER"
	| "OTHER";

export interface User {
	id: number;
	email: string;
	first_name: string;
	last_name: string;
	phone_number: string;
	role: UserRole;
	professional_role: ProfessionalRole | null;
}

export interface LoginRequest {
	email: string;
	password: string;
}

export interface RegisterRequest {
	email: string;
	first_name: string;
	last_name: string;
	phone_number: string;
	role: UserRole;
	professional_role: ProfessionalRole;
	password: string;
	confirm_password: string;
}

export interface AuthResponse {
	access: string;
	refresh: string;
	user: User;
}
