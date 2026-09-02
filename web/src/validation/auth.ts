import { z } from "zod";

export const professionalRoleSchema = z.enum([
	"DOCTOR",
	"NURSE",
	"CLINICAL_RESEARCHER",
	"OTHER",
]);

export const registerSchema = z
	.object({
		first_name: z
			.string()
			.trim()
			.min(1, "First name is required.")
			.max(100, "First name is too long."),

		last_name: z
			.string()
			.trim()
			.min(1, "Last name is required.")
			.max(100, "Last name is too long."),

		email: z.string().trim().email("Enter a valid email address."),

		phone_number: z
			.string()
			.trim()
			.min(7, "Enter a valid phone number.")
			.max(20, "Phone number is too long."),

		professional_role: professionalRoleSchema,

		password: z
			.string()
			.min(8, "Password must be at least 8 characters.")
			.max(128, "Password is too long."),

		confirm_password: z.string().min(1, "Please confirm your password."),

		accepted_terms: z.boolean().refine((val) => val === true, {
			message: "You must accept the terms of use.",
		}),
	})
	.refine((data) => data.password === data.confirm_password, {
		message: "Passwords do not match.",
		path: ["confirm_password"],
	});

export type RegisterFormValues = z.infer<typeof registerSchema>;

export const loginSchema = z.object({
	email: z.email("Enter a valid email address."),

	password: z.string().min(1, "Password is required."),
});

export type LoginFormValues = z.infer<typeof loginSchema>;
