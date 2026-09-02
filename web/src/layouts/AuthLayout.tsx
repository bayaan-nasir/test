import { Outlet } from "react-router";
import { AuthShell } from "../components/auth/AuthShell";

export function AuthLayout() {
	return <AuthShell><Outlet /></AuthShell>;
}