import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "./AuthContext";

export function ProtectedRoute() {
	const { isAuthenticated, isLoading } = useAuth();
	const location = useLocation();

	if (isLoading) {
		return (
			<div className="flex min-h-screen items-center justify-center bg-gray-50">
				<div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-gray-900" />
			</div>
		);
	}

	if (!isAuthenticated) {
		return (
			<Navigate
				to="/login"
				replace
				state={{ from: location }}
			/>
		);
	}

	return <Outlet />;
}