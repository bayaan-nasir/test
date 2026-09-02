import { createBrowserRouter, Navigate } from "react-router-dom";

import { AppLayout } from "./layouts/AppLayout";

import { Login } from "./pages/auth/Login";
import { Register } from "./pages/auth/Register";

import { Dashboard } from "./pages/dashboard/Dashboard";
import { Patients } from "./pages/patients/Patients";

import { InferenceResult } from "./pages/inference/InferenceResult";
import { InferenceHistory } from "./pages/inference/InferenceHistory";

import { ModelDetails } from "./pages/models/ModelDetails";
import { PatientDetails } from "./pages/patients/PatientDetails";
import { Models } from "./pages/models/Models";
import { InferenceWorkspace } from "./pages/inference/InferenceWorkspace";
import { ProtectedRoute } from "./auth/ProtectedRoute";
import { AuthLayout } from "./layouts/AuthLayout";

export const router = createBrowserRouter([
	{
		path: "/",
		element: <Navigate to="/app/dashboard" replace />,
	},

	{
		element: <AuthLayout />,
		children: [
			{
				path: "/login",
				element: <Login />,
			},

			{
				path: "/register",
				element: <Register />,
			}
		]
	},
	{
		element: <ProtectedRoute />,
		children: [
			{
				path: "/app",
				element: <AppLayout />,
				children: [
					{
						index: true,
						element: <Navigate to="dashboard" replace />,
					},

					{
						path: "dashboard",
						element: <Dashboard />,
					},

					{
						path: "patients",
						element: <Patients />,
					},

					{
						path: "patients/:patientId",
						element: <PatientDetails />,
					},

					{
						path: "inference",
						element: <InferenceWorkspace />,
					},

					{
						path: "inference/history",
						element: <InferenceHistory />,
					},

					{
						path: "inference/:inferenceId",
						element: <InferenceResult />,
					},

					{
						path: "models",
						element: <Models />,
					},

					{
						path: "models/:modelId",
						element: <ModelDetails />,
					},
				],
			}
		]
	},

	{
		path: "*",
		element: <Navigate to="/app/dashboard" replace />,
	},
]);