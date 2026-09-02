import {
	createContext,
	useCallback,
	useContext,
	useEffect,
	useState,
	type ReactNode,
} from "react";

import {
	getCurrentUser,
	login as loginRequest,
	logout as logoutRequest,
	register as registerRequest,
} from "../api/auth";

import type {
	LoginRequest,
	RegisterRequest,
	User,
} from "../types/auth";

interface AuthContextValue {
	user: User | null;
	accessToken: string | null;
	isAuthenticated: boolean;
	isLoading: boolean;

	login: (
		data: LoginRequest,
	) => Promise<void>;

	register: (
		data: RegisterRequest,
	) => Promise<void>;

	logout: () => Promise<void>;

	refreshSession: () => Promise<boolean>;
}

const AuthContext =
	createContext<
		AuthContextValue | undefined
	>(undefined);

const ACCESS_TOKEN_KEY =
	"access_token";

const REFRESH_TOKEN_KEY =
	"refresh_token";

export function AuthProvider({
	children,
}: {
	children: ReactNode;
}) {
	const [user, setUser] =
		useState<User | null>(null);

	const [accessToken, setAccessToken] =
		useState<string | null>(
			() =>
				sessionStorage.getItem(
					ACCESS_TOKEN_KEY,
				),
		);

	const [isLoading, setIsLoading] =
		useState(true);

	const clearSession =
		useCallback(() => {
			sessionStorage.removeItem(
				ACCESS_TOKEN_KEY,
			);

			sessionStorage.removeItem(
				REFRESH_TOKEN_KEY,
			);

			setAccessToken(null);
			setUser(null);
		}, []);

	const refreshSession =
		useCallback(async () => {
			try {
				const currentUser =
					await getCurrentUser();

				setUser(currentUser);

				const token =
					sessionStorage.getItem(
						ACCESS_TOKEN_KEY,
					);

				setAccessToken(token);

				return true;
			} catch {
				clearSession();
				return false;
			}
		}, [clearSession]);

	useEffect(() => {
		const initialise =
			async () => {
				const token =
					sessionStorage.getItem(
						ACCESS_TOKEN_KEY,
					);

				if (!token) {
					setIsLoading(false);
					return;
				}

				try {
					setAccessToken(token);

					const currentUser =
						await getCurrentUser();

					setUser(currentUser);
				} catch {
					/*
					 * The Axios interceptor will
					 * automatically attempt a refresh.
					 *
					 * If refresh succeeds, getCurrentUser()
					 * is retried automatically.
					 */
					const refreshed =
						await refreshSession();

					if (!refreshed) {
						clearSession();
					}
				} finally {
					setIsLoading(false);
				}
			};

		initialise();
	}, [
		refreshSession,
		clearSession,
	]);

	/*
	 * Listen for the case where the refresh
	 * token itself has expired/revoked.
	 */
	useEffect(() => {
		const handleAuthExpired =
			() => {
				clearSession();
			};

		window.addEventListener(
			"auth:expired",
			handleAuthExpired,
		);

		return () => {
			window.removeEventListener(
				"auth:expired",
				handleAuthExpired,
			);
		};
	}, [clearSession]);

	const login = useCallback(
		async (data: LoginRequest) => {
			const response =
				await loginRequest(data);

			sessionStorage.setItem(
				ACCESS_TOKEN_KEY,
				response.access,
			);

			sessionStorage.setItem(
				REFRESH_TOKEN_KEY,
				response.refresh,
			);

			setAccessToken(response.access);
			setUser(response.user);
		},
		[],
	);

	const register = useCallback(
		async (
			data: RegisterRequest,
		) => {
			await registerRequest(data);

			/*
			 * Registration does not issue tokens.
			 * Log the newly-created user in.
			 */
			await login({
				email: data.email,
				password: data.password,
			});
		},
		[login],
	);

	const logout = useCallback(
		async () => {
			const refreshToken =
				sessionStorage.getItem(
					REFRESH_TOKEN_KEY,
				);

			try {
				if (refreshToken) {
					await logoutRequest(
						refreshToken,
					);
				}
			} finally {
				clearSession();
			}
		},
		[clearSession],
	);

	return (
		<AuthContext.Provider
			value={{
				user,
				accessToken,
				isAuthenticated:
					Boolean(user && accessToken),
				isLoading,
				login,
				register,
				logout,
				refreshSession,
			}}
		>
			{children}
		</AuthContext.Provider>
	);
}

export function useAuth() {
	const context =
		useContext(AuthContext);

	if (!context) {
		throw new Error(
			"useAuth must be used inside AuthProvider",
		);
	}

	return context;
}