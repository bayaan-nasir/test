import {
  Bell,
  ChevronDown,
  Menu,
  Search,
} from "lucide-react";
import { useState } from "react";
import {
  useLocation,
  useNavigate,
} from "react-router-dom";

import { useAuth } from "../../auth/AuthContext";

interface HeaderProps {
  onMenuClick?: () => void;
}

const pageTitles: Record<string, string> = {
  "/app/dashboard": "Dashboard",
  "/app/patients": "Patients",
  "/app/inference": "New inference",
  "/app/inference/history": "Inference history",
  "/app/models": "Model registry",
};

function getPageTitle(pathname: string) {
  if (pageTitles[pathname]) {
    return pageTitles[pathname];
  }

  if (pathname.startsWith("/app/patients/")) {
    return "Patient";
  }

  if (pathname.startsWith("/app/inference/")) {
    return "Inference result";
  }

  if (pathname.startsWith("/app/models/")) {
    return "Model details";
  }

  return "Clinical Intelligence";
}

function formatRole(role?: string | null) {
  if (!role) {
    return "";
  }

  return role
    .toLowerCase()
    .split("_")
    .map(
      (word) =>
        word.charAt(0).toUpperCase() +
        word.slice(1),
    )
    .join(" ");
}

function getInitials(
  firstName?: string | null,
  lastName?: string | null,
) {
  const first = firstName?.trim().charAt(0) ?? "";
  const last = lastName?.trim().charAt(0) ?? "";

  if (first || last) {
    return `${first}${last}`.toUpperCase();
  }

  return "U";
}

export function Header({
  onMenuClick,
}: HeaderProps) {
  const { user, logout } = useAuth();

  const navigate = useNavigate();
  const location = useLocation();

  const [profileOpen, setProfileOpen] =
    useState(false);

  const pageTitle = getPageTitle(
    location.pathname,
  );

  const fullName =
    [user?.first_name, user?.last_name]
      .filter(Boolean)
      .join(" ") || "User";

  const initials = getInitials(
    user?.first_name,
    user?.last_name,
  );

  const professionalRole = formatRole(
    user?.professional_role,
  );

  const accountRole = formatRole(
    user?.role,
  );

  const handleLogout = async () => {
    setProfileOpen(false);

    await logout();

    navigate("/login", {
      replace: true,
    });
  };

  const handleProfileNavigation = (
    path: string,
  ) => {
    setProfileOpen(false);
    navigate(path);
  };

  return (
    <header className="sticky top-0 z-30 h-16 border-b border-gray-200 bg-white/95 backdrop-blur">
      <div className="flex h-full items-center gap-3 px-4 sm:px-6">
        {/* Mobile menu */}
        <button
          type="button"
          onClick={onMenuClick}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-900 lg:hidden"
          aria-label="Open navigation"
        >
          <Menu size={19} />
        </button>

        {/* Page title */}
        <div className="min-w-0">
          <div className="truncate text-[12px] font-semibold text-gray-900">
            {pageTitle}
          </div>

          <div className="hidden text-[9px] text-gray-400 sm:block">
            Clinical Intelligence Platform
          </div>
        </div>

        {/* Right side */}
        <div className="ml-auto flex items-center gap-1.5 sm:gap-2">
          {/* Search */}
          <button
            type="button"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700 sm:w-auto sm:gap-2 sm:px-3"
            aria-label="Search"
          >
            <Search size={15} />

            <span className="hidden text-[10px] font-medium sm:block">
              Search
            </span>

            <kbd className="ml-1 hidden rounded border border-gray-200 bg-gray-50 px-1.5 py-0.5 font-mono text-[8px] text-gray-400 md:block">
              /
            </kbd>
          </button>

          {/* Notifications */}
          <button
            type="button"
            className="relative flex h-9 w-9 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700"
            aria-label="Notifications"
          >
            <Bell size={15} />

            <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-blue-500 ring-2 ring-white" />
          </button>

          <div className="mx-1 hidden h-6 w-px bg-gray-200 sm:block" />

          {/* Profile */}
          <div className="relative">
            <button
              type="button"
              onClick={() =>
                setProfileOpen(
                  (value) => !value,
                )
              }
              className="flex items-center gap-2 rounded-lg p-1.5 hover:bg-gray-50"
              aria-expanded={profileOpen}
              aria-haspopup="menu"
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gray-900 text-[9px] font-semibold text-white">
                {initials}
              </div>

              <div className="hidden min-w-0 text-left md:block">
                <div className="max-w-32 truncate text-[10px] font-semibold text-gray-800">
                  {fullName}
                </div>

                <div className="max-w-32 truncate text-[8px] text-gray-400">
                  {professionalRole ||
                    accountRole ||
                    "User"}
                </div>
              </div>

              <ChevronDown
                size={13}
                className={`hidden text-gray-400 transition-transform md:block ${profileOpen
                    ? "rotate-180"
                    : ""
                  }`}
              />
            </button>

            {profileOpen && (
              <div
                role="menu"
                className="absolute right-0 top-[calc(100%+8px)] w-47.5 overflow-hidden rounded-xl border border-gray-200 bg-white p-1.5 shadow-lg shadow-gray-200/50"
              >
                {/* User information */}
                <div className="border-b border-gray-100 px-3 py-2.5">
                  <div className="text-[10px] font-semibold text-gray-800">
                    {fullName}
                  </div>

                  <div className="mt-0.5 truncate text-[9px] text-gray-400">
                    {user?.email}
                  </div>

                  <div className="mt-1.5 flex items-center gap-1.5">
                    {professionalRole && (
                      <span className="rounded-md bg-blue-50 px-1.5 py-0.5 text-[8px] font-medium text-blue-600">
                        {professionalRole}
                      </span>
                    )}

                    {accountRole && (
                      <span className="rounded-md bg-gray-100 px-1.5 py-0.5 text-[8px] font-medium text-gray-500">
                        {accountRole}
                      </span>
                    )}
                  </div>
                </div>

                {/* Profile */}
                <button
                  type="button"
                  role="menuitem"
                  onClick={() =>
                    handleProfileNavigation(
                      "/app/profile",
                    )
                  }
                  className="mt-1 flex w-full rounded-lg px-3 py-2 text-left text-[10px] text-gray-600 hover:bg-gray-50"
                >
                  Profile
                </button>

                {/* Settings */}
                <button
                  type="button"
                  role="menuitem"
                  onClick={() =>
                    handleProfileNavigation(
                      "/app/settings",
                    )
                  }
                  className="flex w-full rounded-lg px-3 py-2 text-left text-[10px] text-gray-600 hover:bg-gray-50"
                >
                  Settings
                </button>

                {/* Sign out */}
                <button
                  type="button"
                  role="menuitem"
                  onClick={handleLogout}
                  className="flex w-full rounded-lg px-3 py-2 text-left text-[10px] text-red-600 hover:bg-red-50"
                >
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}