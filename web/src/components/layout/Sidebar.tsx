import {
  Activity,
  Brain,
  BrainCircuit,
  History,
  LayoutDashboard,
  LogOut,
  Settings,
  Users,
  X,
} from "lucide-react";
import {
  NavLink,
  useNavigate,
} from "react-router-dom";

import { useAuth } from "../../auth/AuthContext";

interface SidebarProps {
  open?: boolean;
  onClose?: () => void;
}

const navigation = [
  {
    label: "Dashboard",
    path: "/app/dashboard",
    icon: LayoutDashboard,
  },
  {
    label: "Patients",
    path: "/app/patients",
    icon: Users,
  },
];

const inferenceNavigation = [
  {
    label: "New inference",
    path: "/app/inference",
    icon: Brain,
    end: true,
  },
  {
    label: "Inference history",
    path: "/app/inference/history",
    icon: History,
    end: true,
  },
  {
    label: "Models",
    path: "/app/models",
    icon: BrainCircuit,
  },
];

const secondaryNavigation = [
  {
    label: "Settings",
    path: "/app/settings",
    icon: Settings,
  },
];

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
  const first =
    firstName?.trim().charAt(0) ?? "";

  const last =
    lastName?.trim().charAt(0) ?? "";

  if (first || last) {
    return `${first}${last}`.toUpperCase();
  }

  return "U";
}

function NavigationLink({
  label,
  path,
  icon: Icon,
  end = false,
  onNavigate,
}: {
  label: string;
  path: string;
  icon: typeof LayoutDashboard;
  end?: boolean;
  onNavigate?: () => void;
}) {
  return (
    <NavLink
      to={path}
      end={end}
      onClick={onNavigate}
      className={({ isActive }) =>
        [
          "group flex items-center gap-3 rounded-lg px-3 py-2",
          "text-[11px] font-medium transition-colors",
          isActive
            ? "bg-gray-100 text-gray-950"
            : "text-gray-500 hover:bg-gray-50 hover:text-gray-900",
        ].join(" ")
      }
    >
      {({ isActive }) => (
        <>
          <Icon
            size={15}
            strokeWidth={
              isActive ? 2 : 1.8
            }
            className={
              isActive
                ? "text-gray-950"
                : "text-gray-400 group-hover:text-gray-700"
            }
          />

          <span>{label}</span>
        </>
      )}
    </NavLink>
  );
}

export function Sidebar({
  open = false,
  onClose,
}: SidebarProps) {
  const { user, logout } = useAuth();

  const navigate = useNavigate();

  const handleNavigate = () => {
    onClose?.();
  };

  const handleLogout = async () => {
    onClose?.();

    await logout();

    navigate("/login", {
      replace: true,
    });
  };

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

  const displayRole =
    professionalRole ||
    accountRole ||
    "User";

  return (
    <>
      {/* Mobile backdrop */}
      <div
        className={[
          "fixed inset-0 z-40 bg-black/30 backdrop-blur-[2px]",
          "transition-opacity duration-200 lg:hidden",
          open
            ? "pointer-events-auto opacity-100"
            : "pointer-events-none opacity-0",
        ].join(" ")}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Sidebar */}
      <aside
        className={[
          "fixed inset-y-0 left-0 z-50 flex w-60 flex-col",
          "border-r border-gray-200 bg-white",
          "transition-transform duration-200 ease-out",
          "lg:translate-x-0",
          open
            ? "translate-x-0"
            : "-translate-x-full",
        ].join(" ")}
      >
        {/* Brand */}
        <div className="flex h-16 shrink-0 items-center justify-between border-b border-gray-100 px-5">
          <NavLink
            to="/app/dashboard"
            onClick={handleNavigate}
            className="flex min-w-0 items-center gap-2.5"
          >
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gray-950 text-white">
              <Activity
                size={14}
                strokeWidth={2.2}
              />
            </div>

            <div className="min-w-0">
              <div className="truncate text-[12px] font-semibold tracking-tight text-gray-950">
                AI Diagnosis Assistant
              </div>

              <div className="truncate text-[8px] font-medium uppercase tracking-[0.12em] text-gray-400">
                Medical Intelligence
              </div>
            </div>
          </NavLink>

          {/* Mobile close */}
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-50 hover:text-gray-900 lg:hidden"
            aria-label="Close navigation"
          >
            <X size={17} />
          </button>
        </div>

        {/* Navigation */}
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-5">
          {/* Workspace */}
          <div>
            <div className="px-3 pb-2 text-[8px] font-semibold uppercase tracking-[0.14em] text-gray-400">
              Workspace
            </div>

            <nav className="space-y-0.5">
              {navigation.map((item) => (
                <NavigationLink
                  key={item.path}
                  {...item}
                  onNavigate={handleNavigate}
                />
              ))}
            </nav>
          </div>

          {/* AI inference */}
          <div className="mt-7">
            <div className="px-3 pb-2 text-[8px] font-semibold uppercase tracking-[0.14em] text-gray-400">
              AI inference
            </div>

            <nav className="space-y-0.5">
              {inferenceNavigation.map(
                (item) => (
                  <NavigationLink
                    key={item.path}
                    {...item}
                    onNavigate={
                      handleNavigate
                    }
                  />
                ),
              )}
            </nav>
          </div>

          {/* System */}
          <div className="mt-7">
            <div className="px-3 pb-2 text-[8px] font-semibold uppercase tracking-[0.14em] text-gray-400">
              System
            </div>

            <nav className="space-y-0.5">
              {secondaryNavigation.map(
                (item) => (
                  <NavigationLink
                    key={item.path}
                    {...item}
                    onNavigate={
                      handleNavigate
                    }
                  />
                ),
              )}
            </nav>
          </div>
        </div>

        {/* User section */}
        <div className="shrink-0 border-t border-gray-100 bg-white p-3">
          <div className="flex items-center gap-3 rounded-lg px-2 py-2">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gray-900 text-[9px] font-semibold text-white">
              {initials}
            </div>

            <div className="min-w-0 flex-1">
              <div className="truncate text-[10px] font-semibold text-gray-800">
                {fullName}
              </div>

              <div className="truncate text-[9px] text-gray-400">
                {displayRole}
              </div>
            </div>

            <button
              type="button"
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-gray-300 hover:bg-gray-50 hover:text-gray-700"
              aria-label="Log out"
              onClick={handleLogout}
            >
              <LogOut size={13} />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}