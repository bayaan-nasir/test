import { ArrowDownRight, ArrowUpRight, type LucideIcon } from "lucide-react";

export function StatCard({
  label, value, change, trend, icon: Icon,
}: {
  label: string; value: string; change: string; trend: "up" | "down"; icon: LucideIcon;
}) {
  const positive = trend === "up";
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 transition-shadow hover:shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[12px] font-medium text-gray-500">{label}</p>
          <p className="mt-2 text-[27px] font-semibold tracking-tight text-gray-950">{value}</p>
        </div>
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gray-50 text-gray-500">
          <Icon size={17} strokeWidth={1.8} />
        </div>
      </div>
      <div className="mt-4 flex items-center gap-1.5 text-[11px]">
        <span className={positive ? "text-green-600" : "text-orange-600"}>
          {positive ? <ArrowUpRight size={13} className="inline" /> : <ArrowDownRight size={13} className="inline" />}
          {change}
        </span>
        <span className="text-gray-400">vs. last month</span>
      </div>
    </div>
  );
}