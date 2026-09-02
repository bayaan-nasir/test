import { Stethoscope } from "lucide-react";

export function AuthLogo() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gray-950 text-white">
        <Stethoscope size={19} strokeWidth={1.9} />
      </div>

      <div>
        <div className="text-[15px] font-bold tracking-tight text-gray-500">
          AI Diagnosis Assistant
        </div>

        <div className="text-[10px] font-medium uppercase tracking-[0.13em] text-gray-400">
          Clinical Intelligence
        </div>
      </div>
    </div>
  );
}