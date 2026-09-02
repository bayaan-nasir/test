type PageHeaderProps = {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
};

export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: PageHeaderProps) {
  return (
    <div className="mb-6 flex flex-col gap-4 sm:mb-8 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        {eyebrow && (
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-blue-600">
            {eyebrow}
          </p>
        )}

        <h1 className="mt-1.5 text-[23px] font-semibold tracking-tight text-gray-950 sm:text-[28px]">
          {title}
        </h1>

        {description && (
          <p className="mt-1 max-w-2xl text-[12px] leading-5 text-gray-500 sm:text-[13px]">
            {description}
          </p>
        )}
      </div>

      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}