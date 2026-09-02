type PatientAvatarProps = {
	name: string;
	size?: "sm" | "md" | "lg";
};

export function PatientAvatar({
	name,
	size = "md",
}: PatientAvatarProps) {
	const initials = name
		.split(" ")
		.map((part) => part[0])
		.join("")
		.slice(0, 2)
		.toUpperCase();

	const sizes = {
		sm: "h-8 w-8 text-[9px]",
		md: "h-9 w-9 text-[10px]",
		lg: "h-12 w-12 text-[12px]",
	};

	return (
		<div
			className={`flex shrink-0 items-center justify-center rounded-full bg-gray-100 font-semibold text-gray-600 ${sizes[size]}`}
		>
			{initials}
		</div>
	);
}