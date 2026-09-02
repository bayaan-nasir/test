import {
	Activity,
	AlertTriangle,
	ArrowLeft,
	ArrowUpRight,
	CheckCircle2,
	ChevronDown,
	Clock3,
	FileImage,
	Info,
	ShieldCheck,
	Sparkles,
	UserRound,
} from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

const predictions = [
	{
		label: "Pneumonia",
		probability: 94.7,
		primary: true,
	},
	{
		label: "Other abnormality",
		probability: 3.8,
		primary: false,
	},
	{
		label: "Normal",
		probability: 1.5,
		primary: false,
	},
];

const clinicalFactors = [
	{
		label: "Temperature",
		value: "38.4 °C",
		impact: "Elevated",
	},
	{
		label: "Respiratory rate",
		value: "24 bpm",
		impact: "Elevated",
	},
	{
		label: "Oxygen saturation",
		value: "94%",
		impact: "Reduced",
	},
	{
		label: "Symptom duration",
		value: "4 days",
		impact: "Relevant",
	},
];

const modelInfo = [
	["Model", "Respiratory Assessment"],
	["Version", "v2.4.1"],
	["Modality", "Multimodal"],
	["Inference ID", "INF-7F82A1"],
	["Started", "29 Aug 2026 · 14:31:58"],
	["Completed", "29 Aug 2026 · 14:32:04"],
	["Processing time", "6.21 seconds"],
];

export function InferenceAssessment() {
	const navigate = useNavigate();
	const { inferenceId } = useParams();

	const [showTechnical, setShowTechnical] =
		useState(false);
	const [reviewed, setReviewed] = useState(false);
	const [note, setNote] = useState("");

	return (
		<div className="mx-auto max-w-[1440px] px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
			{/* Back */}
			<button
				type="button"
				onClick={() => navigate(-1)}
				className="mb-5 flex items-center gap-2 text-[9px] font-medium text-gray-400 hover:text-gray-800"
			>
				<ArrowLeft size={12} />
				Back
			</button>

			{/* Header */}
			<div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
				<div>
					<div className="flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-blue-600">
						<Sparkles size={11} />
						AI Assessment
					</div>

					<div className="mt-2 flex flex-wrap items-center gap-3">
						<h1 className="text-[25px] font-semibold tracking-tight text-gray-950 sm:text-[29px]">
							Inference result
						</h1>

						<span className="rounded-md bg-amber-50 px-2 py-1 text-[8px] font-semibold text-amber-700">
							Review required
						</span>
					</div>

					<p className="mt-1 text-[10px] leading-5 text-gray-400">
						AI-assisted assessment for Ama Mensah.
					</p>
				</div>

				<div className="flex flex-wrap gap-2">
					<button
						type="button"
						onClick={() => navigate("/app/inference")}
						className="flex h-9 items-center gap-2 rounded-lg border border-gray-200 px-3 text-[8px] font-medium text-gray-600 hover:bg-gray-50"
					>
						<Activity size={11} />
						New inference
					</button>

					<button
						type="button"
						onClick={() => navigate("/app/patients/PT-10482")}
						className="flex h-9 items-center gap-2 rounded-lg bg-gray-950 px-3.5 text-[8px] font-semibold text-white hover:bg-gray-800"
					>
						View patient
						<ArrowUpRight size={10} />
					</button>
				</div>
			</div>

			{/* Result hero */}
			<section className="mb-5 overflow-hidden rounded-xl border border-gray-200 bg-white">
				<div className="grid lg:grid-cols-[1.1fr_0.9fr]">
					<div className="border-b border-gray-100 p-5 sm:p-7 lg:border-b-0 lg:border-r">
						<div className="text-[8px] font-semibold uppercase tracking-[0.12em] text-gray-400">
							Primary prediction
						</div>

						<div className="mt-3 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
							<div>
								<h2 className="text-[29px] font-semibold tracking-tight text-gray-950">
									Pneumonia
								</h2>

								<div className="mt-2 flex flex-wrap items-center gap-2">
									<span className="flex items-center gap-1.5 text-[8px] text-gray-400">
										<UserRound size={10} />
										Ama Mensah
									</span>

									<span className="text-gray-200">•</span>

									<span className="font-mono text-[8px] text-gray-400">
										PT-10482
									</span>
								</div>
							</div>

							<div className="sm:text-right">
								<div className="text-[28px] font-semibold tracking-tight text-gray-950">
									94.7%
								</div>

								<div className="mt-1 text-[8px] text-gray-400">
									model confidence
								</div>
							</div>
						</div>

						<div className="mt-7">
							<div className="mb-2 flex items-center justify-between">
								<span className="text-[7px] font-medium text-gray-400">
									Confidence
								</span>

								<span className="font-mono text-[7px] text-gray-400">
									0.947
								</span>
							</div>

							<div className="h-2 overflow-hidden rounded-full bg-gray-100">
								<div
									className="h-full rounded-full bg-gray-900"
									style={{ width: "94.7%" }}
								/>
							</div>
						</div>

						<div className="mt-6 flex items-start gap-3 rounded-lg border border-amber-100 bg-amber-50/50 p-3">
							<AlertTriangle
								size={13}
								className="mt-0.5 shrink-0 text-amber-600"
							/>

							<div>
								<div className="text-[8px] font-semibold text-amber-900">
									Clinical review recommended
								</div>

								<p className="mt-1 text-[7px] leading-4 text-amber-700">
									This result is generated by an AI model and is not
									a definitive diagnosis. A qualified clinician
									should review the finding alongside the patient's
									complete clinical context.
								</p>
							</div>
						</div>
					</div>

					{/* Probability distribution */}
					<div className="p-5 sm:p-7">
						<div className="text-[8px] font-semibold uppercase tracking-[0.12em] text-gray-400">
							Prediction distribution
						</div>

						<div className="mt-5 space-y-5">
							{predictions.map((prediction) => (
								<div key={prediction.label}>
									<div className="mb-2 flex items-center justify-between gap-3">
										<span
											className={[
												"text-[9px]",
												prediction.primary
													? "font-semibold text-gray-800"
													: "font-medium text-gray-500",
											].join(" ")}
										>
											{prediction.label}
										</span>

										<span className="font-mono text-[8px] text-gray-500">
											{prediction.probability.toFixed(1)}%
										</span>
									</div>

									<div className="h-1.5 overflow-hidden rounded-full bg-gray-100">
										<div
											className={[
												"h-full rounded-full",
												prediction.primary
													? "bg-gray-900"
													: "bg-gray-300",
											].join(" ")}
											style={{
												width: `${prediction.probability}%`,
											}}
										/>
									</div>
								</div>
							))}
						</div>
					</div>
				</div>

				{/* Metadata strip */}
				<div className="grid border-t border-gray-100 sm:grid-cols-3">
					<div className="border-b border-gray-100 p-4 sm:border-b-0 sm:border-r">
						<div className="text-[7px] uppercase tracking-wide text-gray-400">
							Model
						</div>

						<div className="mt-1.5 text-[8px] font-medium text-gray-700">
							Respiratory Assessment
						</div>
					</div>

					<div className="border-b border-gray-100 p-4 sm:border-b-0 sm:border-r">
						<div className="text-[7px] uppercase tracking-wide text-gray-400">
							Modality
						</div>

						<div className="mt-1.5 text-[8px] font-medium text-gray-700">
							Multimodal
						</div>
					</div>

					<div className="p-4">
						<div className="text-[7px] uppercase tracking-wide text-gray-400">
							Processing time
						</div>

						<div className="mt-1.5 flex items-center gap-1.5 text-[8px] font-medium text-gray-700">
							<Clock3 size={10} className="text-gray-400" />
							6.21 seconds
						</div>
					</div>
				</div>
			</section>

			{/* Main content */}
			<div className="grid gap-5 xl:grid-cols-[1.15fr_0.85fr]">
				{/* Explainability */}
				<div className="space-y-5">
					<section className="rounded-xl border border-gray-200 bg-white">
						<div className="border-b border-gray-100 p-5">
							<div className="flex items-center gap-2">
								<Sparkles size={13} className="text-gray-400" />

								<h2 className="text-[11px] font-semibold text-gray-900">
									Contributing factors
								</h2>
							</div>

							<p className="mt-1 text-[8px] leading-4 text-gray-400">
								Input characteristics associated with the model's
								assessment.
							</p>
						</div>

						<div className="p-5">
							<div className="mb-4 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
								Clinical factors
							</div>

							<div className="grid gap-2 sm:grid-cols-2">
								{clinicalFactors.map((factor) => (
									<div
										key={factor.label}
										className="rounded-lg border border-gray-100 bg-gray-50/50 p-3"
									>
										<div className="text-[7px] text-gray-400">
											{factor.label}
										</div>

										<div className="mt-1.5 flex items-end justify-between gap-2">
											<span className="text-[9px] font-semibold text-gray-700">
												{factor.value}
											</span>

											<span className="rounded bg-white px-1.5 py-1 text-[6px] font-medium text-gray-500">
												{factor.impact}
											</span>
										</div>
									</div>
								))}
							</div>

							<div className="mt-6 mb-4 text-[8px] font-semibold uppercase tracking-wide text-gray-400">
								Image findings
							</div>

							<div className="overflow-hidden rounded-xl border border-gray-200 bg-gray-950">
								<div className="relative aspect-[16/8]">
									<div className="flex h-full items-center justify-center bg-gray-900">
										<div className="relative h-[78%] w-[42%] rounded-[50%] border border-gray-700 bg-gray-800/70">
											<div className="absolute left-[18%] top-[13%] h-[74%] w-[25%] rounded-[50%] bg-gray-950/80" />
											<div className="absolute right-[18%] top-[13%] h-[74%] w-[25%] rounded-[50%] bg-gray-950/80" />
											<div className="absolute left-1/2 top-[13%] h-[74%] w-px -translate-x-1/2 bg-gray-600" />

											<div className="absolute left-[12%] top-[43%] h-5 w-8 rounded-full border border-gray-400/30" />
											<div className="absolute right-[16%] top-[50%] h-6 w-9 rounded-full border border-gray-400/30" />
										</div>
									</div>

									<div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-md bg-black/60 px-2 py-1.5">
										<FileImage size={10} className="text-gray-300" />

										<span className="text-[7px] text-gray-300">
											chest_xray_29082026.png
										</span>
									</div>
								</div>
							</div>

							<div className="mt-3 flex gap-3 rounded-lg bg-gray-50 p-3">
								<Info
									size={12}
									className="mt-0.5 shrink-0 text-gray-400"
								/>

								<p className="text-[7px] leading-4 text-gray-400">
									Image analysis indicates bilateral lower-lobe
									opacity patterns that contributed to the predicted
									classification.
								</p>
							</div>
						</div>
					</section>

					{/* Input snapshot */}
					<section className="rounded-xl border border-gray-200 bg-white">
						<div className="border-b border-gray-100 p-5">
							<h2 className="text-[11px] font-semibold text-gray-900">
								Input snapshot
							</h2>

							<p className="mt-1 text-[8px] text-gray-400">
								Data submitted to the inference engine for this run.
							</p>
						</div>

						<div className="grid grid-cols-2 gap-px bg-gray-100 sm:grid-cols-4">
							{[
								["Age", "34 years"],
								["Temperature", "38.4 °C"],
								["Respiratory rate", "24 bpm"],
								["Heart rate", "96 bpm"],
								["O₂ saturation", "94%"],
								["Systolic BP", "128 mmHg"],
								["Diastolic BP", "82 mmHg"],
								["Symptoms", "4 days"],
							].map(([label, value]) => (
								<div key={label} className="bg-white p-4">
									<div className="text-[7px] text-gray-400">
										{label}
									</div>

									<div className="mt-1.5 text-[9px] font-semibold text-gray-700">
										{value}
									</div>
								</div>
							))}
						</div>
					</section>
				</div>

				{/* Right */}
				<div className="space-y-5">
					{/* Clinical review */}
					<section className="rounded-xl border border-gray-200 bg-white">
						<div className="border-b border-gray-100 p-5">
							<div className="flex items-center gap-2">
								<ShieldCheck size={13} className="text-gray-400" />

								<h2 className="text-[11px] font-semibold text-gray-900">
									Clinical review
								</h2>
							</div>

							<p className="mt-1 text-[8px] leading-4 text-gray-400">
								Record your interpretation of this AI assessment.
							</p>
						</div>

						<div className="p-5">
							{!reviewed ? (
								<>
									<label>
										<div className="mb-1.5 text-[8px] font-medium text-gray-600">
											Clinical note
										</div>

										<textarea
											value={note}
											onChange={(event) =>
												setNote(event.target.value)
											}
											rows={5}
											placeholder="Add your clinical interpretation..."
											className="w-full resize-none rounded-lg border border-gray-200 bg-gray-50/50 p-3 text-[8px] leading-5 text-gray-700 outline-none placeholder:text-gray-300 focus:border-gray-400 focus:bg-white"
										/>
									</label>

									<button
										type="button"
										onClick={() => setReviewed(true)}
										className="mt-3 flex h-9 w-full items-center justify-center gap-2 rounded-lg bg-gray-950 text-[8px] font-semibold text-white hover:bg-gray-800"
									>
										<CheckCircle2 size={12} />
										Mark as reviewed
									</button>
								</>
							) : (
								<div className="rounded-lg border border-green-100 bg-green-50/50 p-4">
									<div className="flex items-center gap-2 text-[8px] font-semibold text-green-700">
										<CheckCircle2 size={12} />
										Assessment reviewed
									</div>

									<p className="mt-2 text-[8px] leading-4 text-green-700">
										{note ||
											"Reviewed by clinician. No additional note was provided."}
									</p>

									<button
										type="button"
										onClick={() => setReviewed(false)}
										className="mt-3 text-[7px] font-medium text-green-700 underline"
									>
										Edit review
									</button>
								</div>
							)}
						</div>
					</section>

					{/* Technical details */}
					<section className="rounded-xl border border-gray-200 bg-white">
						<button
							type="button"
							onClick={() =>
								setShowTechnical((value) => !value)
							}
							className="flex w-full items-center justify-between p-5 text-left"
						>
							<div>
								<h2 className="text-[11px] font-semibold text-gray-900">
									Technical details
								</h2>

								<p className="mt-1 text-[8px] text-gray-400">
									Model and execution metadata.
								</p>
							</div>

							<ChevronDown
								size={13}
								className={[
									"text-gray-400 transition",
									showTechnical ? "rotate-180" : "",
								].join(" ")}
							/>
						</button>

						{showTechnical && (
							<div className="border-t border-gray-100">
								{modelInfo.map(([label, value]) => (
									<div
										key={label}
										className="flex items-center justify-between gap-4 border-b border-gray-100 px-5 py-3 last:border-0"
									>
										<span className="text-[7px] text-gray-400">
											{label}
										</span>

										<span className="text-right font-mono text-[7px] text-gray-600">
											{value}
										</span>
									</div>
								))}
							</div>
						)}
					</section>

					{/* Audit */}
					<section className="rounded-xl border border-gray-200 bg-white p-5">
						<div className="flex items-start gap-3">
							<Clock3
								size={13}
								className="mt-0.5 shrink-0 text-gray-400"
							/>

							<div>
								<div className="text-[8px] font-semibold text-gray-700">
									Audit information
								</div>

								<p className="mt-1 text-[7px] leading-4 text-gray-400">
									Inference{" "}
									<span className="font-mono">
										{inferenceId ?? "INF-7F82A1"}
									</span>{" "}
									was executed by the authenticated clinical
									user and recorded in the system audit trail.
								</p>
							</div>
						</div>
					</section>
				</div>
			</div>

			{/* Disclaimer */}
			<div className="mt-6 flex justify-center">
				<div className="flex max-w-2xl items-start gap-2 text-center">
					<Info
						size={10}
						className="mt-0.5 shrink-0 text-gray-300"
					/>

					<p className="text-[7px] leading-4 text-gray-400">
						This AI assessment is intended to support, not replace,
						professional clinical judgment. Predictions may be
						incorrect and should be interpreted in conjunction with
						the patient's complete clinical information.
					</p>
				</div>
			</div>
		</div>
	);
}