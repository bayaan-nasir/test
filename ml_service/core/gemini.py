"""
core/gemini.py
Gemini integration for clinical summary generation.

Takes ML model results + doctor's clinical notes and returns a
structured clinical summary the doctor can act on.

The prompt is carefully designed for medical reasoning:
  - Provides all ML results with confidence scores
  - Includes the doctor's own observations
  - Asks for a structured differential diagnosis summary
  - Reminds the model this is for a qualified clinician
  - Requests actionable next steps

Install: pip install google-generativeai
"""
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from core.config import settings
from core.logger import logger


def _build_prompt(
    ml_results: list[dict],
    clinical_notes: str | None,
    image_type: str | None,
) -> str:
    """
    Build the prompt sent to Gemini.
    Structured as a clinical handoff from the ML system to the LLM.
    """
    results_text = "\n".join([
        f"  - {r['disease']} ({r['domain']}): {r['predicted_class']} "
        f"— confidence {r['confidence_pct']} — triage: {r['triage']}"
        for r in ml_results
    ])

    image_context = (
        f"An uploaded {image_type.replace('_', ' ')} image was analysed."
        if image_type else
        "No medical image was uploaded. Analysis is based on clinical/lab values only."
    )

    notes_section = (
        f"\nClinical notes from the attending doctor:\n\"{clinical_notes}\""
        if clinical_notes and clinical_notes.strip()
        else "\nNo additional clinical notes were provided."
    )

    return f"""You are an expert clinical decision support AI assisting a qualified medical professional in Ghana.

{image_context}

MACHINE LEARNING MODEL RESULTS:
{results_text}
{notes_section}

Based on the ML results and the doctor's clinical notes above, provide a concise clinical summary with the following structure:

1. PRIMARY ASSESSMENT: The most likely diagnosis with reasoning, referencing both the ML confidence and clinical notes.

2. DIFFERENTIAL DIAGNOSES: Other conditions to consider, ranked by likelihood. Include why each should or should not be prioritised.

3. RECOMMENDED INVESTIGATIONS: Specific tests or imaging to confirm the primary diagnosis (e.g. "order a full blood count", "request sputum AFB smear").

4. IMMEDIATE ACTIONS: Based on the triage level, what should happen next (e.g. admit, refer, prescribe, monitor).

5. CLINICAL CAVEAT: Any important limitations of the ML analysis the doctor should be aware of.

Keep the response concise, clinical, and actionable. Write for a doctor, not a patient. Do not repeat the raw confidence numbers — synthesise them into clinical language."""


async def generate_clinical_summary(
    ml_results: list[dict],
    clinical_notes: str | None,
    image_type: str | None,
) -> tuple[str, bool]:
    """
    Call Gemini API and return (summary_text, success_bool).
    Falls back gracefully if API key is missing or call fails.

    Args:
        ml_results:     list of model result dicts (disease, predicted_class, confidence_pct, triage)
        clinical_notes: free text from the doctor (may be None/empty)
        image_type:     string like "xray", "skin", etc. (may be None)

    Returns:
        (clinical_summary_string, gemini_was_called_successfully)
    """
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        logger.warning("Gemini API key not configured — skipping LLM summary")
        return _fallback_summary(ml_results), False

    prompt = _build_prompt(ml_results, clinical_notes, image_type)

    try:
        # Configure SDK with API key
        genai.configure(api_key=settings.gemini_api_key)

        generation_config = GenerationConfig(
            temperature=0.3,
            max_output_tokens=2048,
            top_p=0.8,
        )

        # Set safety blocks to BLOCK_NONE to prevent early cutoffs on clinical discussion
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        response = await model.generate_content_async(prompt)

        logger.info("Gemini clinical summary generated successfully")
        return response.text.strip(), True

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return _fallback_summary(ml_results), False


def _fallback_summary(ml_results: list[dict]) -> str:
    """
    Rule-based summary used when Gemini is unavailable.
    Still useful — gives the doctor the ranked results in plain language.
    """
    if not ml_results:
        return "No model results available to summarise."

    top = ml_results[0]
    lines = [
        f"PRIMARY ASSESSMENT: ML analysis suggests {top['predicted_class']} "
        f"({top['disease']}) with {top['confidence_pct']} confidence. "
        f"Triage level: {top['triage'].upper()}.",
        "",
        "DIFFERENTIAL DIAGNOSES:",
    ]
    for r in ml_results[1:]:
        lines.append(
            f"  - {r['predicted_class']} ({r['disease']}): {r['confidence_pct']} confidence"
        )

    lines += [
        "",
        "NOTE: Gemini clinical summary is unavailable. "
        "Please interpret the ML results above using your clinical judgement. "
        "Set GEMINI_API_KEY in your .env file to enable AI-generated summaries.",
    ]
    return "\n".join(lines)