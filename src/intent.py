"""
Klasifikimi i qëllimit (intent) dhe nxjerrja e informacionit të profilit nga mesazhi.

Një thirrje e vetme JSON te modeli kthen: qëllimin, përditësimet e profilit
dhe pyetjet shtesë të kërkimit. Nëse përgjigjja nuk është JSON i vlefshëm,
kthehet intent "other" pa përditësime, kështu që sistemi bie te RAG standard.
"""

import json

import llm

INTENTS = {
    "general_info": "pyetje e përgjithshme për UET-në",
    "program_info": "informacion për një program studimi",
    "course_info": "informacion për një lëndë",
    "study_help": "ndihmë për lëndët e semestrit aktual ose për studimin (student aktual)",
    "admission": "pranimi, kriteret, dokumentet",
    "tuition_fees": "tarifat dhe pagesat",
    "academic_calendar": "kalendari akademik, afatet",
    "schedule": "orari i mësimit",
    "career_opportunities": "mundësitë e punësimit pas një programi",
    "program_comparison": "krahasim programesh ose 'cili program është më mirë për mua'",
    "personalized_recommendation": "kërkon rekomandim të personalizuar programi ose lënde",
    "skill_gap_analysis": "krahasim i aftësive aktuale me ato të kërkuara nga një objektiv",
    "academic_pathway": "kërkon një rrugë akademike hap pas hapi",
    "what_if": "skenar alternativ: 'po sikur të zgjedh X në vend të Y'",
    "other": "çdo gjë tjetër",
}

# Këto kërkesa trajtohen me profilin dhe formatin e personalizuar.
PERSONALIZED_INTENTS = {
    "program_comparison",
    "personalized_recommendation",
    "skill_gap_analysis",
    "academic_pathway",
    "what_if",
}

SYSTEM_PROMPT = """You analyze one message from a student talking to a university academic-advisor chatbot (Albanian university UET). Return ONLY a JSON object with exactly these keys:

{
  "intent": one of %s,
  "profile_updates": {
    "academic_background": string|null,   // e.g. "Bachelor in Business Administration"
    "current_program": string|null,
    "current_level": string|null,         // e.g. "Bachelor", "Master"
    "desired_program": "Master Profesional"|"Master Shkencor"|null,  // only if the student says which type of Master they want
    "technical_skills": [string],
    "experience_level": {skill: "beginner"|"intermediate"|"advanced"},
    "interests": [string],
    "career_goal": string|null,
    "preferred_area": string|null,        // e.g. "Technical / IT"
    "preferred_specialization": string|null,
    "existing_experience": string|null,
    "improve_areas": [string]
  },
  "search_queries": [up to 3 short Albanian search strings for the university documents]
}

Rules:
- profile_updates: include ONLY what the student explicitly states in the latest message or confirms from the conversation. Never guess or infer. Use null or [] when absent.
- Use the existing profile and recent messages to resolve references like "that program" or "do it".
- search_queries: program names, subjects or career areas worth looking up. For a comparison, one query per program. For a what-if, queries about the NEW target. Empty list for simple questions.
- Personalized intents are only for requests that need the student's own situation. Plain factual questions are never personalized.""" % json.dumps(list(INTENTS))


def analyze_turn(question: str, profile_text: str, history: list[dict] | None = None, role: str = "prospective") -> dict:
    """Kthen {'intent', 'profile_updates', 'search_queries'}."""
    recent = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in (history or [])[-6:]
    )
    user_prompt = (
        f"Student type: {'current student' if role == 'current' else 'prospective student (not enrolled yet)'}\n\n"
        f"Existing profile:\n{profile_text or '(empty)'}\n\n"
        f"Recent messages:\n{recent or '(none)'}\n\n"
        f"Latest student message:\n{question}"
    )
    try:
        data = json.loads(llm.chat_json(SYSTEM_PROMPT, user_prompt))
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except (ValueError, TypeError):
        return {"intent": "other", "profile_updates": {}, "search_queries": []}

    intent = data.get("intent") if data.get("intent") in INTENTS else "other"
    updates = data.get("profile_updates") if isinstance(data.get("profile_updates"), dict) else {}
    queries = data.get("search_queries") if isinstance(data.get("search_queries"), list) else []
    return {
        "intent": intent,
        "profile_updates": updates,
        "search_queries": [str(q) for q in queries if str(q).strip()][:3],
    }
