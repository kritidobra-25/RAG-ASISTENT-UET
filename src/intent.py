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

STUDY_TASKS = {"explain", "question", "quiz", "exercises", "flashcards", "mock_exam", "review_plan", "summary", "overview"}

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
  "search_queries": [up to 3 short Albanian search strings for the university documents],
  "standalone_question": string,  // the latest message rewritten as a complete question using the recent messages ("Po REST?" after a talk about APIs -> "Çfarë është REST në kontekstin e API-ve?"). Same language as the student. Equal to the message if it is already complete.
  "study_task": "explain"|"question"|"quiz"|"exercises"|"flashcards"|"mock_exam"|"review_plan"|"summary"|"overview"|null,  // what the student wants to do with a course: explain a concept, ask a question, get a quiz / practice exercises / flashcards / a mock exam / a revision plan / a summary, or "overview" (what does a course cover, how many ECTS). null if not about studying a course.
  "topic": string|null,           // the concept or topic the student names, e.g. "REST API", "JavaScript"
  "course_mentioned": string|null, // a course name the student mentions, exactly as written
  "asks_other_programs": boolean   // true ONLY if the student explicitly asks about other UET programs, comparing programs or switching programs
}

Rules:
- profile_updates: include ONLY what the student explicitly states in the latest message or confirms from the conversation. Never guess or infer. Use null or [] when absent.
- Use the existing profile and recent messages to resolve references like "that program" or "do it".
- search_queries: program names, subjects or career areas worth looking up. For a comparison, one query per program. For a what-if, queries about the NEW target. Empty list for simple questions.
- Personalized intents are only for requests that need the student's own situation. Plain factual questions are never personalized.""" % json.dumps(list(INTENTS))


def neutral_analysis(question: str) -> dict:
    """Analiza bosh (pa thirrje LLM), p.sh. për veprimet e shpejta të Study Mode."""
    return {
        "intent": "study_help", "profile_updates": {}, "search_queries": [], "standalone_question": question,
        "study_task": None, "topic": None, "course_mentioned": None, "asks_other_programs": False,
    }


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
        return {
            "intent": "other", "profile_updates": {}, "search_queries": [], "standalone_question": question,
            "study_task": None, "topic": None, "course_mentioned": None, "asks_other_programs": False,
        }

    intent = data.get("intent") if data.get("intent") in INTENTS else "other"
    updates = data.get("profile_updates") if isinstance(data.get("profile_updates"), dict) else {}
    queries = data.get("search_queries") if isinstance(data.get("search_queries"), list) else []
    standalone = data.get("standalone_question")
    task = data.get("study_task")
    return {
        "intent": intent,
        "profile_updates": updates,
        "search_queries": [str(q) for q in queries if str(q).strip()][:3],
        "standalone_question": standalone.strip() if isinstance(standalone, str) and standalone.strip() else question,
        "study_task": task if task in STUDY_TASKS else None,
        "topic": data.get("topic") if isinstance(data.get("topic"), str) and data["topic"].strip() else None,
        "course_mentioned": data.get("course_mentioned") if isinstance(data.get("course_mentioned"), str) and data["course_mentioned"].strip() else None,
        "asks_other_programs": data.get("asks_other_programs") is True,
    }
