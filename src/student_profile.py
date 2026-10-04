"""
Profili i studentit (personalizimi i RAG).

Profili ruhet vetëm te sesioni i shfletuesit (st.session_state), nuk shkruhet
në disk dhe nuk përmban të dhëna identifikuese (emër, email, nr. amze etj.).
"""

FIELDS = {
    "bachelor": "Bachelor i përfunduar",
    "interest": "Interesi kryesor",
    "level": "Niveli i studimit që kërkon",
    "experience": "Eksperienca dhe aftësitë",
    "goal": "Objektivi pas studimeve",
    "preference": "Preferenca e programit",
}


def is_empty(profile: dict | None) -> bool:
    return not profile or not any(str(v).strip() for v in profile.values())


def profile_to_text(profile: dict | None) -> str:
    """Profili si listë me pika, e përdorur te prompti dhe te kërkimi."""
    if is_empty(profile):
        return ""
    lines = [f"- {label}: {profile[key].strip()}" for key, label in FIELDS.items() if str(profile.get(key, "")).strip()]
    return "\n".join(lines)


def profile_to_query(profile: dict | None) -> str:
    """Tekst i shkurtër për të pasuruar kërkimin semantik në dokumente."""
    if is_empty(profile):
        return ""
    keys = ("interest", "level", "experience", "goal", "preference", "bachelor")
    return ". ".join(str(profile[k]).strip() for k in keys if str(profile.get(k, "")).strip())
