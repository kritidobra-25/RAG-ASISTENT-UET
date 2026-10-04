"""
Profili i studentit: skema, bashkimi i përditësimeve (ProfileManager) dhe
kontrolli i informacionit që mungon.

Profili ruhet vetëm te sesioni i shfletuesit (st.session_state), nuk shkruhet
në disk dhe nuk përmban të dhëna identifikuese (emër, email, nr. amze etj.).
"""

import copy

# Fushat tekst, lista dhe fjalori. Çelësat ndjekin skemën e specifikimit.
SCALAR_FIELDS = {
    "academic_background": "Sfondi akademik",
    "current_program": "Programi aktual",
    "current_level": "Niveli akademik",
    "career_goal": "Objektivi i karrierës",
    "preferred_area": "Fusha e preferuar",
    "preferred_specialization": "Specializimi i preferuar",
    "existing_experience": "Përvoja ekzistuese",
}
LIST_FIELDS = {
    "technical_skills": "Aftësi teknike",
    "interests": "Interesa",
    "improve_areas": "Fusha për t'u përmirësuar",
}
DICT_FIELDS = {"experience_level": "Niveli i aftësive"}

LEVELS = {"beginner": "fillestar", "intermediate": "mesatar", "advanced": "i avancuar"}


def new_profile() -> dict:
    profile: dict = {key: "" for key in SCALAR_FIELDS}
    profile.update({key: [] for key in LIST_FIELDS})
    profile.update({key: {} for key in DICT_FIELDS})
    return profile


def is_empty(profile: dict | None) -> bool:
    return not profile or not any(profile.get(key) for key in [*SCALAR_FIELDS, *LIST_FIELDS, *DICT_FIELDS])


def merge_profile(profile: dict | None, updates: dict | None) -> tuple[dict, list[str]]:
    """Bashkon përditësimet në profil. Kthen (profili i ri, lista e ndryshimeve).

    Vlerat tekst mbishkruhen nga e reja, listat zgjerohen pa dublikatë, fjalori
    i niveleve përditësohet aftësi pas aftësie.
    """
    merged = copy.deepcopy(profile) if profile else new_profile()
    changes: list[str] = []
    for key, value in (updates or {}).items():
        if value in (None, "", [], {}):
            continue
        if key in SCALAR_FIELDS and isinstance(value, str):
            value = value.strip()
            if value and value != merged.get(key):
                merged[key] = value
                changes.append(f"{SCALAR_FIELDS[key]}: {value}")
        elif key in LIST_FIELDS and isinstance(value, list):
            known = {item.lower() for item in merged[key]}
            for item in value:
                item = str(item).strip()
                if item and item.lower() not in known:
                    merged[key].append(item)
                    known.add(item.lower())
                    changes.append(f"{LIST_FIELDS[key]}: + {item}")
        elif key in DICT_FIELDS and isinstance(value, dict):
            for skill, level in value.items():
                level = str(level).strip().lower()
                if skill and level in LEVELS and merged[key].get(skill) != level:
                    merged[key][skill] = level
                    changes.append(f"{skill}: {LEVELS[level]}")
    return merged, changes


def profile_to_text(profile: dict | None) -> str:
    """Profili si listë me pika, e përdorur te prompti dhe te ndërfaqja."""
    if is_empty(profile):
        return ""
    lines = []
    for key, label in SCALAR_FIELDS.items():
        if profile.get(key):
            lines.append(f"- {label}: {profile[key]}")
    for key, label in LIST_FIELDS.items():
        if profile.get(key):
            lines.append(f"- {label}: {', '.join(profile[key])}")
    if profile.get("experience_level"):
        levels = ", ".join(f"{skill} ({LEVELS[level]})" for skill, level in profile["experience_level"].items())
        lines.append(f"- {DICT_FIELDS['experience_level']}: {levels}")
    return "\n".join(lines)


def profile_to_query(profile: dict | None) -> str:
    """Tekst i shkurtër për të pasuruar kërkimin semantik në dokumente."""
    if is_empty(profile):
        return ""
    parts = [
        profile.get("career_goal", ""),
        profile.get("preferred_specialization", ""),
        profile.get("preferred_area", ""),
        profile.get("current_level", ""),
        *profile.get("interests", []),
        *profile.get("technical_skills", []),
        profile.get("academic_background", ""),
    ]
    return ". ".join(part for part in parts if part)


def missing_for(intent: str, profile: dict) -> list[str]:
    """Çfarë informacioni mungon për këtë lloj kërkese: 'background', 'goal', 'skills'."""
    missing = []
    has_goal = any(profile.get(k) for k in ("career_goal", "preferred_specialization", "preferred_area")) or profile.get("interests")
    has_background = any(profile.get(k) for k in ("academic_background", "current_program", "current_level"))
    if intent != "what_if" and not has_goal:
        missing.append("goal")
    if intent != "program_comparison" and not has_background:
        missing.append("background")
    if intent == "skill_gap_analysis" and not (profile.get("technical_skills") or profile.get("experience_level")):
        missing.append("skills")
    return missing


FOLLOW_UP_PARTS = {
    "background": "çfarë diplome Bachelor ke mbaruar ose cili është niveli yt aktual akademik",
    "goal": "cili është interesi ose objektivi yt i karrierës (p.sh. Data Engineering, zhvillim softueri)",
    "skills": "çfarë aftësish teknike ke dhe në çfarë niveli (p.sh. SQL i avancuar, Python mesatar)",
}


def follow_up_question(missing: list[str]) -> str:
    parts = [FOLLOW_UP_PARTS[key] for key in missing[:2]]
    return "Për të të rekomanduar një rrugë të përshtatshme, më trego " + " dhe ".join(parts) + "?"
