"""
Profili i studentit (Student Profile në arkitekturë).

Dy lloje profilesh:
- studenti aktual: fakulteti, departamenti, programi, specializimi, viti, semestri
- studenti potencial: arsimi, interesat, fusha, niveli, objektivat

Profili është një dictionary i thjeshtë, që ruhet te llogaria (accounts.py) dhe
i kalohet rag.py për të personalizuar kërkimin dhe përgjigjen.
"""

from accounts import ROLE_ACTUAL, ROLE_POTENTIAL

ACTUAL_FIELDS = ("fakulteti", "departamenti", "programi", "specializimi", "viti", "semestri")
POTENTIAL_FIELDS = ("arsimi", "interesat", "fusha", "niveli", "objektivat")

LEVELS = ("Bachelor", "Master Shkencor", "Master Profesional", "Doktoraturë")
YEARS = (1, 2)
SEMESTERS = (1, 2, 3, 4)

# Fushat e programeve që ka UET në dokumentet e ngarkuara.
FIELDS = (
    "Inxhinieri Ndërtimi",
    "Inxhinieri Elektrike",
    "Inxhinieri Mekanike",
    "Inxhinieri Informatike",
    "Teknologji Informacioni",
    "Tjetër",
)

REQUIRED_ACTUAL = ("programi", "viti", "semestri")
REQUIRED_POTENTIAL = ("interesat", "fusha", "niveli")

LABELS = {
    "fakulteti": "Fakulteti",
    "departamenti": "Departamenti",
    "programi": "Programi",
    "specializimi": "Specializimi",
    "viti": "Viti",
    "semestri": "Semestri",
    "arsimi": "Arsimi",
    "interesat": "Interesat",
    "fusha": "Fusha",
    "niveli": "Niveli",
    "objektivat": "Objektivat",
}


def semesters_for_year(year: int) -> tuple[int, int]:
    """Semestrat (absolutë) që i përkasin një viti: viti 1 -> 1,2; viti 2 -> 3,4."""
    return (2 * year - 1, 2 * year)


def validate(role: str, profile: dict) -> list[str]:
    """Kthen listën e problemeve të profilit (bosh nëse është i vlefshëm)."""
    problems = []
    required = REQUIRED_ACTUAL if role == ROLE_ACTUAL else REQUIRED_POTENTIAL
    for field in required:
        value = profile.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"Plotëso fushën: {LABELS[field]}.")
    if role == ROLE_ACTUAL and not problems:
        if profile["semestri"] not in semesters_for_year(profile["viti"]):
            problems.append("Semestri nuk i përket vitit të zgjedhur.")
    return problems


def describe(role: str, profile: dict) -> str:
    """Përshkrim tekstual i profilit, i dërguar te modeli bashkë me pyetjen."""
    fields = ACTUAL_FIELDS if role == ROLE_ACTUAL else POTENTIAL_FIELDS
    lines = []
    for field in fields:
        value = profile.get(field)
        if value not in (None, ""):
            lines.append(f"- {LABELS[field]}: {value}")
    kind = "student aktual" if role == ROLE_ACTUAL else "student potencial (kandidat)"
    return f"Profili ({kind}):\n" + "\n".join(lines)
