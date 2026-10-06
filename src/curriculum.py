"""
Plani mësimor i strukturuar, i nxjerrë nga tabelat e PDF-ve te data/.

Çdo lëndë ruhet me: programin, kategorinë (A-E), vitin, semestrin, emrin, ECTS,
dokumentin burimor dhe faqen. Asgjë nuk plotësohet me dorë: nëse një rresht nuk
njihet nga tabela, ai nuk futet dhe raportohet te validimi.

Përdorimi:
    python src/build_curriculum.py      # ndërton data/curriculum.json
"""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from pypdf import PdfReader

import config
from extract_and_chunk import program_from_filename

CATEGORIES = {
    "A": "Lëndë bazë",
    "B": "Lëndë karakterizuese",
    "C": "Lëndë ndërdisiplinore ose integruese",
    "D": "Lëndë plotësuese",
    "E": "Detyrime përmbyllëse",
}

# Dy paraqitje të tabelës: me kolonën "Viti" (I/II) dhe pa të (programe një-vjeçare).
_ROW_WITH_YEAR = re.compile(r"^(\d+)\s+(II|I)\s+(1-2|1|2)\s+(.+?)\s+(\d+)\s*\**$")
_ROW_NO_YEAR = re.compile(r"^(\d+)\s+(1-2|1|2)\s+(.+?)\s+(\d+)\s*\**$")
_CATEGORY = re.compile(r"^\s*([A-E])\s*[-–]\s*(.+)$")
_STATED_ECTS = re.compile(r"(\d+)\s*ECTS", re.IGNORECASE)
_TOTAL = re.compile(r"^\s*TOTALI\s+(\d+)\s*$", re.IGNORECASE)
_ELECTIVE_MARKER = re.compile(r"zgjedhje", re.IGNORECASE)
_LOOKS_LIKE_ROW = re.compile(r"^\d+\s+(?:(?:II|I)\s+)?(?:1-2|1|2)\s+\S")
UNKNOWN_CATEGORY = "?"
_NOT_A_PROFILE_TITLE = re.compile(r"(?:^(?:profili|nr\.?|totali|master|msh|msc|mp)\b|ects|%|\d)", re.IGNORECASE)


def _is_profile_title(line: str, category: str | None) -> bool:
    """Titulli i një grupi lëndësh me zgjedhje (p.sh. 'IT e biznesit') brenda kategorisë C.

    Njihet nga forma: tekst i shkurtër pa numra, jo me shkronja të mëdha, pas titullit C.
    """
    return (
        category == "C"
        and 3 <= len(line) <= 80
        and not line.isupper()
        and not _NOT_A_PROFILE_TITLE.search(line)
    )


def norm(text: str) -> str:
    """Tekst i normalizuar për krahasim: shkronja të vogla, pa theksa, pa shenja."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def parse_pdf(path: Path) -> dict:
    """Kthen {'courses', 'total_ects', 'stated': {kategoria: ECTS}, 'unparsed': [rreshta të pakuptuar]}."""
    courses: list[dict] = []
    unparsed: list[str] = []
    stated: dict[str, int] = {}
    total = None
    category = None
    group = 0
    previous_nr = 0
    elective = False
    profile_title = None
    program = program_from_filename(path.name)

    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        for raw in (page.extract_text() or "").split("\n"):
            line = raw.strip()
            if not line:
                continue

            match = _CATEGORY.match(line)
            if match:
                category = match.group(1).upper()
                numbers = _STATED_ECTS.findall(match.group(2))
                if numbers:
                    stated[category] = int(numbers[-1])
                group, previous_nr, elective, profile_title = 0, 0, False, None
                continue
            if _TOTAL.match(line):
                total = int(_TOTAL.match(line).group(1))
                continue
            if _ELECTIVE_MARKER.search(line) and not line[0].isdigit():
                elective = True
                continue

            row = _ROW_WITH_YEAR.match(line)
            if row:
                nr, year, semester, name, ects = row.groups()
                year = 1 if year == "I" else 2
            else:
                row = _ROW_NO_YEAR.match(line)
                if not row:
                    if _LOOKS_LIKE_ROW.match(line):
                        unparsed.append(f"f. {page_number}: {line}")
                    elif _is_profile_title(line, category):
                        profile_title = line
                        group, previous_nr = group + 1, 0
                    continue
                nr, semester, name, ects = row.groups()
                year = 1
            if category is None:
                continue
            nr = int(nr)
            if nr <= previous_nr:  # numërimi rifillon: grup i ri lëndësh (p.sh. profil ose zgjedhje)
                group += 1
            previous_nr = nr
            courses.append(
                {
                    "program": program,
                    "source": path.name,
                    "page": page_number,
                    "category": category,
                    "category_label": CATEGORIES[category],
                    "group": group,
                    "elective": elective and category in {"B", "C"},
                    "profile": profile_title if category == "C" else None,
                    "year": year,
                    "semester": semester,  # "1", "2" ose "1-2" (brenda vitit)
                    "name": name.strip(),
                    "ects": int(ects),
                }
            )
    profiles = sorted({c["profile"] for c in courses if c["profile"]})
    return {"courses": courses, "total_ects": total, "stated": stated, "unparsed": unparsed, "profiles": profiles}


def build(data_dir: Path | None = None) -> dict:
    """Parson të gjitha PDF-të dhe kthen {program: {...}}."""
    data_dir = Path(data_dir or config.DATA_DIR)
    result: dict[str, dict] = {}
    for path in sorted(data_dir.glob("*.pdf")):
        parsed = parse_pdf(path)
        if not parsed["courses"]:
            continue
        # Nëse ECTS e parsuara për kategori nuk përputhen me ato të deklaruara, ndodh që
        # titujt A-E janë të çrregullt në tekstin e PDF-së. Atëherë kategoria nuk besohet.
        parsed["category_reliable"] = not _category_mismatches(parsed)
        if not parsed["category_reliable"]:
            for course in parsed["courses"]:
                course["category"] = UNKNOWN_CATEGORY
                course["category_label"] = "Kategoria e papërcaktuar"
                course["profile"] = None
            parsed["profiles"] = []
        result[program_from_filename(path.name)] = {"source": path.name, **parsed}
    return result


def _category_mismatches(data: dict) -> list[str]:
    problems = []
    for category, expected in data["stated"].items():
        got = sum(c["ects"] for c in data["courses"] if c["category"] == category)
        if category in ("A", "D", "E") and got != expected:
            problems.append(f"kategoria {category} ka {got} ECTS të parsuara, deklarohen {expected}")
        if category in ("B", "C") and got < expected:
            problems.append(f"kategoria {category} ka vetëm {got} ECTS të parsuara, deklarohen {expected}")
    return problems


def validate(curriculum: dict) -> list[str]:
    """Paralajmërimet për verifikim me dorë (kategori të pabesueshme, rreshta të pakuptuar)."""
    warnings = []
    for program, data in curriculum.items():
        if not data["courses"]:
            warnings.append(f"{program}: asnjë lëndë e parsuar")
        if not data.get("category_reliable", True):
            warnings.append(f"{program}: kategoritë A-E nuk janë të besueshme (titujt e tabelës janë në rend tjetër). Viti, semestri dhe ECTS janë nxjerrë, kategoria shënohet 'e papërcaktuar'")
        for line in data.get("unparsed", []):
            warnings.append(f"{program}: rresht i pakuptuar, shiko me dorë: {line}")
    return warnings


def save(curriculum: dict, path: Path | None = None) -> Path:
    path = Path(path or config.CURRICULUM_PATH)
    path.write_text(json.dumps(curriculum, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load(path: Path | None = None) -> dict:
    path = Path(path or config.CURRICULUM_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- pyetje mbi planin ----------
def programs(curriculum: dict) -> list[str]:
    return sorted(curriculum)


def years_for(curriculum: dict, program: str) -> list[int]:
    courses = curriculum.get(program, {}).get("courses", [])
    return sorted({c["year"] for c in courses}) or [1]


def profiles_for(curriculum: dict, program: str) -> list[str]:
    """Specializimet (grupet C me titull) që plani i këtij programi i lidh me lëndë."""
    return curriculum.get(program, {}).get("profiles", [])


def match_profile(curriculum: dict, program: str, specialization: str) -> str | None:
    """Specializimi i dhënuar nga studenti, i përputhur me një titull nga plani (ose None)."""
    wanted = norm(specialization or "")
    if not wanted:
        return None
    for title in profiles_for(curriculum, program):
        if norm(title) == wanted or wanted in norm(title) or norm(title) in wanted:
            return title
    return None


def courses_for(curriculum: dict, program: str, year: int, semester: int, specialization: str = "") -> list[dict]:
    """Lëndët e planit për vitin dhe semestrin (1 ose 2 brenda vitit). Janë 4 semestra gjithsej:
    2 te viti i parë dhe 2 te viti i dytë. Lëndët e vitit të plotë ('1-2', p.sh. Praktika dhe
    Punimi i diplomës) numërohen vetëm te semestri i dytë i vitit.

    Nëse programi ka specializime te plani, kthehen lëndët e përbashkëta dhe ato të
    specializimit të zgjedhur. Pa specializim të njohur kthehen vetëm të përbashkëtat.
    """
    courses = curriculum.get(program, {}).get("courses", [])
    chosen = match_profile(curriculum, program, specialization)
    return [
        c for c in courses
        if c["year"] == year
        and (c["semester"] == str(semester) or (c["semester"] == "1-2" and semester == 2))
        and (not c.get("profile") or c["profile"] == chosen)
    ]


def best_match(names: list[str], text: str) -> str | None:
    """Emri më i afërt nga lista me tekstin e shkruar (përafërsisht), ose None."""
    wanted = norm(text or "")
    if not wanted:
        return None
    best, best_score = None, 0.0
    for original in names:
        name = norm(original)
        if name == wanted:
            return original
        score = 0.9 if (wanted in name or name in wanted) and min(len(wanted), len(name)) >= 6 else SequenceMatcher(None, wanted, name).ratio()
        if score > best_score:
            best, best_score = original, score
    return best if best_score >= 0.72 else None


def find_course(curriculum: dict, program: str, text: str) -> dict | None:
    """Gjen një lëndë të programit nga emri i shkruar nga studenti (përafërsisht)."""
    courses = curriculum.get(program, {}).get("courses", [])
    name = best_match(sorted({c["name"] for c in courses}), text)
    return next((c for c in courses if c["name"] == name), None) if name else None


def canonical_course_name(curriculum: dict, text: str) -> str | None:
    """Emri zyrtar i lëndës nga çdo program (p.sh. për emrat e skedarëve të materialeve)."""
    return best_match(sorted({c["name"] for d in curriculum.values() for c in d["courses"]}), text)


def course_facts(course: dict) -> str:
    """Faktet zyrtare të një lënde nga plani, për prompt dhe për shfaqje."""
    semester = "viti i plotë (semestri 1-2)" if course["semester"] == "1-2" else f"semestri {course['semester']}"
    profile = f"; specializimi: {course['profile']}" if course.get("profile") else ""
    return (
        f"{course['name']}: {course['ects']} ECTS; viti {course['year']}, {semester}; "
        f"{course['category_label'].lower()}{profile} [{course['program']}, f. {course['page']}]"
    )


def format_courses(courses: list[dict]) -> str:
    """Teksti i lëndëve për prompt, i grupuar sipas kategorisë, me faqen e burimit."""
    lines = []
    for category in [*CATEGORIES, UNKNOWN_CATEGORY]:
        label = CATEGORIES.get(category, "Lëndë (kategoria nuk përcaktohet nga dokumenti)")
        subset = [c for c in courses if c["category"] == category]
        if not subset:
            continue
        lines.append(f"{label}:")
        for c in subset:
            if c.get("profile"):
                note = f" (specializimi: {c['profile']})"
            elif c["category"] in {"B", "C"} and (c["elective"] or c["group"] > 0):
                note = " (me zgjedhje ose sipas profilit)"
            else:
                note = ""
            page = f", f. {c['page']}" if c.get("page") else ""
            lines.append(f"- {c['name']}, {c['ects']} ECTS{note} [{c['program']}{page}]")
    return "\n".join(lines)
