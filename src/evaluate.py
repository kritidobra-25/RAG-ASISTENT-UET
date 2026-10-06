"""
Hapi 4: vlerësimi me setin e pyetjeve testuese.

Ekzekuton çdo pyetje nga eval/test_questions.csv dhe ruan rezultatet te
eval/rezultatet.csv. Çdo pyetje kalon në një nga dy rrugët (kolona 'rruga'):

  - rag:      RagAssistant.answer(), rrjedha bazë RAG pa personalizim
  - asistent: Advisor.respond(), rrjedha e plotë si në aplikacion (klasifikimi i
              qëllimit, profili, studenti aktual, Study Mode, pyetjet ndjekëse)

Profilet testuese ndodhen te eval/test_profiles.json (kolona 'profili').
Kolona 'pyetja_paraprake' simulon një bisedë me dy hapa: pyetja paraprake
dërgohet e para dhe përgjigjja e saj përdoret si historik për pyetjen e testit.
Çdo pyetje nis me sesion të ri (pa historik nga pyetjet e tjera).

Pas ekzekutimit, hap skedarin në Excel dhe plotëso dorazi kolonat:
  - vleresimi:          "e saktë", "pjesërisht e saktë" ose "e pasaktë"
  - burimi_mbeshtet:    "po" ose "jo" (a e mbështet burimi i shfaqur përgjigjen)
  - halucinacion:       "po" ose "jo" (a ka emra, numra, ECTS që nuk janë te dokumentet)
  - personalizim:       "po", "jo" ose bosh (vetëm për pyetjet me profil)

    python src/evaluate.py
    python src/evaluate.py --ids 26-35      (vetëm disa pyetje)
"""

import argparse
import copy
import csv
import json
import sys
import time

import config
import curriculum as curriculum_module
import llm
from advisor import Advisor
from rag import KnowledgeBaseMissingError, RagAssistant
from student_profile import merge_profile, new_academic, new_profile

QUESTIONS_FILE = config.EVAL_DIR / "test_questions.csv"
PROFILES_FILE = config.EVAL_DIR / "test_profiles.json"
RESULTS_FILE = config.EVAL_DIR / "rezultatet.csv"

FIELDS = [
    "id",
    "kategoria",
    "rruga",
    "profili",
    "lenda_study",
    "detyra",
    "pyetja_paraprake",
    "pyetja",
    "pergjigja_e_pritur",
    "brenda_korpusit",
    "pergjigja_e_asistentit",
    "burimet",
    "intent",
    "pyetje_ndjekese",
    "kohe_sekonda",
    "refuzoi_automatikisht",
    "vleresimi",
    "burimi_mbeshtet",
    "halucinacion",
    "personalizim",
    "shenime",
]


def parse_ids(spec: str | None) -> set[str] | None:
    """'26-35,40' -> {'26', ..., '35', '40'}"""
    if not spec:
        return None
    ids: set[str] = set()
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-")
            ids.update(str(n) for n in range(int(start), int(end) + 1))
        elif part.strip():
            ids.add(part.strip())
    return ids


def load_profiles() -> dict:
    if not PROFILES_FILE.exists():
        return {}
    return json.loads(PROFILES_FILE.read_text(encoding="utf-8"))


def build_session(spec: dict | None) -> dict:
    """Krijon gjendjen e një sesioni të ri nga përshkrimi i profilit testues."""
    spec = spec or {}
    profile, _ = merge_profile(new_profile(), spec.get("profile") or {})
    academic = new_academic()
    academic.update(spec.get("academic") or {})
    return {"role": spec.get("role", "prospective"), "profile": profile, "academic": academic}


def format_sources(result: dict) -> str:
    parts = []
    for source in result.get("sources", []):
        name = source.get("source") or source.get("program", "")
        pages = source.get("pages") or []
        parts.append(f"{name} (f. {', '.join(str(p) for p in pages)})" if pages else name)
    return " | ".join(parts)


def run_advisor(advisor: Advisor, item: dict, session: dict, curriculum: dict) -> dict:
    """Ekzekuton pyetjen përmes rrjedhës së plotë, si te aplikacioni."""
    role = session["role"]
    study_course = item.get("lenda_study") or None
    forced_task = item.get("detyra") or None
    profile = session["profile"]
    history: list[dict] = []

    common = dict(
        role=role,
        academic=session["academic"] if role == "current" else None,
        curriculum=curriculum,
        study_course=study_course if role == "current" else None,
    )

    # Bisedë me dy hapa: pyetja paraprake nuk matet, shërben vetëm si historik.
    previous = (item.get("pyetja_paraprake") or "").strip()
    if previous:
        first = advisor.respond(previous, profile=profile, history=[], followups_asked=0, **common)
        profile = first.get("profile", profile)
        history = [{"role": "user", "content": previous}, {"role": "assistant", "content": first["answer"]}]

    started = time.perf_counter()
    result = advisor.respond(
        item["pyetja"], profile=profile, history=history, followups_asked=0, forced_task=forced_task, **common
    )
    result["total_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Vlerësimi i UETassist me pyetjet testuese.")
    parser.add_argument("--ids", help="Vetëm këto pyetje, p.sh. 26-35,40")
    args = parser.parse_args()
    selected = parse_ids(args.ids)

    if not QUESTIONS_FILE.exists():
        print(f"Mungon skedari me pyetje: {QUESTIONS_FILE}")
        sys.exit(1)

    try:
        assistant = RagAssistant()
    except KnowledgeBaseMissingError as error:
        print(f"Gabim: {error}")
        sys.exit(1)
    advisor = Advisor(assistant)

    curriculum = curriculum_module.load() or curriculum_module.build()
    profiles = load_profiles()

    with open(QUESTIONS_FILE, newline="", encoding="utf-8-sig") as file:
        questions = [q for q in csv.DictReader(file) if not selected or q["id"] in selected]

    rows = []
    for number, item in enumerate(questions, start=1):
        route = (item.get("rruga") or "rag").strip().lower()
        print(f"[{number}/{len(questions)}] ({route}) {item['pyetja']}")

        profile_key = (item.get("profili") or "").strip()
        if profile_key and profile_key not in profiles:
            print(f"  Kujdes: profili '{profile_key}' nuk gjendet te {PROFILES_FILE.name}; përdoret profil bosh.")

        try:
            if route == "asistent":
                session = build_session(profiles.get(profile_key))
                result = run_advisor(advisor, item, session, curriculum)
            else:
                result = assistant.answer(item["pyetja"])
        except llm.MissingApiKeyError as error:
            print(f"\nGabim: {error}")
            sys.exit(1)
        except Exception as error:  # gabim rrjeti/kuote: regjistrohet dhe vazhdon
            print(f"  Gabim: {type(error).__name__}: {error}")
            result = {"answer": f"GABIM: {type(error).__name__}: {error}", "sources": [], "refused": False, "total_seconds": None}

        rows.append(
            {
                **{key: item.get(key, "") for key in FIELDS},
                "rruga": route,
                "pergjigja_e_asistentit": result["answer"],
                "burimet": format_sources(result),
                "intent": result.get("intent", ""),
                "pyetje_ndjekese": "po" if result.get("follow_up") else "jo",
                "kohe_sekonda": "" if result.get("total_seconds") is None else f"{result['total_seconds']:.2f}",
                "refuzoi_automatikisht": "po" if result.get("refused") else "jo",
                "vleresimi": "",
                "burimi_mbeshtet": "",
                "halucinacion": "",
                "personalizim": "",
                "shenime": "",
            }
        )

    # utf-8-sig që Excel t'i hapë saktë shkronjat shqipe (ë, ç)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---------- Përmbledhje e shpejtë ----------
    times = [float(r["kohe_sekonda"]) for r in rows if r["kohe_sekonda"]]
    inside = [r for r in rows if r["brenda_korpusit"].lower() == "po"]
    outside = [r for r in rows if r["brenda_korpusit"].lower() == "jo"]
    false_refusals = [r for r in inside if r["refuzoi_automatikisht"] == "po"]
    correct_refusals = [r for r in outside if r["refuzoi_automatikisht"] == "po"]

    print("\n=== Përmbledhje ===")
    print(f"Pyetje gjithsej: {len(rows)}")
    if times:
        print(f"Koha mesatare e përgjigjes: {sum(times) / len(times):.2f} s (min {min(times):.2f} s, maks {max(times):.2f} s)")
        for route in ("rag", "asistent"):
            route_times = [float(r["kohe_sekonda"]) for r in rows if r["rruga"] == route and r["kohe_sekonda"]]
            if route_times:
                print(f"  - rruga '{route}': {sum(route_times) / len(route_times):.2f} s mesatarisht ({len(route_times)} pyetje)")
    if outside:
        print(f"Pyetje jashtë korpusit të refuzuara automatikisht: {len(correct_refusals)}/{len(outside)}")
        print("  (Disa abstenime, p.sh. në Study Mode, nuk e përdorin prefiksin e refuzimit: kontrolloji dorazi.)")
    if inside:
        print(f"Pyetje brenda korpusit të refuzuara gabimisht: {len(false_refusals)}/{len(inside)}")
    print(f"\nRezultatet u ruajtën te: {RESULTS_FILE}")
    print("Hapi tjetër: plotëso kolonat 'vleresimi', 'burimi_mbeshtet', 'halucinacion' dhe 'personalizim'.")


if __name__ == "__main__":
    main()
