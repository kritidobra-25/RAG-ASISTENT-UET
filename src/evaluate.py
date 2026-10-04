"""
Hapi 4: vlerësimi me setin e pyetjeve testuese (Kapitulli 3.8 i tezës).

Ekzekuton çdo pyetje nga eval/test_questions.csv dhe ruan rezultatet te
eval/rezultatet.csv. Pastaj hap skedarin në Excel dhe plotëso dorazi kolonat:
  - vleresimi: "e saktë", "pjesërisht e saktë" ose "e pasaktë"
  - burimi_mbeshtet: "po" ose "jo" (a e mbështet burimi i shfaqur përgjigjen)

    python src/evaluate.py
"""

import csv
import sys

import config
import llm
from rag import KnowledgeBaseMissingError, RagAssistant

QUESTIONS_FILE = config.EVAL_DIR / "test_questions.csv"
RESULTS_FILE = config.EVAL_DIR / "rezultatet.csv"

FIELDS = [
    "id",
    "kategoria",
    "pyetja",
    "pergjigja_e_pritur",
    "brenda_korpusit",
    "pergjigja_e_asistentit",
    "burimet",
    "kohe_sekonda",
    "refuzoi_automatikisht",
    "vleresimi",
    "burimi_mbeshtet",
    "shenime",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not QUESTIONS_FILE.exists():
        print(f"Mungon skedari me pyetje: {QUESTIONS_FILE}")
        sys.exit(1)

    try:
        assistant = RagAssistant()
    except KnowledgeBaseMissingError as error:
        print(f"Gabim: {error}")
        sys.exit(1)

    with open(QUESTIONS_FILE, newline="", encoding="utf-8-sig") as file:
        questions = list(csv.DictReader(file))

    rows = []
    for number, item in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {item['pyetja']}")
        try:
            result = assistant.answer(item["pyetja"])
        except llm.MissingApiKeyError as error:
            print(f"\nGabim: {error}")
            sys.exit(1)

        rows.append(
            {
                "id": item["id"],
                "kategoria": item["kategoria"],
                "pyetja": item["pyetja"],
                "pergjigja_e_pritur": item["pergjigja_e_pritur"],
                "brenda_korpusit": item["brenda_korpusit"],
                "pergjigja_e_asistentit": result["answer"],
                "burimet": " | ".join(s["source"] for s in result["sources"]),
                "kohe_sekonda": f"{result['total_seconds']:.2f}",
                "refuzoi_automatikisht": "po" if result["refused"] else "jo",
                "vleresimi": "",
                "burimi_mbeshtet": "",
                "shenime": "",
            }
        )

    # utf-8-sig që Excel t'i hapë saktë shkronjat shqipe (ë, ç)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---------- Përmbledhje e shpejtë ----------
    times = [float(r["kohe_sekonda"]) for r in rows]
    inside = [r for r in rows if r["brenda_korpusit"].lower() == "po"]
    outside = [r for r in rows if r["brenda_korpusit"].lower() == "jo"]
    false_refusals = [r for r in inside if r["refuzoi_automatikisht"] == "po"]
    correct_refusals = [r for r in outside if r["refuzoi_automatikisht"] == "po"]

    print("\n=== Përmbledhje ===")
    print(f"Pyetje gjithsej: {len(rows)}")
    print(f"Koha mesatare e përgjigjes: {sum(times) / len(times):.2f} s (maksimumi {max(times):.2f} s)")
    if outside:
        print(f"Pyetje jashtë korpusit që u refuzuan saktë: {len(correct_refusals)}/{len(outside)}")
    if inside:
        print(f"Pyetje brenda korpusit që u refuzuan gabimisht: {len(false_refusals)}/{len(inside)}")
    print(f"\nRezultatet u ruajtën te: {RESULTS_FILE}")
    print("Hapi tjetër: hap skedarin në Excel dhe plotëso kolonat 'vleresimi' dhe 'burimi_mbeshtet'.")


if __name__ == "__main__":
    main()
