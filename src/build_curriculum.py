"""
Ndërton data/curriculum.json nga tabelat e planeve mësimore në PDF.

    python src/build_curriculum.py

Pas ekzekutimit lexo paralajmërimet: ato tregojnë ku ECTS e parsuara nuk
përputhen me ato të deklaruara te tabela, që plani të verifikohet me dorë.
"""

import sys

import curriculum

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = curriculum.build()
    path = curriculum.save(data)
    print(f"U ruajt: {path}")
    for program, info in data.items():
        by_sem: dict[tuple, int] = {}
        for c in info["courses"]:
            key = (c["year"], c["semester"])
            by_sem[key] = by_sem.get(key, 0) + 1
        print(f"- {program}: {len(info['courses'])} lëndë, totali {info['total_ects']} ECTS, (viti, semestri): {dict(sorted(by_sem.items()))}")
    warnings = curriculum.validate(data)
    print(f"\nParalajmërime ({len(warnings)}):")
    for w in warnings:
        print(" !", w)
