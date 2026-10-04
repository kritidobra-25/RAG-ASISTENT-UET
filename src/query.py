"""
Hapi 3: testimi i asistentit nga terminali.

Bisedë e vazhdueshme (shkruaj "dil" për të mbyllur):
    python src/query.py

Një pyetje e vetme:
    python src/query.py "Sa ECTS ka lënda Inteligjenca artificiale në Inxhinieri Informatike?"
"""

import sys

import llm
from rag import KnowledgeBaseMissingError, RagAssistant


def print_result(result: dict) -> None:
    print("\nPërgjigjja:\n")
    print(result["answer"])
    print("\nBurimet:")
    for source in result["sources"]:
        print(f"  - {source['program']}  ({source['source']})")
    print(
        f"\nKoha: {result['total_seconds']:.1f} s "
        f"(kërkimi {result['retrieval_seconds']:.1f} s, "
        f"gjenerimi {result['generation_seconds']:.1f} s)"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        assistant = RagAssistant()
    except KnowledgeBaseMissingError as error:
        print(f"Gabim: {error}")
        sys.exit(1)

    try:
        if len(sys.argv) > 1:
            print_result(assistant.answer(" ".join(sys.argv[1:])))
            return

        print("Asistenti i UET. Shkruaj pyetjen tënde ose 'dil' për të mbyllur.")
        while True:
            question = input("\nPyetja: ").strip()
            if question.lower() in {"dil", "exit", "quit"}:
                break
            if not question:
                continue
            print_result(assistant.answer(question))
    except llm.MissingApiKeyError as error:
        print(f"\nGabim: {error}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nU mbyll.")


if __name__ == "__main__":
    main()
