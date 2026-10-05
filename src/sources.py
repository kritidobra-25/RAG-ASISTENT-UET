"""
Atribuimi i burimeve: nga segmentet e gjetura dhe nga teksti i përgjigjes ndërtohet lista
"Burimi: Titulli, faqja N". Titulli është emri i programit (dokumentet zyrtare) ose emri
i skedarit (materialet e lëndëve). Asnjë ID ose etiketë e brendshme nuk shfaqet.
"""

import re


def _title(chunk: dict) -> str:
    return chunk.get("title") or chunk["program"]


def attribute_sources(text: str, chunks: list[dict]) -> list[dict]:
    """Burimet që përgjigjja i citon sipas titullit dhe faqes, p.sh. "(Titulli, f. 2)".

    Nëse përgjigjja nuk emërton asnjë burim, kthehen deri në 3 burimet e para të kërkimit,
    që lexuesi të mos mbetet pa kontekst. Çelësi "program" mban titullin (përputhshmëri me ndërfaqen).
    """
    first_source: dict[str, dict] = {}
    for chunk in chunks:
        first_source.setdefault(_title(chunk), chunk)
    titles = [t for t in first_source if t in text] or list(first_source)[:3]

    sources = []
    for title in titles:
        pages: list[int] = []
        for match in re.finditer(re.escape(title) + r"\s*,?\s*(?:f\.|faqja|faqe)\s*(\d+(?:\s*(?:,|-|–)\s*\d+)*)", text):
            pages += [int(n) for n in re.findall(r"\d+", match.group(1))]
        sources.append(
            {
                "program": title,
                "source": first_source[title]["source"],
                "pages": sorted(set(pages)),
                "doc_type": first_source[title].get("doc_type", "program"),
            }
        )
    return sources


def format_sources(sources: list[dict]) -> str:
    lines = ["## Burimet"]
    for entry in sources:
        if entry["pages"]:
            lines.append(f"- Burimi: {entry['program']}, faqja {', '.join(str(p) for p in entry['pages'])}")
        else:
            lines.append(f"- Burimi: {entry['program']}")
    return "\n".join(lines)
