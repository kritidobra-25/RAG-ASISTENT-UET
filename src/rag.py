"""
Logjika kryesore e asistentit (Faza B e arkitekturës):
pyetje -> embedding -> kërkim në ChromaDB -> prompt me kontekst -> përgjigje.

Përdoret njësoj nga query.py (terminali), app.py (ndërfaqja) dhe evaluate.py.
"""

import re
import time
import unicodedata

import chromadb

import config
import llm
from student_profile import is_empty, profile_to_query, profile_to_text

# Fraza me të cilën asistenti fillon kur nuk e gjen informacionin.
# evaluate.py e përdor për të numëruar automatikisht refuzimet.
REFUSAL_PREFIX = "Nuk e gjej këtë informacion në dokumentet e disponueshme."

SYSTEM_PROMPT = f"""Je asistenti inteligjent i Universitetit Europian të Tiranës (UET) për studentët aktualë dhe kandidatët e mundshëm.
Përgjigju VETËM duke u bazuar në fragmentet e dokumenteve zyrtare që jepen në mesazhin e përdoruesit.

Rregullat:
1. Mos përdor njohuri nga jashtë fragmenteve dhe mos shpik emra, numra, data apo kredite.
2. Çdo fragment fillon me emrin e programit. Nëse pyetja përmend një program, përdor vetëm fragmentet e atij programi.
3. Nëse pyetja nuk e specifikon programin dhe informacioni ndryshon mes programeve, trego përgjigjen për secilin program veç e veç.
4. Nëse informacioni nuk gjendet në fragmente, fillo përgjigjen saktësisht me: "{REFUSAL_PREFIX}" dhe këshillo studentin të kontaktojë administratën e UET-së.
5. Përgjigju në shqip, shkurt dhe qartë. Për kredite ECTS, vit dhe semestër, përdor vlerat saktësisht siç shfaqen në fragmente.
6. Përjashtim nga rregulli 4: nëse mesazhi është vetëm përshëndetje ose bisedë e shkurtër (p.sh. "përshëndetje", "si je", "faleminderit"), mos përdor frazën e refuzimit. Përgjigju shkurt dhe miqësisht dhe thuaji se mund ta ndihmosh me pyetje rreth programeve të studimit të UET-së, si profilet, lëndët, kreditet ECTS dhe mundësitë e punësimit. Pyetjet faktike që nuk gjenden në fragmente vazhdojnë të marrin përgjigjen e refuzimit.
7. Nëse mesazhi përmban "Profili i studentit", përdore për të personalizuar përgjigjen: rendit sipas rëndësisë programet dhe lëndët që i përshtaten interesit, nivelit, eksperiencës, objektivit dhe preferencës së tij. Profili nuk është burim fakti: emrat e programeve, lëndët, kreditet ECTS dhe mundësitë e punësimit merren vetëm nga fragmentet. Nëse asnjë program nuk i përshtatet qartë profilit, thuaje hapur dhe trego programin më të afërt që gjendet në fragmente."""

PATH_INSTRUCTION = """Detyra: përgatit "Rrugën akademike të sugjeruar" për këtë student, duke u bazuar në profilin e tij dhe vetëm në fragmentet e dokumenteve. Përdor saktësisht këtë format:

**Rruga akademike e sugjeruar**
1. **Programi:** programi më i përshtatshëm dhe një fjali pse i përshtatet profilit
2. **Lënda kyçe 1:** emri, viti/semestri dhe ECTS siç shfaqen në fragmente
3. **Lënda kyçe 2:** po ashtu
4. **Specializimi ose profili:** nëse fragmentet përmendin profile ose drejtime, trego atë më të përshtatshmin; përndryshe shkruaj "Nuk përmendet në dokumente"
5. **Mundësi karriere:** vetëm ato që përmenden te fragmentet për atë program

Në fund shto një pasazh të shkurtër "Alternativa" me një program tjetër që i afrohet profilit, nëse ka. Mos shpik lëndë ose programe që nuk janë te fragmentet."""


# Fraza hapëse dhe bisede e shkurtër (pa theksa, të vogla). Për to nuk kërkohet në
# dokumente dhe nuk shfaqen burime.
SMALL_TALK = (
    "pershendetje", "pershendetje te gjitheve", "tung", "tungjatjeta", "hej", "hello", "hi",
    "mire se vjen", "mireserdhe", "miredita", "mirembrema", "mirmengjes", "mire mengjes",
    "si je", "si jeni", "si po ia kalon", "çfare ka", "cfare ka", "si ja kalon",
    "faleminderit", "falemnderit", "shume faleminderit", "rrofsh", "flm",
    "mire", "ne rregull", "ok", "okay", "pa pa", "mirupafshim", "naten e mire", "ciao",
    "kush je", "kush jeni", "si quhesh", "si quheni",
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text).split())


def is_small_talk(question: str) -> bool:
    """True nëse mesazhi është vetëm përshëndetje ose bisedë e shkurtër."""
    normalized = _normalize(question)
    phrases = {_normalize(p) for p in SMALL_TALK}
    if normalized in phrases:
        return True
    words = normalized.split()
    # p.sh. "përshëndetje, si je?" ose "faleminderit shumë"
    return 0 < len(words) <= 4 and all(
        word in {w for p in phrases for w in p.split()} for word in words
    )


class KnowledgeBaseMissingError(RuntimeError):
    pass


class RagAssistant:
    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        try:
            self.collection = client.get_collection(config.COLLECTION_NAME)
        except Exception as error:
            raise KnowledgeBaseMissingError(
                "Baza e njohurive nuk ekziston ende. "
                "Ekzekuto fillimisht: python src/build_index.py"
            ) from error
        if self.collection.count() == 0:
            raise KnowledgeBaseMissingError(
                "Baza e njohurive është bosh. Ekzekuto: python src/build_index.py"
            )

    def stats(self) -> dict:
        """Numri i segmenteve dhe lista e programeve/dokumenteve në bazë."""
        data = self.collection.get(include=["metadatas"])
        programs = sorted({m["program"] for m in data["metadatas"]})
        return {"segments": self.collection.count(), "programs": programs}

    def retrieve(self, question: str, k: int = config.TOP_K) -> list[dict]:
        """Kthen k segmentet më të ngjashme semantikisht me pyetjen."""
        vector = llm.embed_texts([question])[0]
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunks.append(
                {
                    "text": text,
                    "program": meta["program"],
                    "source": meta["source"],
                    "distance": float(distance),
                }
            )
        return chunks

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        blocks = []
        for number, chunk in enumerate(chunks, start=1):
            blocks.append(f"[Fragmenti {number} | Dokumenti: {chunk['source']}]\n{chunk['text']}")
        return "\n\n---\n\n".join(blocks)

    def answer(
        self,
        question: str,
        k: int = config.TOP_K,
        profile: dict | None = None,
        path_mode: bool = False,
    ) -> dict:
        """Përgjigjet një pyetjeje. Kthen përgjigjen, burimet dhe kohët e matura.

        profile: profili i studentit (opsional), përdoret për kërkim dhe për prompt.
        path_mode: kërkon formatin "Rruga akademike e sugjeruar" (kërkon profil).
        """
        started = time.perf_counter()
        small_talk = is_small_talk(question) and not path_mode
        use_profile = not small_talk and not is_empty(profile)
        search_text = question
        if use_profile:
            search_text = f"{question}. {profile_to_query(profile)}"
        if path_mode:
            k = max(k, 8)
        chunks = [] if small_talk else self.retrieve(search_text, k)
        retrieval_done = time.perf_counter()

        if small_talk:
            user_prompt = f"Mesazhi i studentit (bisedë e shkurtër, pa fragmente): {question}"
        else:
            user_prompt = f"Fragmentet e dokumenteve:\n\n{self._build_context(chunks)}\n\n"
            if use_profile:
                user_prompt += f"Profili i studentit:\n{profile_to_text(profile)}\n\n"
            if path_mode:
                user_prompt += f"{PATH_INSTRUCTION}\n\n"
            user_prompt += f"Pyetja e studentit: {question}"
        text = llm.chat_completion(SYSTEM_PROMPT, user_prompt)
        finished = time.perf_counter()

        sources = []
        seen = set()
        for chunk in chunks:
            key = (chunk["program"], chunk["source"])
            if key not in seen:
                seen.add(key)
                sources.append({"program": chunk["program"], "source": chunk["source"]})

        return {
            "question": question,
            "answer": text,
            "refused": text.strip().startswith(REFUSAL_PREFIX),
            "sources": sources,
            "chunks": chunks,
            "retrieval_seconds": retrieval_done - started,
            "generation_seconds": finished - retrieval_done,
            "total_seconds": finished - started,
        }
