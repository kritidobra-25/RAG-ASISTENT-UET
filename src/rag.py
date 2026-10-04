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
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunks.append(
                {
                    "id": chunk_id,
                    "page": meta.get("page") or None,
                    "text": text,
                    "program": meta["program"],
                    "source": meta["source"],
                    "distance": float(distance),
                }
            )
        return chunks

    def retrieve_multi(self, queries: list[str], k: int = config.TOP_K, limit: int = 12) -> list[dict]:
        """Kërkon për disa pyetje dhe i bashkon segmentet pa dublikatë.

        Merr radhazi nga lista e secilës pyetje (round-robin), që krahasimi i dy
        programeve të marrë segmente nga të dyja, jo vetëm nga më i ngjashmi.
        """
        per_query = [self.retrieve(q, k) for q in queries if q.strip()]
        merged, seen = [], set()
        for rank in range(k):
            for results in per_query:
                if rank < len(results) and results[rank]["id"] not in seen:
                    seen.add(results[rank]["id"])
                    merged.append(results[rank])
        return merged[:limit]

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        blocks = []
        for number, chunk in enumerate(chunks, start=1):
            page = f" | Faqja: {chunk['page']}" if chunk.get("page") else ""
            blocks.append(f"[Fragmenti {number} | Dokumenti: {chunk['source']}{page}]\n{chunk['text']}")
        return "\n\n---\n\n".join(blocks)

    def answer(self, question: str, k: int = config.TOP_K) -> dict:
        """Përgjigjet një pyetjeje. Kthen përgjigjen, burimet dhe kohët e matura."""
        started = time.perf_counter()
        small_talk = is_small_talk(question)
        chunks = [] if small_talk else self.retrieve(question, k)
        retrieval_done = time.perf_counter()

        if small_talk:
            user_prompt = f"Mesazhi i studentit (bisedë e shkurtër, pa fragmente): {question}"
        else:
            user_prompt = (
                f"Fragmentet e dokumenteve:\n\n{self._build_context(chunks)}\n\n"
                f"Pyetja e studentit: {question}"
            )
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
