"""
Logjika kryesore e asistentit (Faza B e arkitekturës):
pyetje -> embedding -> kërkim në ChromaDB -> prompt me kontekst -> përgjigje.

Përdoret njësoj nga query.py (terminali), app.py (ndërfaqja) dhe evaluate.py.
"""

import time

import chromadb

import config
import llm
import student_profile
from accounts import ROLE_ACTUAL, ROLE_POTENTIAL

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
6. Përjashtim nga rregulli 4: nëse mesazhi është vetëm përshëndetje ose bisedë e shkurtër (p.sh. "përshëndetje", "si je", "faleminderit"), mos përdor frazën e refuzimit. Përgjigju shkurt dhe miqësisht dhe thuaji se mund ta ndihmosh me pyetje rreth programeve të studimit të UET-së, si profilet, lëndët, kreditet ECTS dhe mundësitë e punësimit. Pyetjet faktike që nuk gjenden në fragmente vazhdojnë të marrin përgjigjen e refuzimit."""

# Rrugët e përdorimit (Academic Information, Study Assistant, Program Guidance).
MODE_ACADEMIC = "academic"
MODE_STUDY = "study"
MODE_GUIDANCE = "guidance"

MODE_PROMPTS = {
    MODE_ACADEMIC: (
        "Mënyra: informacion akademik. Përgjigju pyetjes për lëndët, kreditet ECTS, "
        "vitin, semestrin dhe rregullat e programit të studentit."
    ),
    MODE_STUDY: (
        "Mënyra: asistent studimi. Ndihmo studentin të organizojë studimin (p.sh. "
        "çfarë lëndësh ka në semestër, sa kredite ECTS janë gjithsej, si t'i grupojë), "
        "por vetëm me të dhëna që gjenden në fragmente. Mos jep këshilla me numra apo "
        "orare që nuk janë në dokumente."
    ),
    MODE_GUIDANCE: (
        "Mënyra: orientim për programet. Ndihmo kandidatin të kuptojë dhe krahasojë "
        "programet, profilet, lëndët dhe mundësitë e punësimit që përmenden në fragmente."
    ),
}

PROFILE_RULE = (
    "Profili i studentit jepet në mesazhin e përdoruesit. Përdore vetëm për të zgjedhur "
    "çfarë është e rëndësishme për të. Mos e trajto si burim fakti për programet."
)

RECOMMENDATION_QUESTION = (
    "Cilat programe të UET-së i përshtaten këtij kandidati më shumë? Rendit deri në 3 "
    "programe nga më i përshtatshmi. Për secilin shpjego shkurt pse, duke u bazuar te "
    "profilet, lëndët dhe mundësitë e punësimit në fragmente."
)

CURRENT_COURSES_QUESTION = (
    "Cilat janë lëndët e planit mësimor për vitin {year}, semestri {semester}? "
    "Për secilën trego kreditet ECTS dhe, nëse shfaqet, llojin (e detyrueshme, me zgjedhje)."
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

    def retrieve(self, question: str, k: int = config.TOP_K, program: str | None = None) -> list[dict]:
        """Kthen k segmentet më të ngjashme semantikisht me pyetjen.

        Me `program`, kërkimi kufizohet te segmentet e atij programi.
        """
        vector = llm.embed_texts([question])[0]
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            where={"program": program} if program else None,
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
        role: str | None = None,
        profile: dict | None = None,
        mode: str | None = None,
    ) -> dict:
        """Përgjigjet një pyetjeje. Kthen përgjigjen, burimet dhe kohët e matura.

        Pa `role` dhe `profile` sjellja është e pandryshuar (query.py, evaluate.py).
        Me profil, kërkimi përshtatet me studentin dhe përgjigjja merr parasysh profilin.
        """
        program = self._program_filter(role, profile)
        retrieval_query = self._build_retrieval_query(question, role, profile)
        return self._run(question, retrieval_query, k, program, role, profile, mode)

    def current_courses(self, profile: dict, k: int = 8) -> dict:
        """Përputhja me planin mësimor: lëndët e vitit dhe semestrit të studentit aktual."""
        question = CURRENT_COURSES_QUESTION.format(year=profile["viti"], semester=profile["semestri"])
        retrieval_query = (
            f"{profile['programi']} plani mësimor viti {profile['viti']} "
            f"semestri {profile['semestri']} lëndët kredite ECTS"
        )
        return self._run(question, retrieval_query, k, profile["programi"], ROLE_ACTUAL, profile, MODE_ACADEMIC)

    def recommend_programs(self, profile: dict, k: int = 8) -> dict:
        """Motori i rekomandimeve: programet që i përshtaten kandidatit (student potencial)."""
        retrieval_query = self._build_retrieval_query(RECOMMENDATION_QUESTION, ROLE_POTENTIAL, profile)
        result = self._run(
            RECOMMENDATION_QUESTION, retrieval_query, k, None, ROLE_POTENTIAL, profile, MODE_GUIDANCE
        )
        return result

    @staticmethod
    def _program_filter(role: str | None, profile: dict | None) -> str | None:
        """Studenti aktual merr përgjigje vetëm nga programi i vet."""
        if role == ROLE_ACTUAL and profile:
            return profile.get("programi") or None
        return None

    @staticmethod
    def _build_retrieval_query(question: str, role: str | None, profile: dict | None) -> str:
        """Përpunimi i pyetjes: pyetja pasurohet me fjalë kyçe nga profili."""
        if not profile:
            return question
        if role == ROLE_ACTUAL:
            hints = [profile.get("programi"), profile.get("specializimi")]
        else:
            hints = [profile.get("fusha"), profile.get("interesat"), profile.get("objektivat")]
        hints = [h for h in hints if h and h != "Tjetër"]
        return f"{question}\n{' '.join(hints)}" if hints else question

    def _run(
        self,
        question: str,
        retrieval_query: str,
        k: int,
        program: str | None,
        role: str | None,
        profile: dict | None,
        mode: str | None,
    ) -> dict:
        started = time.perf_counter()
        chunks = self.retrieve(retrieval_query, k, program)
        retrieval_done = time.perf_counter()

        system_prompt = SYSTEM_PROMPT
        user_prompt = f"Fragmentet e dokumenteve:\n\n{self._build_context(chunks)}\n\n"
        if mode in MODE_PROMPTS:
            system_prompt += f"\n7. {MODE_PROMPTS[mode]}"
        if role and profile:
            system_prompt += f"\n8. {PROFILE_RULE}"
            user_prompt += f"{student_profile.describe(role, profile)}\n\n"
        user_prompt += f"Pyetja e studentit: {question}"
        text = llm.chat_completion(system_prompt, user_prompt)
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
