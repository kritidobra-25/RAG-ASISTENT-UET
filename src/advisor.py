"""
Këshilluesi akademik: RAG + profil studenti + personalizim.

Mbi RagAssistant (që mbetet i pandryshuar për pyetjet faktike) shton:
1. klasifikimin e qëllimit dhe nxjerrjen e profilit nga mesazhi (intent.py),
2. pyetje ndjekëse kur mungon informacion i nevojshëm,
3. kërkim të personalizuar (pyetja + profili + pyetje shtesë),
4. përgjigje me strukturë sipas llojit: rekomandim, analizë boshllëqesh,
   krahasim, rrugë akademike, what-if,
5. atribuimin e burimeve nga segmentet e përdorura realisht.

Pyetjet faktike (tarifa, kalendar, informacion programi etj.) kalojnë te
RagAssistant.answer() pa asnjë ndryshim.
"""


import config
import intent as intent_module
import llm
from rag import REFUSAL_PREFIX, SYSTEM_PROMPT, RagAssistant, is_small_talk, sanitize_labels
from sources import attribute_sources, format_sources
from student_assistant import CurrentStudentAssistant
from student_profile import (
    follow_up_question,
    merge_profile,
    missing_for,
    new_profile,
    profile_to_query,
    profile_to_text,
)

# Pas kaq pyetjesh ndjekëse radhazi, këshilluesi vazhdon me atë që di.
MAX_FOLLOW_UPS = 2

COMMON_RULES = """Rregulla shtesë për përgjigjet e personalizuara:
- Faktet për UET-në (programe, lëndë, ECTS, profile, mundësi karriere) merren VETËM nga fragmentet dhe shoqërohen me burimin në kllapa, p.sh. (Emri i programit, f. 2), ose "Burimi: emri i programit" kur faqja mungon. Mos shpik programe, lëndë, tarifa, kushte pranimi apo mundësi karriere.
- Profili i studentit nuk është burim fakti. Çdo gjykim që vjen nga profili, jo nga fragmentet, shënoje me "(interpretim)".
- Nëse një pjesë e kërkuar nuk gjendet te fragmentet, shkruaj saktësisht: "Nuk e gjej këtë informacion në dokumentet e disponueshme." për atë pjesë dhe vazhdo me pjesët e tjera.
- Mos përmend kurrë etiketa të brendshme si "Fragmenti 1", "chunk" ose "kontekst", dhe mos shkruaj numra fragmentesh. Thuaj "dokumentet" dhe cito vetëm emrin e programit dhe faqen.
- Mos shkruaj seksionin "Burimet": shtohet automatikisht nga sistemi.
- Mos jep pikë numerike ose përqindje. Për përshtatshmërinë përdor vetëm "e lartë", "e mesme", "e ulët" ose "pa të dhëna", me arsyetim nga fragmentet.
- Përgjigju në shqip. Përdor titujt Markdown "##" saktësisht siç janë dhënë."""

STRUCTURE = """## Rekomandimi
[programi ose rruga e rekomanduar]

## Pse i përshtatet profilit tënd
[2 deri 4 pika; çdo pikë lidh një element të profilit me një fakt nga fragmentet, me burimin (Emri i programit, f. N)]

## Lëndët përkatëse
[lista e lëndëve nga fragmentet, me vit/semestër dhe ECTS kur shfaqen]

## Boshllëqet e aftësive
[shih udhëzimin për boshllëqet]

## Rruga e sugjeruar
[hapa të renditur]

## Drejtimi i karrierës
[vetëm mundësitë që përmenden te fragmentet për atë program]"""

GAP_RULES = """Për "Boshllëqet e aftësive": krahaso aftësitë aktuale të studentit me ato që mbështeten nga fragmentet (lëndë, objektiva, profile). Për çdo aftësi përdor formatin:
**Aftësia** — Aktuale: [nga profili ose "nuk është dhënë"] | E kërkuar: [vetëm nëse fragmentet e tregojnë nivelin, përndryshe "nuk specifikohet në dokumente"] | Mbulohet nga: [lënda dhe burimi (Emri i programit, f. N), ose "asnjë lëndë e gjetur"] | Boshllëku: [asnjë / i vogël / i mesëm / i madh, vetëm kur niveli i kërkuar dihet; përndryshe "nuk përcaktohet"]
Nëse dokumentet nuk përmbajnë rezultate të pritura të të nxënit ose nivele të kërkuara, thuaje hapur në fund të seksionit: "Dokumentet aktuale nuk përmbajnë nivele të kërkuara të aftësive, prandaj niveli i boshllëkut nuk mund të përcaktohet. Duhen shtuar rezultatet e të nxënit të lëndëve."."""

INTENT_INSTRUCTIONS = {
    "personalized_recommendation": f"Detyra: rekomando programin dhe lëndët më të përshtatshme.\n\nStruktura:\n\n{STRUCTURE}\n\n{GAP_RULES}",
    "academic_pathway": f"Detyra: ndërto rrugën akademike të personalizuar nga profili deri te objektivi.\n\nStruktura:\n\n{STRUCTURE}\n\n{GAP_RULES}",
    "skill_gap_analysis": f"Detyra: analizo boshllëqet e aftësive për objektivin e studentit. Seksioni 'Boshllëqet e aftësive' është më i rëndësishmi dhe duhet të jetë i detajuar.\n\nStruktura:\n\n{STRUCTURE}\n\n{GAP_RULES}",
    "program_comparison": f"""Detyra: krahaso programet e pyetura sipas profilit të studentit.

Struktura:

## Krahasimi
[tabelë Markdown: rreshta = kriteret (përshtatja me profilin, përmbajtja e programimit, fokusi te të dhënat/fusha e objektivit, përputhja me karrierën); kolona = programet. Çdo qelizë: "e lartë", "e mesme", "e ulët" ose "pa të dhëna", me burimin (Emri i programit, f. N)]

## Rekomandimi
[programi më i përshtatshëm]

## Pse
[arsyetim me pika, me burimin (Emri i programit, f. N)]

## Lëndët përkatëse
[lëndët kryesore të secilit program nga fragmentet]

## Drejtimi i karrierës
[vetëm nga fragmentet]""",
    "what_if": f"""Detyra: studenti po eksploron një rrugë alternative. Përdor profilin ekzistues dhe ndërto rrugën për objektivin e ri.

Struktura:

## Alternativa e re
[objektivi i ri dhe programi përkatës nga fragmentet]

## Çfarë ndryshon
[krahasim i shkurtër me rrugën e mëparshme ose me objektivin e mëparshëm nga profili]

## Lëndët që bëhen më të rëndësishme
[nga fragmentet, me vit/semestër dhe ECTS]

## Aftësitë që bëhen më të rëndësishme
[vetëm ato që mbështeten nga fragmentet; krahaso me aftësitë e profilit]

## Rruga e sugjeruar
[hapa të renditur]

## Drejtimi i karrierës
[vetëm nga fragmentet]""",
}


class Advisor:
    def __init__(self, rag: RagAssistant) -> None:
        self.rag = rag
        self.current = CurrentStudentAssistant(rag)  # rruga e studentit aktual (student_assistant.py)

    # ---------- pika hyrëse ----------
    def respond(
        self,
        question: str,
        profile: dict | None = None,
        history: list[dict] | None = None,
        followups_asked: int = 0,
        role: str = "prospective",
        academic: dict | None = None,
        curriculum: dict | None = None,
        study_course: str | None = None,
        forced_task: str | None = None,
        user_material: str = "",
    ) -> dict:
        """Përgjigjet një mesazhi. Kthen të njëjtat çelësa si RagAssistant.answer(),
        plus: intent, profile (i përditësuar), profile_changes, follow_up (bool)."""
        profile = profile if profile is not None else new_profile()

        if is_small_talk(question):
            return self._wrap(self.rag.answer(question), "small_talk", profile, [], False)

        # Veprimet e shpejta të Study Mode (quiz, flashcards...) e dinë detyrën: pa thirrje klasifikimi.
        if forced_task:
            analysis = intent_module.neutral_analysis(question)
        else:
            analysis = intent_module.analyze_turn(question, profile_to_text(profile), history, role)
        profile, changes = merge_profile(profile, analysis["profile_updates"])
        intent = analysis["intent"]

        # Studenti aktual: moduli i veçantë. Kthen None vetëm kur studenti kërkon qartë programe të tjera.
        if role == "current":
            reply = self.current.handle(
                question, analysis, academic, curriculum or {}, history,
                study_course=study_course, forced_task=forced_task, user_material=user_material, profile=profile,
            )
            if reply is not None:
                return self._wrap(reply, intent, profile, changes, False)

        if intent not in intent_module.PERSONALIZED_INTENTS:
            return self._wrap(self.rag.answer(analysis["standalone_question"]), intent, profile, changes, False)

        missing = missing_for(intent, profile)
        if missing and followups_asked < MAX_FOLLOW_UPS:
            reply = {
                "question": question,
                "answer": follow_up_question(missing),
                "refused": False,
                "sources": [],
                "chunks": [],
            }
            return self._wrap(reply, intent, profile, changes, True)

        return self._wrap(self._personalized(question, intent, profile, history, analysis), intent, profile, changes, False)

    # ---------- personalizimi ----------
    def _personalized(self, question: str, intent: str, profile: dict, history: list[dict] | None, analysis: dict) -> dict:
        queries = [f"{question}. {profile_to_query(profile)}".strip(". "), *analysis["search_queries"]]
        chunks = self.rag.retrieve_multi(queries, k=config.TOP_K, limit=12)

        recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in (history or [])[-4:])
        user_prompt = (
            f"Fragmentet e dokumenteve:\n\n{self.rag._build_context(chunks)}\n\n"
            f"Profili i studentit:\n{profile_to_text(profile) or '(bosh)'}\n\n"
            + (f"Biseda e fundit:\n{recent}\n\n" if recent else "")
            + f"{INTENT_INSTRUCTIONS[intent]}\n\n{COMMON_RULES}\n\n"
            f"Pyetja e studentit: {question}"
        )
        text = sanitize_labels(llm.chat_completion(SYSTEM_PROMPT, user_prompt), chunks)
        refused = text.strip().startswith(REFUSAL_PREFIX)
        sources = [] if refused else attribute_sources(text, chunks)
        if sources:
            text += "\n\n" + format_sources(sources)
        return {"question": question, "answer": text, "refused": refused, "sources": sources, "chunks": chunks}

    @staticmethod
    def _wrap(result: dict, intent: str, profile: dict, changes: list[str], follow_up: bool) -> dict:
        return {**result, "intent": intent, "profile": profile, "profile_changes": changes, "follow_up": follow_up}
