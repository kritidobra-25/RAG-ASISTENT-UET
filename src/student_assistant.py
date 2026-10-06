"""
Asistenti i studentit aktual (modul i veçantë, bashkëjeton me studentin potencial te advisor.py).

Rrjedha:  profili akademik (program, specializim, vit, semestër)
          → plani mësimor zyrtar (curriculum.json) → lëndët e semestrit
          → Study Mode për një lëndë (ose pyetje akademike të përgjithshme)
          → RAG: materialet e lëndës së pari, pastaj dokumentet zyrtare të programit → LLM.

Studenti aktual nuk merr rekomandime për programe të tjera, përveç kur i kërkon vetë.
Burimi i së vërtetës janë faktet e planit dhe segmentet e gjetura; përmbajtja e
përgjithshme (kur mungojnë materialet) shënohet qartë si e tillë.
"""

import config
import curriculum as curriculum_module
import llm
from rag import REFUSAL_PREFIX, RagAssistant, sanitize_labels
from sources import attribute_sources, format_sources
from student_profile import academic_complete, academic_to_text, profile_to_text

# Detyrat që krijojnë përmbajtje mësimore dhe kërkojnë material ose temë.
CONTENT_TASKS = {"quiz", "exercises", "flashcards", "mock_exam", "review_plan", "summary"}
TASK_LABELS = {
    "quiz": "quiz-in", "exercises": "ushtrimet", "flashcards": "flashcards", "mock_exam": "provimin e provës",
    "review_plan": "planin e përsëritjes", "summary": "përmbledhjen",
}

STUDY_SYSTEM_PROMPT = f"""Je asistenti akademik dhe i studimit i Universitetit Europian të Tiranës (UET) për studentët aktualë.

Burimi i së vërtetës janë faktet zyrtare të lëndës, materialet e lëndës dhe dokumentet zyrtare që jepen në mesazh.
Rregullat:
1. Mos shpik informacion për lëndët: ECTS, tema të kurrikulës, përmbajtje, kërkesa për provime ose rregulla të universitetit. Nëse diçka nuk gjendet te burimet, thuaje hapur: "Nuk e gjej këtë informacion në burimet e disponueshme."
2. Faktet për UET-në (lëndë, ECTS, vit, semestër, programi) merren vetëm nga faktet zyrtare dhe dokumentet. Cito burimin në kllapa, p.sh. (Titulli i burimit, f. 3). Nëse faqja mungon, shkruaj "Burimi: titulli".
3. Quiz-et, ushtrimet, flashcards, përmbledhjet dhe provimet e provës bazohen kryesisht te materialet e lëndës dhe materiali i ngjitur nga studenti.
4. Kur nuk ka materiale të lëndës, mund të shpjegosh një koncept të përgjithshëm që studenti e emërton, por duhet ta shënosh qartë në fillim: "Nuk kam materiale të lëndës në bazën e njohurive; kjo është përmbajtje e përgjithshme për temën, jo nga materialet e lëndës." Mos pretendo kurrë që është pjesë e lëndës ose e provimit.
5. Nëse faktet e një lënde kanë "SHËNIM KORRIGJIMI", vlera e planit të konfirmuar ka përparësi mbi tekstin e PDF-së. Përmende shënimin vetëm kur studenti pyet për vitin ose semestrin e asaj lënde.
6. Mos rekomando programe të tjera të UET. Studenti është tashmë në programin e tij.
7. Mos përmend kurrë etiketa të brendshme si "Fragmenti 1", "chunk", "kontekst" ose pikë ngjashmërie. Cito vetëm titullin e burimit dhe faqen.
8. Përgjigju në shqip, qartë dhe në nivelin e studentit. Përdor Markdown ("##" për tituj, listë me pika)."""

TASK_INSTRUCTIONS = {
    "explain": "Detyra: shpjego konceptin hap pas hapi, thjesht, me një shembull. Nëse studenti kërkon 'më thjesht', thjeshto edhe më shumë.",
    "question": "Detyra: përgjigju pyetjes së studentit.",
    "overview": "Detyra: trego çfarë dihet për lëndën: ECTS, viti, semestri, kategoria dhe specializimi nga faktet zyrtare. Përshkrimin, temat dhe objektivat e të nxënit trego VETËM nëse gjenden te burimet; përndryshe shkruaj: 'Përshkrimi, temat dhe objektivat e të nxënit nuk janë të disponueshme në burimet e ngarkuara.'",
    "quiz": "Detyra: krijo një quiz me 8 pyetje (përzierje: zgjedhje e shumëfishtë, E vërtetë/E gabuar, përgjigje e shkurtër). Pyetjet vijnë nga materiali. Jep përgjigjet në fund, nën titullin '## Përgjigjet', me arsyetim të shkurtër.",
    "exercises": "Detyra: krijo 5 ushtrime praktike me vështirësi në rritje, nga materiali. Jep zgjidhjet në fund, nën titullin '## Zgjidhjet'.",
    "flashcards": "Detyra: krijo 10 deri 15 flashcards nga materiali, secila në formatin:\n**Pyetje:** ...\n**Përgjigje:** ...",
    "mock_exam": "Detyra: krijo një provim prove nga materiali me tri pjesë (pyetje teorike të shkurtra, zgjedhje e shumëfishtë, një problem ose rast praktik). Në fillim shkruaj: 'Ky është një model prove për ushtrim. Formati dhe kërkesat e provimit zyrtar nuk gjenden te dokumentet.' Mos jep pikë zyrtare. Jep përgjigjet në fund.",
    "review_plan": "Detyra: krijo një plan përsëritjeje ditë pas dite (nëse studenti tha numrin e ditëve, përdore; përndryshe 7 ditë), vetëm me temat që gjenden te materiali. Për çdo ditë: temat dhe aktiviteti (lexim, ushtrime, vetëtestim).",
    "summary": "Detyra: përgatit një përmbledhje të strukturuar të materialit (tituj dhe pika kryesore). Mos shto tema që nuk janë te materiali.",
}


class CurrentStudentAssistant:
    def __init__(self, rag: RagAssistant) -> None:
        self.rag = rag

    # ---------- pika hyrëse ----------
    def handle(
        self,
        question: str,
        analysis: dict,
        academic: dict | None,
        curriculum: dict,
        history: list[dict] | None = None,
        study_course: str | None = None,
        forced_task: str | None = None,
        user_material: str = "",
        profile: dict | None = None,
    ) -> dict | None:
        """Përgjigjet studentit aktual. Kthen None kur studenti kërkon qartë programe të tjera
        (atëherë Advisor përdor rrjedhën e rekomandimit)."""
        base = {"question": question, "refused": False, "sources": [], "chunks": []}
        intent, task = analysis["intent"], forced_task or analysis.get("study_task")

        if intent in {"personalized_recommendation", "academic_pathway", "program_comparison", "what_if"} and analysis.get("asks_other_programs") and not study_course:
            return None

        if not academic_complete(academic):
            return {**base, "answer": "Për të të treguar lëndët e semestrit, plotëso te profili im akademik programin, vitin dhe semestrin."}

        program, year, semester = academic["program"], int(academic["year"]), int(academic["semester"])
        specialization = academic.get("specialization", "")
        courses = curriculum_module.courses_for(curriculum, program, year, semester, specialization)
        if not courses:
            return {
                **base,
                "answer": f"Nuk gjej lëndë për {program}, viti {year}, semestri {semester} te plani mësimor i nxjerrë nga dokumentet. "
                "Kontrollo vitin dhe semestrin te profili, ose pyet administratën e UET-së.",
            }

        # 1) Study Mode: lënda është zgjedhur te ndërfaqja.
        course_name = None
        if study_course:
            course_name = curriculum_module.best_match(sorted({c["name"] for c in courses}), study_course) or study_course
        # 2) Lënda e përmendur te mesazhi.
        elif analysis.get("course_mentioned"):
            course_name = curriculum_module.best_match(sorted({c["name"] for c in courses}), analysis["course_mentioned"])

        if course_name:
            return self._study(question, analysis, course_name, task or ("overview" if intent == "course_info" else "question"),
                               academic, courses, curriculum, history, user_material, profile)

        # 3) Detyrë mësimore pa lëndë të caktuar.
        if task in CONTENT_TASKS or task == "explain":
            names = "\n".join(f"{i}. {n}" for i, n in enumerate(sorted({c['name'] for c in courses}), start=1))
            return {**base, "answer": f"Për ta bërë këtë, zgjidh një nga lëndët e semestrit te Study Mode ose shkruaj emrin e saj:\n\n{names}"}

        # 4) Lëndët e semestrit dhe pyetjet e tjera akademike.
        if intent in {"study_help", "course_info"}:
            return self._semester_courses(question, analysis, academic, courses, curriculum_module.profiles_for(curriculum, program))

        # 5) Informacion akademik: RAG standard i kufizuar te programi i studentit.
        where = None if analysis.get("asks_other_programs") else {"program": program}
        extra = f"Studenti është aktual te {program}. Mos rekomando programe të tjera të UET-së."
        return self.rag.answer(analysis["standalone_question"], where=where, extra=extra)

    # ---------- lëndët e semestrit ----------
    def _semester_courses(self, question: str, analysis: dict, academic: dict, courses: list[dict], profiles: list[str]) -> dict:
        program, year, semester = academic["program"], academic["year"], academic["semester"]
        notes = []
        if profiles and not academic.get("specialization"):
            notes.append("Programi ka specializime dhe lëndët e tyre me zgjedhje nuk u përfshinë, sepse specializimi nuk është plotësuar te profili.")
        user_prompt = (
            f"Plani mësimor i semestrit aktual ({program}, viti {year}, semestri {semester}):\n"
            f"{curriculum_module.format_courses(courses)}\n\n"
            + (f"Shënim: {' '.join(notes)}\n\n" if notes else "")
            + f"Profili akademik i studentit:\n{academic_to_text(academic)}\n\n"
            "Detyra: përgjigju pyetjes vetëm me lëndët e planit më sipër. Kur liston lëndët, numëroji dhe trego ECTS. "
            "Lëndët e shënuara 'me zgjedhje ose sipas profilit' nuk janë të gjitha të detyrueshme: thuaje këtë.\n\n"
            f"Pyetja e studentit: {analysis['standalone_question']}"
        )
        text = sanitize_labels(llm.chat_completion(STUDY_SYSTEM_PROMPT, user_prompt), [])
        pages = sorted({c["page"] for c in courses if c.get("page")})
        sources = [{"program": program, "source": courses[0]["source"], "pages": pages, "doc_type": "program"}]
        text += "\n\n" + format_sources(sources)
        return {"question": question, "answer": text, "refused": False, "sources": sources, "chunks": []}

    # ---------- Study Mode ----------
    def _study(self, question, analysis, course_name, task, academic, courses, curriculum, history, user_material, profile) -> dict:
        program = academic["program"]
        rows = [c for c in courses if c["name"] == course_name]
        facts = "\n".join(curriculum_module.course_facts(c) for c in rows) or f"{course_name}: nuk gjendet te plani i semestrit."
        standalone = analysis["standalone_question"]
        topic = analysis.get("topic")
        user_material = (user_material or "").strip()[:6000]

        # RAG: së pari materialet e lëndës, pastaj dokumentet zyrtare të programit.
        material_where = {"$and": [{"doc_type": "material"}, {"course": course_name}]}
        materials = self.rag.retrieve_multi([f"{standalone}. {course_name}"], k=config.TOP_K, limit=6, where=material_where)
        official = self.rag.retrieve_multi([f"{course_name}. {standalone}"], k=3, limit=3, where={"program": program})
        chunks = materials + [c for c in official if c["id"] not in {m["id"] for m in materials}]
        has_material = bool(materials) or bool(user_material)

        base = {"question": question, "refused": False, "sources": [], "chunks": chunks}
        if task in CONTENT_TASKS and not has_material and not topic:
            return {
                **base,
                "answer": f"Nuk kam materiale për lëndën {course_name} te baza e njohurive, ndaj nuk mund ta bazoj {TASK_LABELS[task]} te lënda. "
                "Mund ta përgatis nëse më shkruan një temë (p.sh. \"REST API\") dhe do të jetë përmbajtje e përgjithshme, e shënuar si e tillë. "
                "Ose ngjit materialin tënd te Study Mode. Materialet e lëndës mund të shtohen te dosja data/lendet.",
            }

        recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in (history or [])[-6:])
        parts = [
            f"Lënda e zgjedhur: {course_name}",
            f"Faktet zyrtare të lëndës (plani mësimor):\n{facts}",
            f"Materialet e lëndës (burimi kryesor):\n{self.rag._build_context(materials)}" if materials else "Materialet e lëndës: asnjë material i gjetur te baza e njohurive për këtë lëndë.",
            f"Dokumente zyrtare të programit:\n{self.rag._build_context([c for c in chunks if c not in materials])}" if len(chunks) > len(materials) else "",
            f"Materiali i ngjitur nga studenti (përdoret si burim, cito si 'Materiali i studentit'):\n{user_material}" if user_material else "",
            f"Profili akademik:\n{academic_to_text(academic)}",
            f"Profili i studentit (përshtat nivelin e shpjegimit):\n{profile_to_text(profile)}" if profile_to_text(profile) else "",
            f"Biseda e fundit:\n{recent}" if recent else "",
            TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS["question"]),
            (f"Tema e kërkuar nga studenti: {topic}" if topic else ""),
            "" if has_material else f'Nuk ka materiale të lëndës. Përmbajtja e përgjithshme lejohet vetëm për temën e studentit dhe duhet të fillojë me: "Nuk kam materiale të lëndës {course_name} në bazën e njohurive; kjo është përmbajtje e përgjithshme për temën, jo nga materialet e lëndës."',
            f"Pyetja e studentit (e plotësuar me kontekstin e bisedës): {standalone}",
        ]
        text = sanitize_labels(llm.chat_completion(STUDY_SYSTEM_PROMPT, "\n\n".join(p for p in parts if p)), chunks)
        refused = text.strip().startswith(REFUSAL_PREFIX)

        sources: list[dict] = []
        if not refused:
            sources = attribute_sources(text, chunks) if chunks else []
            sources = [s for s in sources if s["doc_type"] == "material" or s["program"] in text]
            plan_pages = sorted({c["page"] for c in rows if c.get("page")})
            if rows and not any(s["program"] == program for s in sources):
                sources.append({"program": program, "source": rows[0]["source"], "pages": plan_pages, "doc_type": "program"})
            if user_material:
                sources.append({"program": "Materiali i studentit", "source": "", "pages": [], "doc_type": "user"})
        if sources:
            text += "\n\n" + format_sources(sources)
        return {"question": question, "answer": text, "refused": refused, "sources": sources, "chunks": chunks}
