"""
Hapi 5: ndërfaqja bisedore (Streamlit).

Nise nga dosja kryesore e projektit:
    streamlit run src/app.py
"""

# Importet standarde të Python: html (mbrojtje nga HTML i padëshiruar), shutil, sys dhe Path
import html
import shutil
import sys
from pathlib import Path

# Shton dosjen src/ te rruga e moduleve, që importet si 'import rag' të funksionojnë kudo
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shumë serverë (p.sh. Streamlit Community Cloud) kanë SQLite të vjetër që ChromaDB
# nuk e pranon. Nëse pysqlite3 është i instaluar, përdoret ai.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# Streamlit: biblioteka e ndërfaqes web
import streamlit as st  # noqa: E402

# Modulet e projektit: llogaritë, plani mësimor, LLM, profili, këshilltari, indeksi dhe RAG
import accounts  # noqa: E402
import config  # noqa: E402
import curriculum  # noqa: E402
import llm  # noqa: E402
from student_profile import academic_to_text, is_empty, new_academic, new_profile, profile_to_text  # noqa: E402
from advisor import Advisor  # noqa: E402
from build_index import build_index  # noqa: E402
from rag import KnowledgeBaseMissingError, RagAssistant  # noqa: E402

# Zgjedhjet për 'Programi i dëshiruar' te profili i studentit potencial
PROGRAM_OPTIONS = ["Master Profesional", "Master Shkencor"]
# Pyetja e gatshme që dërgohet kur shtypet 'Gjenero rrugën akademike'
PATH_QUESTION = "Më sugjero rrugën akademike më të përshtatshme sipas profilit tim."

# Rruga e logos së UET (assets/uet_logo.png)
LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "uet_logo.png"

# Cilësimet e faqes: titulli i skedës, ikona dhe gjerësia
st.set_page_config(page_title="UETassist", page_icon=str(LOGO_PATH), layout="centered")

# Ngarkon stilet nga style.css dhe i injekton te faqja
st.markdown(
    f"<style>{(Path(__file__).resolve().parent / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


# Krijon tabelën e përdoruesve te SQLite nëse nuk ekziston
accounts.init_db()


# Ndihmës për selectbox: gjen pozicionin e vlerës ekzistuese te lista
def pick(options: list, value):
    """Indeksi i vlerës te lista, ose None (që selectbox të mbetet pa zgjedhje)."""
    return options.index(value) if value in options else None


# Faqja e hyrjes dhe regjistrimit (shfaqet para çdo gjëje tjetër)
def login_screen() -> None:
    logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
    logo_col.image(str(LOGO_PATH), width=72)
    title_col.title("UETassist")
    st.caption(
        "Hyr ose krijo një llogari që profili yt të ruhet. Përdor një pseudonim, jo emrin real. "
        "Biseda nuk ruhet, vetëm profili."
    )
    # Dy skeda: hyrje me llogari ekzistuese dhe regjistrim i ri
    login_tab, register_tab = st.tabs(["Hyr", "Regjistrohu"])
    with login_tab, st.form("login_form"):
        username = st.text_input("Emri i përdoruesit")
        password = st.text_input("Fjalëkalimi", type="password")
        if st.form_submit_button("Hyr", width="stretch"):
            # Verifikon fjalëkalimin kundrejt hash-it të ruajtur; kthen None nëse është gabim
            user = accounts.authenticate(username, password)
            if user:
                start_session(user)
                st.rerun()
            st.error("Emri i përdoruesit ose fjalëkalimi nuk është i saktë.")
    with register_tab, st.form("register_form"):
        new_username = st.text_input("Emri i përdoruesit (pseudonim)")
        new_password = st.text_input("Fjalëkalimi (të paktën 8 karaktere)", type="password")
        role_label = st.radio("Je:", list(accounts.ROLES.values()), index=None)
        if st.form_submit_button("Krijo llogarinë", width="stretch"):
            try:
                # Kthen çelësin e rolit ('current' ose 'prospective') nga etiketa e zgjedhur
                role = next((key for key, label in accounts.ROLES.items() if label == role_label), "")
                # Krijon llogarinë dhe e hap menjëherë sesionin
                start_session(accounts.create_user(new_username, new_password, role))
                st.rerun()
            except accounts.AccountError as error:
                st.error(str(error))


# Ngarkon të dhënat e përdoruesit te gjendja e sesionit (st.session_state)
def start_session(user: dict) -> None:
    st.session_state.update(
        user_id=user["id"], username=user["username"], role=user["role"],
        profile=user["profile"], academic=user["academic"], messages=[], followups=0,
    )


# Ruan profilin, të dhënat akademike dhe rolin te baza SQLite
def save_user() -> None:
    accounts.save_user_data(
        st.session_state["user_id"], st.session_state["profile"], st.session_state["academic"], st.session_state["role"]
    )


# Plani mësimor ngarkohet një herë dhe mbahet në kujtesë (cache)
@st.cache_data
def load_curriculum() -> dict:
    data = curriculum.load()
    if not data:  # data/curriculum.json mungon: ndërtohet nga PDF-të (pa kosto API)
        data = curriculum.build()
        try:
            curriculum.save(data)
        except OSError:
            pass
    return data


# Pa hyrje nuk ngarkohet baza e njohurive: vizitorët e panjohur nuk shkaktojnë kosto OpenAI.
if "user_id" not in st.session_state:
    login_screen()
    st.stop()

# Asistenti RAG ngarkohet një herë për gjithë serverin, jo për çdo ndërveprim
@st.cache_resource(show_spinner="Duke ngarkuar bazën e njohurive...")
def load_assistant() -> RagAssistant:
    try:
        return RagAssistant()
    except KnowledgeBaseMissingError:
        # Serveri i ri nuk e ka bazën (chroma_db/ nuk ruhet te GitHub): ndërtohet
        # vetë nga dokumentet te data/ herën e parë. Kërkon OPENAI_API_KEY.
        with st.spinner("Duke ndërtuar bazën e njohurive nga dokumentet (vetëm herën e parë)..."):
            shutil.rmtree(config.DB_DIR, ignore_errors=True)  # nis nga e para, edhe nëse baza ishte e papajtueshme
            build_index()
        return RagAssistant()


# Ndërton asistentin RAG dhe këshilltarin; trajton gabimet e çelësit API dhe të bazës
try:
    assistant = load_assistant()
    advisor = Advisor(assistant)
except llm.MissingApiKeyError as error:
    st.error(str(error))
    st.stop()
except KnowledgeBaseMissingError as error:
    st.error(
        f"{error}\n\nHap terminalin në dosjen kryesore të projektit dhe ekzekuto "
        "`python src/build_index.py`, pastaj rifresko këtë faqe."
    )
    if error.__cause__:
        with st.expander("Detaje teknike"):
            st.code(f"{type(error.__cause__).__name__}: {error.__cause__}")
    st.stop()

# ---------- Shiriti anësor ----------
curriculum_data = load_curriculum()
role = st.session_state["role"]
profile = st.session_state["profile"]
academic = st.session_state["academic"]

with st.sidebar:
    st.image(str(LOGO_PATH), width="stretch")
    st.markdown(f"**{st.session_state['username']}**")
    role_labels = list(accounts.ROLES.values())
    # Ndërrimi i rolit (student aktual ose potencial) ruhet te llogaria
    chosen = st.selectbox("Roli", role_labels, index=role_labels.index(accounts.ROLES[role]))
    if chosen != accounts.ROLES[role]:
        st.session_state["role"] = next(key for key, label in accounts.ROLES.items() if label == chosen)
        save_user()
        st.rerun()

    # Profili akademik: vetëm për studentin aktual
    if role == "current":
        with st.expander("Profili im akademik", expanded=not academic.get("program")):
            st.caption(
                "Programi, viti dhe semestri përdoren për të gjetur lëndët e semestrit nga plani mësimor. "
                "Fakulteti, departamenti dhe specializimi nuk gjenden te dokumentet, prandaj i plotëson ti."
            )
            summary = academic_to_text(academic)
            st.markdown(summary if summary else "*Profili akademik është ende bosh.*")
            # Fakulteti dhe departamenti: tekst i lirë, sepse nuk gjenden te dokumentet
            faculty = st.text_input("Fakulteti", academic["faculty"], key="acad_faculty")
            department = st.text_input("Departamenti", academic["department"], key="acad_department")
            programs = curriculum.programs(curriculum_data)
            program = st.selectbox("Programi", programs, index=pick(programs, academic["program"]), placeholder="Zgjidh programin", key="acad_program")
            # Specializimet merren nga plani mësimor i programit të zgjedhur
            profiles = curriculum.profiles_for(curriculum_data, program) if program else []
            # Me specializime nga plani: listë zgjedhjeje; përndryshe: tekst i lirë
            if profiles:
                specialization = st.selectbox(
                    "Specializimi", profiles, index=pick(profiles, curriculum.match_profile(curriculum_data, program, academic["specialization"])),
                    placeholder="Zgjidh specializimin", key=f"acad_spec_{program}",
                )
            else:
                specialization = st.text_input("Specializimi (nëse ka)", academic["specialization"], key=f"acad_spec_text_{program}")
            year = st.selectbox("Viti", [1, 2], index=pick([1, 2], academic["year"]), placeholder="Zgjidh vitin", key="acad_year")
            semester = st.selectbox("Semestri (brenda vitit)", [1, 2], index=pick([1, 2], academic["semester"]), placeholder="Zgjidh semestrin", key="acad_semester")
            # Ruan zgjedhjet dhe del nga Study Mode, sepse lëndët e semestrit mund të kenë ndryshuar
            if st.button("Ruaj profilin akademik", width="stretch", key="acad_save"):
                academic.update(
                    faculty=faculty.strip(), department=department.strip(), program=program or "",
                    specialization=(specialization or "").strip(), year=year or 0, semester=semester or 0,
                )
                st.session_state["study_course"] = None
                st.session_state.pop("study_select", None)
                save_user()
                st.rerun()

    # Profili i interesave dhe aftësive: plotësohet nga biseda ose manualisht
    with st.expander("Profili im" if role == "prospective" else "Interesat dhe aftësitë", expanded=role == "prospective" and not is_empty(profile)):
        st.caption(
            "Ky profil plotësohet vetë nga biseda dhe përditësohet sipas informacionit që ndan. "
            "Mund ta ndryshosh edhe manualisht."
        )
        summary = profile_to_text(profile)
        st.markdown(summary if summary else "*Profili juaj është ende bosh.*")

        # Formular manual: ndryshimet ruhen vetëm kur shtypet 'Ruaj profilin'
        with st.form("profile_form"):
            st.caption("Opsionale: Mund ta plotësoni ose ndryshoni profilin manualisht.")
            background = st.text_input("Backgroundi akademik", profile["academic_background"], placeholder="p.sh. Bachelor në Administrim Biznesi")
            goal = st.text_input("Karriera e dëshiruar", profile["career_goal"], placeholder="p.sh. Data Engineer")
            interests = st.text_input("Fusha e interesit", ", ".join(profile["interests"]), placeholder="psh: IT, Finance, Biznes")
            skills = st.text_input("Aftësi teknike", ", ".join(profile["technical_skills"]), placeholder="p.sh. SQL, Python")
            desired = st.selectbox(
                "Programi i dëshiruar",
                PROGRAM_OPTIONS,
                index=pick(PROGRAM_OPTIONS, profile["desired_program"]),
                placeholder="Zgjidh programin",
            )
            if st.form_submit_button("Ruaj profilin", width="stretch"):
                profile["academic_background"] = background.strip()
                profile["career_goal"] = goal.strip()
                profile["interests"] = [x.strip() for x in interests.split(",") if x.strip()]
                profile["technical_skills"] = [x.strip() for x in skills.split(",") if x.strip()]
                profile["desired_program"] = desired or ""
                save_user()
                st.rerun()

        # Rruga akademike e gatshme: vetëm për studentin potencial
        if role == "prospective" and st.button("Gjenero rrugën akademike", width="stretch", key="gen_path"):
            st.session_state["pending_question"] = PATH_QUESTION
        # Fshin profilin (dhe të dhënat akademike të studentit aktual)
        if st.button("Rivendos profilin", width="stretch", key="reset_profile"):
            st.session_state["profile"] = new_profile()
            if role == "current":
                st.session_state["academic"] = new_academic()
            st.session_state["followups"] = 0
            save_user()
            st.rerun()

    # Fshin vetëm bisedën; profili mbetet
    if st.button("Pastro bisedën", width="stretch", key="clear_sidebar"):
        st.session_state["messages"] = []
        st.session_state["followups"] = 0
        st.rerun()
    # Dalja: pastron sesionin dhe kthen faqen e hyrjes
    if st.button("Dil", width="stretch", key="logout"):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()

# ---------- Biseda ----------
logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
logo_col.image(str(LOGO_PATH), width=72)
title_col.title("UETassist")
st.caption(
    "Përshëndetje! Jam asistenti virtual i UET. Jam këtu për t’ju ndihmuar me çdo "
    "pyetje rreth studimeve. Çfarë dëshironi të dini?"
)

# Historia e bisedës ruhet vetëm në sesion (nuk shkruhet në disk)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------- Konteksti akademik dhe Study Mode (vetëm studenti aktual) ----------
STUDY_ACTIONS = {
    "quiz": ("Quiz", "Më bëj një quiz për lëndën {course}."),
    "exercises": ("Ushtrime", "Më jep ushtrime praktike për lëndën {course}."),
    "flashcards": ("Flashcards", "Më krijo flashcards për lëndën {course}."),
    "mock_exam": ("Provim prove", "Më bëj një test prove për lëndën {course}."),
    "review_plan": ("Plan përsëritjeje", "Më krijo një plan përsëritjeje për lëndën {course}."),
    "summary": ("Përmbledhje", "Më bëj një përmbledhje të materialit të lëndës {course}."),
}


# Del nga Study Mode (thirret nga butoni 'Dil nga Study Mode')
def leave_study_mode() -> None:
    st.session_state["study_course"] = None
    st.session_state["study_select"] = None


def current_student_panel() -> list[dict]:
    """Karta e kontekstit, lista e lëndëve dhe Study Mode. Kthen lëndët e semestrit."""
    # Pa program të zgjedhur nuk mund të përcaktohen lëndët
    if not academic.get("program"):
        st.info("Modaliteti: Student aktual. Plotëso programin, vitin dhe semestrin te profili im akademik, që lëndët të përcaktohen automatikisht.")
        return []
    # Rreshtat e kartës së kontekstit: programi, specializimi, viti dhe semestri
    lines = [academic["program"]]
    if academic.get("specialization"):
        lines.append(academic["specialization"])
    if academic.get("year") and academic.get("semester"):
        lines.append(f"Viti {academic['year']} · Semestri {academic['semester']}")
    # Karta e kontekstit; html.escape mbron nga futja e HTML-së nga përdoruesi
    st.markdown(
        '<div class="context-card"><span class="context-badge">Student aktual</span>'
        + "".join(f"<div>{html.escape(str(line))}</div>" for line in lines)
        + "</div>",
        unsafe_allow_html=True,
    )
    if not (academic.get("year") and academic.get("semester")):
        return []

    # Lëndët e semestrit sipas programit, specializimit, vitit dhe semestrit
    courses = curriculum.courses_for(curriculum_data, academic["program"], int(academic["year"]), int(academic["semester"]), academic.get("specialization", ""))
    names = sorted({c["name"] for c in courses})
    if not names:
        st.caption("Nuk u gjetën lëndë për këtë vit dhe semestër te plani mësimor.")
        return courses

    # Lista e lëndëve me ECTS
    with st.expander(f"Lëndët e semestrit ({len(names)})", expanded=not st.session_state.get("study_course")):
        for course in sorted(courses, key=lambda c: c["name"]):
            st.markdown(f"- {course['name']}, {course['ects']} ECTS")
        if curriculum.profiles_for(curriculum_data, academic["program"]) and not curriculum.match_profile(curriculum_data, academic["program"], academic.get("specialization", "")):
            st.caption("Lëndët e specializimit nuk shfaqen, sepse specializimi nuk është zgjedhur te profili.")
    # Zgjedhja e lëndës për Study Mode
    st.selectbox("Study Mode: zgjidh një lëndë", names, index=pick(names, st.session_state.get("study_course")), placeholder="Zgjidh lëndën", key="study_select")
    st.session_state["study_course"] = st.session_state.get("study_select")

    # Kur është zgjedhur një lëndë: banner, butonat e veprimeve dhe materiali i ngjitur
    course = st.session_state["study_course"]
    if course:
        ects = next((c["ects"] for c in courses if c["name"] == course), "")
        st.markdown(f'<div class="study-banner">Study Mode: <b>{html.escape(course)}</b> · {ects} ECTS</div>', unsafe_allow_html=True)
        # Një buton për secilin veprim; shtypja vendos pyetjen dhe detyrën në pritje
        columns = st.columns(len(STUDY_ACTIONS))
        for column, (task, (label, template)) in zip(columns, STUDY_ACTIONS.items()):
            if column.button(label, key=f"study_{task}", width="stretch"):
                st.session_state["pending_question"] = template.format(course=course)
                st.session_state["pending_task"] = task
        # Studenti mund të ngjitë materialin e vet, që përdoret si burim
        with st.expander("Materiali im (opsional)"):
            st.text_area(
                "Ngjit shënimet ose materialin e lëndës. Përdoret si burim për quiz, flashcards dhe përmbledhje.",
                key="user_material", height=140,
            )
        st.button("Dil nga Study Mode", key="leave_study", on_click=leave_study_mode)
    return courses


# Paneli i studentit aktual, ose vetëm etiketa e modalitetit për studentin potencial
role_label = accounts.ROLES[role]
if role == "current":
    current_student_panel()
else:
    st.caption(f"Modaliteti: {role_label}")

# Butoni i dytë 'Pastro bisedën', në krye të bisedës
if st.session_state["messages"] and st.button("Pastro bisedën", key="clear_main"):
    st.session_state["messages"] = []
    st.session_state["followups"] = 0
    st.rerun()


# Njoftimi i vogël kur biseda përditëson profilin
def show_profile_changes(changes: list[str]) -> None:
    if changes:
        st.caption("Profili u përditësua: " + "; ".join(changes))


# Kutia 'Burimet e përdorura': titujt, faqet dhe pjesët e dokumenteve
def show_sources(sources: list[dict], chunks: list[dict]) -> None:
    with st.expander("Burimet e përdorura"):
        badges = "".join(f'<span class="citation">{source["program"]}</span>' for source in sources)
        st.markdown(badges, unsafe_allow_html=True)
        for source in sources:
            pages = source.get("pages")
            st.caption(f"{source['source']}" + (f", f. {', '.join(str(p) for p in sorted(pages))}" if pages else ""))
        st.divider()
        for chunk in chunks:
            page = f", f. {chunk['page']}" if chunk.get("page") else ""
            st.caption(f"{chunk['program']}{page}")
            st.text(chunk["text"])


# Rishfaq historinë e bisedës pas çdo rifreskimi të faqes
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_profile_changes(message.get("profile_changes", []))
        if message["role"] == "assistant" and message.get("sources"):
            show_sources(message["sources"], message["chunks"])

# Pyetja vjen nga kutia e shkrimit ose nga butonat e gatshëm (pending_question)
typed_question = st.chat_input("Shkruaj pyetjen tënde për UETassist")
question = typed_question or st.session_state.pop("pending_question", None)
pending_task = st.session_state.pop("pending_task", None)
# Veprimet e shpejta e dinë detyrën; kur shkruhet tekst i lirë, detyra përcaktohet nga modeli
forced_task = None if typed_question else pending_task

# Përpunon pyetjen: e shton te historia, e dërgon te këshilltari dhe shfaq përgjigjen
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Duke kërkuar në dokumente..."):
                # Pika kryesore: këshilltari zgjedh rrugën (RAG standard, personalizim ose Study Mode)
                result = advisor.respond(
                    question,
                    profile=st.session_state.get("profile"),
                    history=st.session_state["messages"][:-1],
                    followups_asked=st.session_state.get("followups", 0),
                    role=role,
                    academic=academic,
                    curriculum=curriculum_data,
                    study_course=st.session_state.get("study_course") if role == "current" else None,
                    forced_task=forced_task,
                    user_material=st.session_state.get("user_material", "") if role == "current" else "",
                )
        # Mungon çelësi OPENAI_API_KEY
        except llm.MissingApiKeyError as error:
            st.error(str(error))
        except Exception as error:  # gabime rrjeti, kuota, çelës i pavlefshëm
            st.error(
                "Nuk u mor përgjigje nga shërbimi i modelit. Kontrollo lidhjen e "
                "internetit dhe çelësin OPENAI_API_KEY te skedari .env, pastaj provo përsëri."
            )
            with st.expander("Detaje teknike"):
                st.code(f"{type(error).__name__}: {error}")
        else:
            # Ruan profilin e përditësuar dhe numëron pyetjet ndjekëse radhazi
            st.session_state["profile"] = result["profile"]
            st.session_state["followups"] = st.session_state.get("followups", 0) + 1 if result["follow_up"] else 0
            st.markdown(result["answer"])
            show_profile_changes(result["profile_changes"])
            # Shfaq burimet vetëm kur ka
            if result["sources"]:
                show_sources(result["sources"], result["chunks"])
            # Ruan përgjigjen e asistentit te historia e bisedës
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "chunks": result["chunks"],
                    "profile_changes": result["profile_changes"],
                }
            )
            # Ruan profilin te llogaria dhe rifreskon shiritin anësor
            if result["profile_changes"]:
                save_user()
                st.rerun()  # rifreskon përmbledhjen e profilit te shiriti anësor
