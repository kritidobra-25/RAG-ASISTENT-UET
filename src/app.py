"""
Hapi 5: ndërfaqja bisedore (Streamlit).

Nise nga dosja kryesore e projektit:
    streamlit run src/app.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shumë serverë (p.sh. Streamlit Community Cloud) kanë SQLite të vjetër që ChromaDB
# nuk e pranon. Nëse pysqlite3 është i instaluar, përdoret ai.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st  # noqa: E402

import accounts  # noqa: E402
import config  # noqa: E402
import curriculum  # noqa: E402
import llm  # noqa: E402
from student_profile import academic_to_text, is_empty, new_academic, new_profile, profile_to_text  # noqa: E402
from advisor import Advisor  # noqa: E402
from build_index import build_index  # noqa: E402
from rag import KnowledgeBaseMissingError, RagAssistant  # noqa: E402

PROGRAM_OPTIONS = ["Master Profesional", "Master Shkencor"]
PATH_QUESTION = "Më sugjero rrugën akademike më të përshtatshme sipas profilit tim."

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "uet_logo.png"

st.set_page_config(page_title="UETassist", page_icon=str(LOGO_PATH), layout="centered")

st.markdown(
    f"<style>{(Path(__file__).resolve().parent / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


accounts.init_db()


def pick(options: list, value):
    """Indeksi i vlerës te lista, ose None (që selectbox të mbetet pa zgjedhje)."""
    return options.index(value) if value in options else None


def login_screen() -> None:
    logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
    logo_col.image(str(LOGO_PATH), width=72)
    title_col.title("UETassist")
    st.caption(
        "Hyr ose krijo një llogari që profili yt të ruhet. Përdor një pseudonim, jo emrin real. "
        "Biseda nuk ruhet, vetëm profili."
    )
    login_tab, register_tab = st.tabs(["Hyr", "Regjistrohu"])
    with login_tab, st.form("login_form"):
        username = st.text_input("Emri i përdoruesit")
        password = st.text_input("Fjalëkalimi", type="password")
        if st.form_submit_button("Hyr", width="stretch"):
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
                role = next((key for key, label in accounts.ROLES.items() if label == role_label), "")
                start_session(accounts.create_user(new_username, new_password, role))
                st.rerun()
            except accounts.AccountError as error:
                st.error(str(error))


def start_session(user: dict) -> None:
    st.session_state.update(
        user_id=user["id"], username=user["username"], role=user["role"],
        profile=user["profile"], academic=user["academic"], messages=[], followups=0,
    )


def save_user() -> None:
    accounts.save_user_data(
        st.session_state["user_id"], st.session_state["profile"], st.session_state["academic"], st.session_state["role"]
    )


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
    chosen = st.selectbox("Roli", role_labels, index=role_labels.index(accounts.ROLES[role]))
    if chosen != accounts.ROLES[role]:
        st.session_state["role"] = next(key for key, label in accounts.ROLES.items() if label == chosen)
        save_user()
        st.rerun()

    if role == "current":
        with st.expander("Profili im akademik", expanded=not academic.get("program")):
            st.caption(
                "Programi, viti dhe semestri përdoren për të gjetur lëndët e semestrit nga plani mësimor. "
                "Fakulteti, departamenti dhe specializimi nuk gjenden te dokumentet, prandaj i plotëson ti."
            )
            summary = academic_to_text(academic)
            st.markdown(summary if summary else "*Profili akademik është ende bosh.*")
            with st.form("academic_form"):
                faculty = st.text_input("Fakulteti", academic["faculty"])
                department = st.text_input("Departamenti", academic["department"])
                programs = curriculum.programs(curriculum_data)
                program = st.selectbox("Programi", programs, index=pick(programs, academic["program"]), placeholder="Zgjidh programin")
                specialization = st.text_input("Specializimi", academic["specialization"])
                year = st.selectbox("Viti", [1, 2], index=pick([1, 2], academic["year"]), placeholder="Zgjidh vitin")
                semester = st.selectbox("Semestri (brenda vitit)", [1, 2], index=pick([1, 2], academic["semester"]), placeholder="Zgjidh semestrin")
                if st.form_submit_button("Ruaj profilin akademik", width="stretch"):
                    academic.update(
                        faculty=faculty.strip(), department=department.strip(), program=program or "",
                        specialization=specialization.strip(), year=year or 0, semester=semester or 0,
                    )
                    save_user()
                    st.rerun()

    with st.expander("Profili im" if role == "prospective" else "Interesat dhe aftësitë", expanded=role == "prospective" and not is_empty(profile)):
        st.caption(
            "Ky profil plotësohet vetë nga biseda dhe përditësohet sipas informacionit që ndan. "
            "Mund ta ndryshosh edhe manualisht."
        )
        summary = profile_to_text(profile)
        st.markdown(summary if summary else "*Profili juaj është ende bosh.*")

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

        if role == "prospective" and st.button("Gjenero rrugën akademike", width="stretch", key="gen_path"):
            st.session_state["pending_question"] = PATH_QUESTION
        if st.button("Rivendos profilin", width="stretch", key="reset_profile"):
            st.session_state["profile"] = new_profile()
            if role == "current":
                st.session_state["academic"] = new_academic()
            st.session_state["followups"] = 0
            save_user()
            st.rerun()

    if st.button("Pastro bisedën", width="stretch", key="clear_sidebar"):
        st.session_state["messages"] = []
        st.session_state["followups"] = 0
        st.rerun()
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

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.session_state["messages"] and st.button("Pastro bisedën", key="clear_main"):
    st.session_state["messages"] = []
    st.session_state["followups"] = 0
    st.rerun()


def show_profile_changes(changes: list[str]) -> None:
    if changes:
        st.caption("Profili u përditësua: " + "; ".join(changes))


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


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_profile_changes(message.get("profile_changes", []))
        if message["role"] == "assistant" and message.get("sources"):
            show_sources(message["sources"], message["chunks"])

typed_question = st.chat_input("Shkruaj pyetjen tënde për UETassist")
question = typed_question or st.session_state.pop("pending_question", None)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Duke kërkuar në dokumente..."):
                result = advisor.respond(
                    question,
                    profile=st.session_state.get("profile"),
                    history=st.session_state["messages"][:-1],
                    followups_asked=st.session_state.get("followups", 0),
                    role=role,
                    academic=academic,
                    curriculum=curriculum_data,
                )
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
            st.session_state["profile"] = result["profile"]
            st.session_state["followups"] = st.session_state.get("followups", 0) + 1 if result["follow_up"] else 0
            st.markdown(result["answer"])
            show_profile_changes(result["profile_changes"])
            if result["sources"]:
                show_sources(result["sources"], result["chunks"])
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "chunks": result["chunks"],
                    "profile_changes": result["profile_changes"],
                }
            )
            if result["profile_changes"]:
                save_user()
                st.rerun()  # rifreskon përmbledhjen e profilit te shiriti anësor
