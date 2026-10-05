"""
Hapi 5: ndërfaqja bisedore (Streamlit).

Rrjedha: llogari / hyrje -> lloji i studentit -> profili -> rruga e përdorimit
(informacion akademik, asistent studimi ose orientim për programet) -> RAG.

Nise nga dosja kryesore e projektit:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import accounts  # noqa: E402
import config  # noqa: E402
import llm  # noqa: E402
import student_profile  # noqa: E402
from accounts import ROLE_ACTUAL, ROLE_POTENTIAL  # noqa: E402
from rag import (  # noqa: E402
    MODE_ACADEMIC,
    MODE_GUIDANCE,
    MODE_STUDY,
    KnowledgeBaseMissingError,
    RagAssistant,
)

ROLE_LABELS = {ROLE_ACTUAL: "Student aktual", ROLE_POTENTIAL: "Student potencial"}

# Rrugët e disponueshme për secilin lloj studenti.
MODES_BY_ROLE = {
    ROLE_ACTUAL: (MODE_ACADEMIC, MODE_STUDY),
    ROLE_POTENTIAL: (MODE_GUIDANCE,),
}
MODE_LABELS = {
    MODE_ACADEMIC: "Informacion akademik",
    MODE_STUDY: "Asistent studimi",
    MODE_GUIDANCE: "Orientim për programet",
}
MODE_INTROS = {
    MODE_ACADEMIC: "Pyetje për lëndët, kreditet ECTS, vitin dhe semestrin e programit tënd.",
    MODE_STUDY: "Ndihmë për të organizuar studimin në bazë të planit mësimor të programit tënd.",
    MODE_GUIDANCE: "Pyetje për programet, profilet, lëndët dhe mundësitë e punësimit në UET.",
}

EXAMPLE_QUESTIONS = {
    MODE_ACADEMIC: [
        "Sa ECTS ka lënda Inteligjenca artificiale?",
        "Cilat janë profilet e programit tim?",
        "Cilat lëndë janë me zgjedhje?",
    ],
    MODE_STUDY: [
        "Sa kredite ECTS kam gjithsej këtë semestër?",
        "Cilat lëndë më mbeten pas këtij semestri?",
        "Si mund t'i grupoj lëndët e këtij semestri sipas temës?",
    ],
    MODE_GUIDANCE: [
        "Cilat janë profilet e Master Shkencor në Inxhinieri Mekanike?",
        "Cilat janë mundësitë e punësimit pas Master në Inxhinieri Ndërtimi?",
        "Sa është tarifa vjetore e programit?",
    ],
}

st.set_page_config(page_title="Asistenti i UET", layout="centered")


@st.cache_resource(show_spinner="Duke ngarkuar bazën e njohurive...")
def load_assistant() -> RagAssistant:
    return RagAssistant()


try:
    assistant = load_assistant()
except KnowledgeBaseMissingError as error:
    st.error(
        f"{error}\n\nHap terminalin në dosjen kryesore të projektit dhe ekzekuto "
        "`python src/build_index.py`, pastaj rifresko këtë faqe."
    )
    st.stop()

stats = assistant.stats()


# ---------- Hyrja / regjistrimi ----------
def show_login() -> None:
    st.title("Asistenti i UET")
    st.caption("Hyr ose krijo një llogari që asistenti t'i përshtatet profilit tënd.")
    login_tab, register_tab = st.tabs(["Hyr", "Krijo llogari"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Emri i përdoruesit")
            password = st.text_input("Fjalëkalimi", type="password")
            if st.form_submit_button("Hyr", use_container_width=True):
                try:
                    st.session_state["account"] = accounts.login(username, password)
                except accounts.AccountError as error:
                    st.error(str(error))
                else:
                    st.rerun()

    with register_tab:
        with st.form("register_form"):
            username = st.text_input("Emri i përdoruesit", key="reg_username")
            password = st.text_input("Fjalëkalimi", type="password", key="reg_password")
            role = st.radio(
                "Çfarë je?",
                accounts.ROLES,
                format_func=lambda r: ROLE_LABELS[r],
                help="Studenti aktual studion tashmë në UET. Studenti potencial po e shqyrton ardhmen e tij.",
            )
            if st.form_submit_button("Krijo llogari", use_container_width=True):
                try:
                    accounts.register(username, password, role)
                    st.session_state["account"] = accounts.login(username, password)
                except accounts.AccountError as error:
                    st.error(str(error))
                else:
                    st.rerun()


# ---------- Profili ----------
def show_profile_form(account: dict) -> None:
    role = account["role"]
    saved = account.get("profile") or {}
    st.title("Profili yt")
    st.caption(f"{ROLE_LABELS[role]}. Këto të dhëna përdoren vetëm për të përshtatur përgjigjet.")

    with st.form("profile_form"):
        if role == ROLE_ACTUAL:
            programs = stats["programs"]
            program_index = programs.index(saved["programi"]) if saved.get("programi") in programs else 0
            profile = {
                "fakulteti": st.text_input("Fakulteti", value=saved.get("fakulteti", "")),
                "departamenti": st.text_input("Departamenti", value=saved.get("departamenti", "")),
                "programi": st.selectbox("Programi", programs, index=program_index),
                "specializimi": st.text_input("Specializimi (profili)", value=saved.get("specializimi", "")),
                "viti": st.selectbox(
                    "Viti", student_profile.YEARS, index=student_profile.YEARS.index(saved.get("viti", 1))
                ),
                "semestri": st.selectbox(
                    "Semestri (1-2 për vitin I, 3-4 për vitin II)",
                    student_profile.SEMESTERS,
                    index=student_profile.SEMESTERS.index(saved.get("semestri", 1)),
                ),
            }
        else:
            profile = {
                "arsimi": st.text_input(
                    "Arsimi aktual", value=saved.get("arsimi", ""), placeholder="p.sh. Bachelor në Inxhinieri Civile"
                ),
                "interesat": st.text_area("Interesat", value=saved.get("interesat", "")),
                "fusha": st.selectbox(
                    "Fusha e interesit",
                    student_profile.FIELDS,
                    index=student_profile.FIELDS.index(saved.get("fusha", student_profile.FIELDS[0])),
                ),
                "niveli": st.selectbox(
                    "Niveli i studimeve që kërkon",
                    student_profile.LEVELS,
                    index=student_profile.LEVELS.index(saved.get("niveli", "Master Shkencor")),
                ),
                "objektivat": st.text_area(
                    "Objektivat", value=saved.get("objektivat", ""), placeholder="p.sh. punë në industri, kërkim shkencor"
                ),
            }
        submitted = st.form_submit_button("Ruaj profilin", use_container_width=True)

    if submitted:
        problems = student_profile.validate(role, profile)
        if problems:
            for problem in problems:
                st.error(problem)
            return
        st.session_state["account"] = accounts.save_profile(account["username"], profile)
        st.session_state["editing_profile"] = False
        st.session_state["messages"] = {}
        st.session_state.pop("recommendation", None)
        st.session_state.pop("current_courses", None)
        st.rerun()


# ---------- Biseda ----------
def show_sources(sources: list[dict], chunks: list[dict]) -> None:
    with st.expander("Burimet e përdorura"):
        for source in sources:
            st.write(f"**{source['program']}**  \n`{source['source']}`")
        st.divider()
        for number, chunk in enumerate(chunks, start=1):
            st.caption(f"Fragmenti {number} (largësia {chunk['distance']:.3f})")
            st.text(chunk["text"])


def show_result(result: dict) -> None:
    st.markdown(result["answer"])
    st.caption(f"Koha e përgjigjes: {result['total_seconds']:.1f} s")
    show_sources(result["sources"], result["chunks"])


def run_with_errors(action):
    """Ekzekuton një thirrje te RAG dhe shfaq gabimet në shqip. Kthen rezultatin ose None."""
    try:
        with st.spinner("Duke kërkuar në dokumente..."):
            return action()
    except llm.MissingApiKeyError as error:
        st.error(str(error))
    except Exception as error:  # gabime rrjeti, kuota, çelës i pavlefshëm
        st.error(
            "Nuk u mor përgjigje nga shërbimi i modelit. Kontrollo lidhjen e "
            "internetit dhe çelësin OPENAI_API_KEY te skedari .env, pastaj provo përsëri."
        )
        with st.expander("Detaje teknike"):
            st.code(f"{type(error).__name__}: {error}")
    return None


def show_chat(account: dict, mode: str) -> None:
    role, profile = account["role"], account["profile"]
    messages = st.session_state["messages"].setdefault(mode, [])

    # Pyetja e dërguar nga shiriti anësor ose nga kutia e bisedës
    typed_question = st.chat_input("Shkruaj pyetjen tënde për UET-në")
    question = typed_question or st.session_state.pop("pending_question", None)

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                st.caption(f"Koha e përgjigjes: {message['seconds']:.1f} s")
                show_sources(message["sources"], message["chunks"])

    if not question:
        return
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        result = run_with_errors(
            lambda: assistant.answer(question, role=role, profile=profile, mode=mode)
        )
        if result:
            show_result(result)
            messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "chunks": result["chunks"],
                    "seconds": result["total_seconds"],
                }
            )


def show_current_courses(profile: dict) -> None:
    """Përputhja me planin mësimor: lëndët aktuale të studentit."""
    label = f"Lëndët e mia: viti {profile['viti']}, semestri {profile['semestri']}"
    with st.expander(label, expanded=False):
        if st.button("Shfaq lëndët e semestrit", key="load_courses"):
            result = run_with_errors(lambda: assistant.current_courses(profile))
            if result:
                st.session_state["current_courses"] = result
        if "current_courses" in st.session_state:
            show_result(st.session_state["current_courses"])


def show_recommendations(profile: dict) -> None:
    """Motori i rekomandimeve për studentin potencial."""
    with st.expander("Programet që të përshtaten", expanded=True):
        if st.button("Gjej programet për profilin tim", key="load_recommendation"):
            result = run_with_errors(lambda: assistant.recommend_programs(profile))
            if result:
                st.session_state["recommendation"] = result
        if "recommendation" in st.session_state:
            show_result(st.session_state["recommendation"])


# ---------- Rrjedha kryesore ----------
st.session_state.setdefault("messages", {})
account = st.session_state.get("account")

if not account:
    show_login()
    st.stop()

if not account.get("profile") or st.session_state.get("editing_profile"):
    show_profile_form(account)
    st.stop()

role, profile = account["role"], account["profile"]
modes = MODES_BY_ROLE[role]

with st.sidebar:
    st.header(account["username"])
    st.caption(ROLE_LABELS[role])
    if role == ROLE_ACTUAL:
        st.write(f"{profile['programi']}  \nViti {profile['viti']}, semestri {profile['semestri']}")
    else:
        st.write(f"{profile['fusha']}  \n{profile['niveli']}")
    col_edit, col_logout = st.columns(2)
    if col_edit.button("Profili", use_container_width=True):
        st.session_state["editing_profile"] = True
        st.rerun()
    if col_logout.button("Dil", use_container_width=True):
        for key in ("account", "editing_profile", "messages", "recommendation", "current_courses"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    mode = (
        st.radio("Çfarë të ndihmoj?", modes, format_func=lambda m: MODE_LABELS[m])
        if len(modes) > 1
        else modes[0]
    )

    st.subheader("Pyetje shembull")
    for example in EXAMPLE_QUESTIONS[mode]:
        if st.button(example, use_container_width=True):
            st.session_state["pending_question"] = example

    if st.button("Pastro bisedën", use_container_width=True):
        st.session_state["messages"][mode] = []
        st.rerun()

    st.divider()
    st.write(f"**Baza e njohurive:** {stats['segments']} segmente nga {len(stats['programs'])} dokumente ose programe.")
    with st.expander("Programet dhe dokumentet në bazë"):
        for program in stats["programs"]:
            st.write(f"- {program}")
    st.caption(f"Modeli: {config.CHAT_MODEL} | Embeddings: {config.EMBEDDING_MODEL}")

st.title("Asistenti i UET")
st.subheader(MODE_LABELS[mode])
st.caption(MODE_INTROS[mode])
st.caption(
    "Përgjigjet vijnë vetëm nga dokumentet zyrtare të ngarkuara. Për vendime të "
    "rëndësishme, konfirmo me administratën e UET-së. Pyetja dhe fushat e profilit "
    "dërgohen te ofruesi i modelit; mos shkruaj të dhëna personale."
)

if role == ROLE_ACTUAL and mode == MODE_ACADEMIC:
    show_current_courses(profile)
elif role == ROLE_POTENTIAL:
    show_recommendations(profile)

show_chat(account, mode)
