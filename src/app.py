"""
Hapi 5: ndërfaqja bisedore (Streamlit).

Nise nga dosja kryesore e projektit:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import config  # noqa: E402
import llm  # noqa: E402
from student_profile import is_empty, new_profile, profile_to_text  # noqa: E402
from advisor import Advisor  # noqa: E402
from rag import KnowledgeBaseMissingError, RagAssistant  # noqa: E402

LEVEL_OPTIONS = ["", "Bachelor", "Master Profesional", "Master i Shkencave"]
AREA_OPTIONS = ["", "Teknike / IT", "Biznes / Financë", "Inxhinieri", "E përzier"]
PATH_QUESTION = "Më sugjero rrugën akademike më të përshtatshme sipas profilit tim."

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "uet_logo.png"

st.set_page_config(page_title="UETassist", page_icon=str(LOGO_PATH), layout="centered")

st.markdown(
    f"<style>{(Path(__file__).resolve().parent / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Duke ngarkuar bazën e njohurive...")
def load_assistant() -> RagAssistant:
    return RagAssistant()


try:
    assistant = load_assistant()
    advisor = Advisor(assistant)
except KnowledgeBaseMissingError as error:
    st.error(
        f"{error}\n\nHap terminalin në dosjen kryesore të projektit dhe ekzekuto "
        "`python src/build_index.py`, pastaj rifresko këtë faqe."
    )
    st.stop()

# ---------- Shiriti anësor ----------
with st.sidebar:
    st.image(str(LOGO_PATH), use_container_width=True)
    st.header("Rreth asistentit")
    st.write(
        "Asistenti u përgjigjet pyetjeve të studentëve aktualë dhe kandidatëve "
        "duke u bazuar vetëm në dokumentet zyrtare të UET-së."
    )
    stats = assistant.stats()
    st.write(f"**Baza e njohurive:** {stats['segments']} segmente nga {len(stats['programs'])} dokumente ose programe.")
    with st.expander("Programet dhe dokumentet në bazë"):
        for program in stats["programs"]:
            st.write(f"- {program}")
    st.caption(f"Modeli: {config.CHAT_MODEL} | Embeddings: {config.EMBEDDING_MODEL}")

    with st.expander("Profili im", expanded=not is_empty(st.session_state.get("profile"))):
        st.caption(
            "Profili plotësohet vetë nga biseda dhe përditësohet teksa flet. Ruhet vetëm "
            "në këtë sesion. Mos shkruaj emër ose të dhëna personale."
        )
        profile = st.session_state.setdefault("profile", new_profile())
        summary = profile_to_text(profile)
        st.markdown(summary if summary else "_Profili është ende bosh._")

        with st.form("profile_form"):
            st.caption("Opsionale: ndrysho profilin manualisht.")
            background = st.text_input("Sfondi akademik", profile["academic_background"], placeholder="p.sh. Bachelor në Administrim Biznesi")
            goal = st.text_input("Objektivi i karrierës", profile["career_goal"], placeholder="p.sh. Data Engineer")
            interests = st.text_input("Interesa (me presje)", ", ".join(profile["interests"]))
            skills = st.text_input("Aftësi teknike (me presje)", ", ".join(profile["technical_skills"]), placeholder="p.sh. SQL, Python")
            level = st.selectbox("Niveli akademik", LEVEL_OPTIONS, index=LEVEL_OPTIONS.index(profile["current_level"]) if profile["current_level"] in LEVEL_OPTIONS else 0)
            area = st.selectbox("Fusha e preferuar", AREA_OPTIONS, index=AREA_OPTIONS.index(profile["preferred_area"]) if profile["preferred_area"] in AREA_OPTIONS else 0)
            if st.form_submit_button("Ruaj profilin", use_container_width=True):
                profile["academic_background"] = background.strip()
                profile["career_goal"] = goal.strip()
                profile["interests"] = [x.strip() for x in interests.split(",") if x.strip()]
                profile["technical_skills"] = [x.strip() for x in skills.split(",") if x.strip()]
                profile["current_level"] = level
                profile["preferred_area"] = area
                st.rerun()

        if st.button("Gjenero rrugën akademike", use_container_width=True, key="gen_path"):
            st.session_state["pending_question"] = PATH_QUESTION
        if st.button("Rivendos profilin", use_container_width=True, key="reset_profile"):
            st.session_state["profile"] = new_profile()
            st.session_state["followups"] = 0
            st.rerun()

    if st.button("Pastro bisedën", use_container_width=True, key="clear_sidebar"):
        st.session_state["messages"] = []
        st.session_state["followups"] = 0
        st.rerun()

# ---------- Biseda ----------
logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
logo_col.image(str(LOGO_PATH), width=72)
title_col.title("UETassist")
st.caption(
    "Përgjigjet vijnë vetëm nga dokumentet zyrtare të ngarkuara. Për vendime të "
    "rëndësishme, konfirmo me administratën e UET-së. Mos shkruaj të dhëna personale."
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
            st.caption(f"{source['source']}" + (f", faqja {', '.join(str(p) for p in sorted(pages))}" if pages else ""))
        st.divider()
        for number, chunk in enumerate(chunks, start=1):
            page = f", faqja {chunk['page']}" if chunk.get("page") else ""
            st.caption(f"Fragmenti {number}: {chunk['source']}{page} (largësia {chunk['distance']:.3f})")
            st.text(chunk["text"])


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_profile_changes(message.get("profile_changes", []))
        if message["role"] == "assistant" and message.get("sources"):
            show_sources(message["sources"], message["chunks"])

typed_question = st.chat_input("Shkruaj pyetjen tënde për UET-në")
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
                st.rerun()  # rifreskon përmbledhjen e profilit te shiriti anësor
