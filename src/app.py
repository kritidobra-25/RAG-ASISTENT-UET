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
from rag import KnowledgeBaseMissingError, RagAssistant  # noqa: E402

EXAMPLE_QUESTIONS = [
    "Cilat janë profilet e Master Shkencor në Inxhinieri Mekanike?",
    "Sa ECTS ka lënda Inteligjenca artificiale në Inxhinieri Informatike?",
    "Cilat janë mundësitë e punësimit pas Master në Inxhinieri Ndërtimi?",
    "Sa është tarifa vjetore e programit?",
]

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

    st.subheader("Pyetje shembull")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, use_container_width=True):
            st.session_state["pending_question"] = example

    if st.button("Pastro bisedën", use_container_width=True):
        st.session_state["messages"] = []
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


def show_sources(sources: list[dict], chunks: list[dict]) -> None:
    with st.expander("Burimet e përdorura"):
        badges = "".join(f'<span class="citation">{source["program"]}</span>' for source in sources)
        st.markdown(badges, unsafe_allow_html=True)
        for source in sources:
            st.caption(source["source"])
        st.divider()
        for number, chunk in enumerate(chunks, start=1):
            st.caption(f"Fragmenti {number} (largësia {chunk['distance']:.3f})")
            st.text(chunk["text"])


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
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
                result = assistant.answer(question)
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
            st.markdown(result["answer"])
            if result["sources"]:
                show_sources(result["sources"], result["chunks"])
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "chunks": result["chunks"],
                }
            )
