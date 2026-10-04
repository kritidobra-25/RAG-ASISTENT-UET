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

# ---------- Shiriti anësor ----------
with st.sidebar:
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
st.title("Asistenti i UET")
st.caption(
    "Përgjigjet vijnë vetëm nga dokumentet zyrtare të ngarkuara. Për vendime të "
    "rëndësishme, konfirmo me administratën e UET-së. Mos shkruaj të dhëna personale."
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def show_sources(sources: list[dict], chunks: list[dict]) -> None:
    with st.expander("Burimet e përdorura"):
        for source in sources:
            st.write(f"**{source['program']}**  \n`{source['source']}`")
        st.divider()
        for number, chunk in enumerate(chunks, start=1):
            st.caption(f"Fragmenti {number} (largësia {chunk['distance']:.3f})")
            st.text(chunk["text"])


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.caption(f"Koha e përgjigjes: {message['seconds']:.1f} s")
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
            st.caption(f"Koha e përgjigjes: {result['total_seconds']:.1f} s")
            show_sources(result["sources"], result["chunks"])
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "chunks": result["chunks"],
                    "seconds": result["total_seconds"],
                }
            )
