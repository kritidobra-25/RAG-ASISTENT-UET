"""Teste për rrugët e reja (llogari, profil, rekomandim, përputhje me planin). Pa API."""

import sys
from pathlib import Path

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import accounts  # noqa: E402
import llm  # noqa: E402
import rag  # noqa: E402
import student_profile  # noqa: E402


@pytest.fixture
def users_file(tmp_path):
    return tmp_path / "users.json"


def test_register_login_roundtrip(users_file):
    accounts.register("Ana_01", "sekret1", accounts.ROLE_ACTUAL, users_file)
    account = accounts.login("ana_01", "sekret1", users_file)
    assert account["role"] == accounts.ROLE_ACTUAL and account["profile"] is None
    assert "sekret1" not in users_file.read_text(encoding="utf-8")


def test_login_rejects_bad_credentials(users_file):
    accounts.register("ana", "sekret1", accounts.ROLE_ACTUAL, users_file)
    with pytest.raises(accounts.AccountError):
        accounts.login("ana", "gabim", users_file)
    with pytest.raises(accounts.AccountError):
        accounts.login("askush", "sekret1", users_file)


def test_register_validation(users_file):
    accounts.register("ana", "sekret1", accounts.ROLE_POTENTIAL, users_file)
    for username, password, role in [
        ("ana", "sekret1", accounts.ROLE_ACTUAL),   # i zënë (pa dallim shkronjash)
        ("ANA", "sekret1", accounts.ROLE_ACTUAL),
        ("a", "sekret1", accounts.ROLE_ACTUAL),     # shumë i shkurtër
        ("beni", "123", accounts.ROLE_ACTUAL),      # fjalëkalim i shkurtër
        ("beni", "sekret1", "tjeter"),              # rol i panjohur
    ]:
        with pytest.raises(accounts.AccountError):
            accounts.register(username, password, role, users_file)


def test_profile_is_saved_and_returned_on_login(users_file):
    accounts.register("ana", "sekret1", accounts.ROLE_POTENTIAL, users_file)
    accounts.save_profile("ana", {"fusha": "Inxhinieri Elektrike"}, users_file)
    assert accounts.login("ana", "sekret1", users_file)["profile"] == {"fusha": "Inxhinieri Elektrike"}


def test_profile_validation():
    ok = {"programi": "P", "viti": 2, "semestri": 3}
    assert student_profile.validate(accounts.ROLE_ACTUAL, ok) == []
    assert student_profile.validate(accounts.ROLE_ACTUAL, {**ok, "semestri": 1})  # semestri i gabuar për vitin
    assert student_profile.validate(accounts.ROLE_ACTUAL, {"programi": " "})
    assert student_profile.validate(
        accounts.ROLE_POTENTIAL, {"interesat": "AI", "fusha": "Inxhinieri Informatike", "niveli": "Master Shkencor"}
    ) == []
    assert student_profile.validate(accounts.ROLE_POTENTIAL, {"interesat": ""})


PROGRAM_A = "Master i Shkencave në Inxhinieri Elektrike"
PROGRAM_B = "Master i Shkencave në Inxhinieri Mekanike"


@pytest.fixture
def assistant(monkeypatch):
    """RagAssistant mbi një bazë në memorie, me embeddings dhe LLM të simuluar."""
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("test_kb")
    except Exception:
        pass
    collection = client.create_collection("test_kb")
    collection.add(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=[f"Programi: {PROGRAM_A}\nAutomatizim", f"Programi: {PROGRAM_B}\nEnergjitikë"],
        metadatas=[
            {"source": "a.pdf", "program": PROGRAM_A, "chunk_index": 0},
            {"source": "b.pdf", "program": PROGRAM_B, "chunk_index": 0},
        ],
    )
    instance = rag.RagAssistant.__new__(rag.RagAssistant)
    instance.collection = collection

    calls = {"queries": [], "prompts": []}

    def fake_embed(texts):
        calls["queries"].append(texts[0])
        return [[1.0, 0.0] if "Elektrike" in texts[0] else [0.0, 1.0]]

    def fake_chat(system, user):
        calls["prompts"].append((system, user))
        return "përgjigje"

    monkeypatch.setattr(llm, "embed_texts", fake_embed)
    monkeypatch.setattr(llm, "chat_completion", fake_chat)
    instance.calls = calls
    return instance


def test_answer_without_profile_is_unchanged(assistant):
    result = assistant.answer("pyetje Elektrike")
    system, user = assistant.calls["prompts"][0]
    assert system == rag.SYSTEM_PROMPT and "Profili" not in user
    assert {s["program"] for s in result["sources"]} == {PROGRAM_A, PROGRAM_B}


def test_actual_student_is_restricted_to_own_program(assistant):
    profile = {"programi": PROGRAM_B, "viti": 1, "semestri": 1}
    result = assistant.answer(
        "pyetje Elektrike", role=accounts.ROLE_ACTUAL, profile=profile, mode=rag.MODE_STUDY
    )
    assert [s["program"] for s in result["sources"]] == [PROGRAM_B]
    system, user = assistant.calls["prompts"][0]
    assert rag.MODE_PROMPTS[rag.MODE_STUDY] in system
    assert "Viti: 1" in user and "Semestri: 1" in user


def test_current_courses_uses_year_semester_and_program(assistant):
    profile = {"programi": PROGRAM_A, "viti": 2, "semestri": 3}
    result = assistant.current_courses(profile)
    assert "viti 2 semestri 3" in assistant.calls["queries"][0]
    assert [s["program"] for s in result["sources"]] == [PROGRAM_A]


def test_recommendation_uses_interests_without_program_filter(assistant):
    profile = {"interesat": "automatizim", "fusha": "Inxhinieri Elektrike", "niveli": "Master Shkencor"}
    result = assistant.recommend_programs(profile)
    assert "Inxhinieri Elektrike" in assistant.calls["queries"][0]
    assert {s["program"] for s in result["sources"]} == {PROGRAM_A, PROGRAM_B}
    _, user = assistant.calls["prompts"][0]
    assert "student potencial" in user
