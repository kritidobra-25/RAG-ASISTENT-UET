"""
Hapi 1: nxjerrja e tekstit nga PDF dhe ndarja në segmente (chunking).

Dy përmirësime krahasuar me versionin fillestar:
1. Ndarja respekton rreshtat. Tabelat e planit mësimor kanë një lëndë për
   rresht, kështu që një lëndë nuk pritet më në mes.
2. Çdo segment fillon me emrin e programit ("Programi: ..."). Shumë lëndë
   (p.sh. "Metoda kërkimi në inxhinieri") shfaqen te të gjitha programet, dhe
   pa këtë emër segmentet duken identike për kërkimin semantik.

Ekzekutim i drejtpërdrejtë (vetëm teston ndarjen, pa API):
    python src/extract_and_chunk.py
"""

import re
import sys
from pathlib import Path

from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR

# Emrat e programeve, njohur nga fjalë kyçe në emrin e skedarit.
# Rendi ka rëndësi: rregullat specifike (MP/MSH) para atyre të përgjithshme,
# "informatike-e-aplikuar" kontrollohet para "informatike",
# dhe "informatike" para "teknologji".
PROGRAM_RULES = [
    ("mp-teknologji-informacioni-e-aplikuar", "Master Profesional në Teknologji Informacioni e Aplikuar në Financë"),
    ("msh-finance", "Master i Shkencave në Financë"),
    ("msh-informatike-ekonomike", "Master i Shkencave në Informatikë Ekonomike"),
    ("msh-administrim", "Master i Shkencave në Administrim Biznesi"),
    ("finance", "Master Profesional në Financë"),
    ("menaxhim", "Master Profesional në Menaxhim Biznesi"),
    ("informatike-e-aplikuar", "Master Profesional në Informatikë e Aplikuar"),
    ("ndertimi", "Master i Shkencave në Inxhinieri Ndërtimi"),
    ("elektrike", "Master i Shkencave në Inxhinieri Elektrike"),
    ("mekanike", "Master i Shkencave në Inxhinieri Mekanike"),
    ("informatike", "Master i Shkencave në Inxhinieri Informatike"),
    ("teknologji", "Master i Shkencave në Teknologji Informacioni"),
]


def program_from_filename(filename: str) -> str:
    """Kthen emrin e programit nga emri i skedarit.

    Për dokumente të reja (rregullore, kalendar etj.) që nuk përputhen me
    asnjë program, kthehet emri i skedarit pa shtesë.
    """
    name = filename.lower()
    for keyword, program in PROGRAM_RULES:
        if keyword in name:
            return program
    stem = Path(filename).stem
    return re.sub(r"[_\-]+", " ", stem).strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Nxjerr tekstin nga të gjitha faqet e një PDF."""
    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def extract_text(path: Path) -> str:
    """Lexon tekstin nga një PDF ose nga një skedar .txt (UTF-8)."""
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    return extract_text_from_pdf(path)


def clean_text(text: str) -> str:
    """Heq hapësirat e tepërta dhe rreshtat bosh."""
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Ndan tekstin në segmente me madhësi afërsisht chunk_size karaktere.

    Segmentet ndërtohen nga rreshta të plotë. Rreshtat e fundit të çdo
    segmenti (deri në `overlap` karaktere) përsëriten në fillim të segmentit
    pasardhës, që konteksti të mos humbasë në kufij.
    """
    # Rreshtat shumë të gjatë ndahen te hapësira më e afërt.
    lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        while len(line) > chunk_size:
            cut = line.rfind(" ", 0, chunk_size)
            if cut <= 0:
                cut = chunk_size
            lines.append(line[:cut].strip())
            line = line[cut:].strip()
        if line:
            lines.append(line)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > chunk_size:
            chunks.append("\n".join(current))
            # Mbivendosja: mban rreshtat e fundit brenda kufirit `overlap`.
            carry: list[str] = []
            carry_len = 0
            for previous in reversed(current):
                if carry_len + len(previous) + 1 > overlap:
                    break
                carry.insert(0, previous)
                carry_len += len(previous) + 1
            current, current_len = carry, carry_len
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))
    return chunks


def load_and_chunk_all_pdfs(data_dir: Path = DATA_DIR) -> list[dict]:
    """Lexon të gjitha PDF-të dhe skedarët .txt te data_dir dhe kthen segmentet.

    Skedarët .txt shërbejnë për të shtuar tekst të korrigjuar me dorë (p.sh. një
    tabelë që nuk nxirret mirë nga PDF-ja). Çdo element ka: id, text,
    source (skedari), program, chunk_index.
    """
    data_dir = Path(data_dir)
    all_chunks: list[dict] = []

    files = sorted(
        path for path in data_dir.iterdir() if path.suffix.lower() in {".pdf", ".txt"}
    )
    for path in files:
        program = program_from_filename(path.name)
        text = clean_text(extract_text(path))
        pieces = chunk_text(text)
        for index, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "id": f"{path.stem}_{index:03d}",
                    "text": f"Programi: {program}\n{piece}",
                    "source": path.name,
                    "program": program,
                    "chunk_index": index,
                }
            )
        print(f"  {path.name}: {len(pieces)} segmente  ({program})")
    return all_chunks


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"Dosja e dokumenteve: {DATA_DIR}")
    chunks = load_and_chunk_all_pdfs()
    if not chunks:
        print("Nuk u gjet asnjë dokument. Vendosi PDF-të te dosja data/.")
        sys.exit(1)
    print(f"\nTotali: {len(chunks)} segmente")
    print("\n--- Shembull: segmenti i parë ---")
    print(chunks[0]["text"][:500])
