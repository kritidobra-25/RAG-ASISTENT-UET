# Asistenti RAG për UET: projekti i plotë

Asistent inteligjent që u përgjigjet pyetjeve të studentëve duke u bazuar vetëm në dokumentet zyrtare të UET-së (RAG me ChromaDB dhe OpenAI API). Ky projekt është pjesa praktike e tezës së masterit.

## Çfarë përmban

```
RAG-ASISTENT-UET/
├── data/                    PDF-të e UET-së (5 programe Master)
├── src/
│   ├── config.py            parametrat (modelet, madhësia e segmenteve, top-k)
│   ├── extract_and_chunk.py hapi 1: leximi i PDF dhe ndarja në segmente
│   ├── build_index.py       hapi 2: embeddings dhe ruajtja në ChromaDB
│   ├── rag.py               kërkimi dhe gjenerimi i përgjigjes
│   ├── query.py             hapi 3: testimi nga terminali
│   ├── evaluate.py          hapi 4: vlerësimi me pyetjet testuese
│   ├── app.py               hapi 5: ndërfaqja bisedore (Streamlit)
│   ├── accounts.py          llogaritë dhe hyrja (ruhen te users/, jo në git)
│   ├── student_profile.py   profili i studentit aktual dhe potencial
│   └── llm.py               të gjitha thirrjet drejt OpenAI
├── eval/test_questions.csv  25 pyetje testuese me përgjigje të pritura
├── .env.example             shablloni për çelësin API
└── requirements.txt         bibliotekat e nevojshme
```

## Para se të fillosh

**1. Vendose projektin jashtë OneDrive.** Kopjoje dosjen te `C:\Projekte\RAG-ASISTENT-UET`. OneDrive sinkronizon skedarët e bazës vektoriale gjatë përdorimit dhe shkakton gabimin "database is locked".

**2. Çelësi OpenAI.** Në platform.openai.com krijo një çelës te "API keys". Shto pak kredit te "Billing". Për këtë projekt kostoja është zakonisht disa cent, por kontrolloje te paneli i OpenAI.

## Hapat, një nga një

**1. Hap projektin në VS Code.** File → Open Folder → zgjidh `C:\Projekte\RAG-ASISTENT-UET` → Select Folder.

**2. Hap terminalin.** Terminal → New Terminal. Të gjitha komandat më poshtë shkruhen aty.

**3. Krijo mjedisin virtual dhe aktivizoje.**

```
python -m venv venv
venv\Scripts\activate
```

Para rreshtit duhet të shfaqet `(venv)`. Nëse Windows e bllokon aktivizimin, ekzekuto një herë `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, shkruaj `Y` dhe provo përsëri.

**4. Instalo bibliotekat.**

```
pip install -r requirements.txt
```

**5. Vendos çelësin.** Në panelin Explorer (majtas) kliko djathtas mbi `.env.example` → Copy, pastaj kliko djathtas në hapësirën bosh → Paste. Riemërtoje kopjen në `.env` (kliko djathtas → Rename). Hape dhe zëvendëso `sk-xxxx...` me çelësin tënd real. Ruaje me Ctrl+S.

**6. Testo ndarjen e dokumenteve (pa API, falas).**

```
python src\extract_and_chunk.py
```

Duhet të shohësh 5 dokumente dhe rreth 53 segmente gjithsej.

**7. Ndërto bazën e njohurive.**

```
python src\build_index.py
```

Duhet të përfundojë me "Baza përmban 53 segmente". Ekzekutohet sa herë ndryshojnë dokumentet.

**8. Provo asistentin në terminal.**

```
python src\query.py
```

Shkruaj p.sh. `Sa ECTS ka lënda Inteligjenca artificiale në Inxhinieri Informatike?`. Pritet përgjigje me 6 ECTS, vit II, semestri 1, dhe burimin. Shkruaj `dil` për të mbyllur.

**9. Hap ndërfaqen.**

```
streamlit run src\app.py
```

Hapet shfletuesi në `localhost:8501`. Nëse terminali kërkon email, shtyp Enter. Bëj screenshot për Kapitullin IV të tezës.

**10. Vlerëso asistentin.**

```
python src\evaluate.py
```

Ekzekuton 25 pyetjet dhe ruan `eval\rezultatet.csv`. Hape në Excel dhe plotëso dy kolonat `vleresimi` (e saktë / pjesërisht e saktë / e pasaktë) dhe `burimi_mbeshtet` (po / jo). Nga këto del tabela e Kapitullit V.

## Rrugët e përdorimit sipas profilit

Pas hyrjes, asistenti i përshtatet llojit të studentit:

| Lloji | Profili | Rrugët |
|---|---|---|
| Student aktual | fakulteti, departamenti, programi, specializimi, viti, semestri | **Informacion akademik** (me butonin "Lëndët e mia", që gjen lëndët e vitit dhe semestrit tënd) dhe **Asistent studimi** |
| Student potencial | arsimi, interesat, fusha, niveli, objektivat | **Orientim për programet**, me motorin e rekomandimeve ("Gjej programet për profilin tim") |

Të gjitha rrugët përfundojnë te RAG-u i njëjtë (`rag.py`). Për studentin aktual kërkimi kufizohet te dokumentet e programit të tij; profili i shtohet pyetjes dhe promptit. Llogaritë ruhen lokalisht te `users/users.json` (fjalëkalimet si hash scrypt). Është ruajtje për demonstrim, jo autentikim për prodhim.

Testet (pa API): `pip install pytest` dhe pastaj `python -m pytest tests`.

## Si shtohen dokumente të reja

Vendos PDF-të e reja te `data\` dhe ekzekuto përsëri `python src\build_index.py`. Emri i programit nxirret nga emri i skedarit (shih `PROGRAM_RULES` te `extract_and_chunk.py`). Dokumentet me emër të panjohur përdorin emrin e skedarit si etiketë.

## Kufizime të njohura (për seksionin e kufizimeve në tezë)

- **Tabelat me profile.** Te PDF-të e Informatikës dhe Teknologjisë së Informacionit, emrat e profileve shfaqen në tekstin e nxjerrë larg lëndëve të tyre. Pyetja 15 e setit testues e provon këtë. Zgjidhja: shto te `data\` një skedar `.txt` me tekstin e korrigjuar (p.sh. "Profili Siguria e sistemeve të informacionit përfshin lëndët: ..."). Skedarët `.txt` indeksohen njësoj si PDF-të.
- **Pa kujtesë bisede.** Çdo pyetje trajtohet më vete. Një pyetje vijuese si "po te Elektrika?" nuk e di temën e pyetjes së mëparshme.
- **Varësia nga OpenAI.** Pyetja dhe fragmentet e gjetura dërgohen te ofruesi i API. Mos shkruaj të dhëna personale.
- **Dokumente statike.** Asistenti nuk di tarifa, afate apo orare që nuk janë në dokumente. Në këto raste duhet të thotë që nuk e gjen informacionin.
- **Cilësia në shqip** nuk është vërtetuar paraprakisht. Rezultatet e `evaluate.py` e masin këtë.

## Probleme të zakonshme

| Çfarë sheh | Çfarë ndodh dhe zgjidhja |
|---|---|
| `python is not recognized` | Python nuk është në PATH. Mbyll dhe rihap VS Code, ose riinstalo Python me "Add python.exe to PATH". |
| `running scripts is disabled` | Shih hapin 3 (Set-ExecutionPolicy). |
| `ModuleNotFoundError: No module named 'openai'` | Mjedisi virtual nuk është aktiv. Ekzekuto `venv\Scripts\activate` dhe duhet të shohësh `(venv)`. |
| `Mungon çelësi OPENAI_API_KEY` | Skedari `.env` nuk ekziston, ose ka ende vlerën shembull. Shih hapin 5. |
| `AuthenticationError` ose 401 | Çelësi është i gabuar ose i fshirë. Krijo një të ri. |
| `insufficient_quota` ose 429 | Nuk ka kredit. Shto kredit te Billing në platform.openai.com. |
| `database is locked` ose gabime ChromaDB | Projekti është në OneDrive. Zhvendose te `C:\Projekte`, fshi dosjen `chroma_db` dhe ekzekuto `build_index.py`. |
| `pip install chromadb` dështon me Python 3.14 | Instalo Python 3.12 dhe krijo mjedisin me `py -3.12 -m venv venv`. |
| `ModuleNotFoundError: rag` te Streamlit | Ekzekutoje nga dosja kryesore: `streamlit run src\app.py`. |
| Shkronjat ë, ç dalin gabim | Përdor terminalin e VS Code. Në Command Prompt ekzekuto më parë `chcp 65001`. |

## Lidhja me tezën

| Pjesa e tezës | Ku zbatohet në kod |
|---|---|
| 3.5 Teknologjitë | `requirements.txt`, `config.py` |
| 3.6 Kërkesa 1-2 (pyetje në shqip, kërkim semantik) | `rag.py` → `retrieve()` |
| 3.6 Kërkesa 3 (përgjigje vetëm nga konteksti) | `rag.py` → `SYSTEM_PROMPT`, rregulli 1 |
| 3.6 Kërkesa 4 (burimi te çdo përgjigje) | `app.py` → `show_sources()` |
| 3.6 Kërkesa 5 (pyetje jashtë korpusit) | `rag.py` → `REFUSAL_PREFIX`, rregulli 4 |
| 3.6 Kërkesa 6 (shtimi i dokumenteve) | `build_index.py`, dosja `data\` |
| 3.8 Set pyetjesh testuese | `eval\test_questions.csv`, `evaluate.py` |
| Kapitulli IV, Faza A | `extract_and_chunk.py`, `build_index.py` |
| Kapitulli IV, Faza B | `rag.py`, `query.py`, `app.py` |
