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

## Këshilltari akademik i personalizuar

Mbi RAG-un ekzistues, `src/advisor.py` shton një këshilltar akademik:

- **Profili i studentit** (`src/student_profile.py`) plotësohet vetë nga biseda dhe përditësohet gjatë saj. Mund të ndryshohet edhe manualisht te shiriti anësor. Ruhet vetëm te sesioni i shfletuesit.
- **Qëllimi i mesazhit** (`src/intent.py`) klasifikohet automatikisht. Pyetjet faktike (tarifa, kalendar, informacion programi) kalojnë te RAG standard pa ndryshim. Kërkesat e personalizuara (rekomandim, rrugë akademike, boshllëqe aftësish, krahasim programesh, "po sikur") marrin përgjigje të strukturuar me profilin.
- **Pyetje ndjekëse** bëhen vetëm kur mungon informacion i nevojshëm, deri në 2 radhazi.
- **Burimet** shfaqen me dokumentin dhe faqen, vetëm për fragmentet e cituara. Faqet shfaqen pasi të ekzekutosh përsëri `python src\build_index.py`.

**Kufizim i njohur.** Analiza e boshllëqeve të aftësive kërkon rezultate të pritura të të nxënit ose nivele të kërkuara për lëndët. Dokumentet aktuale kanë vetëm objektiva, plane mësimore (lëndë dhe ECTS) dhe profile, prandaj sistemi tregon cilat lëndë mbulojnë një aftësi dhe e thotë hapur kur niveli i kërkuar nuk specifikohet. Për analizë më të saktë duhen shtuar te `data/` përshkrimet e lëndëve me rezultatet e të nxënit.

## Llogaritë dhe dy rrugët e studentit

- **Hyrja:** `src/accounts.py` ruan llogaritë te SQLite (`users.sqlite3`, i injoruar nga Git). Fjalëkalimet ruhen vetëm si hash PBKDF2-SHA256 me kripë. Përdoret pseudonim, jo emri real. Biseda nuk ruhet, vetëm roli, profili dhe të dhënat akademike. Pa hyrje nuk ngarkohet baza e njohurive.
- **Student aktual:** profili akademik (fakulteti, departamenti, programi, specializimi, viti, semestri) → përputhja me planin mësimor → lëndët e semestrit → informacion akademik ose ndihmë për studimin, me RAG dhe burime.
- **Student potencial:** profili (arsimi, interesat, aftësitë, niveli, objektivat) → motori i rekomandimit → orientim për programet, me RAG dhe burime.
- **Plani mësimor i strukturuar:** `python src/build_curriculum.py` nxjerr lëndët (viti, semestri, ECTS, faqja) nga tabelat e PDF-ve te `data/curriculum.json` dhe printon paralajmërimet për verifikim. Ekzekutoje sa herë ndryshojnë planet.
- **Kufizime:** dokumentet nuk përmbajnë fakultetin, departamentin, përmbajtjen e lëndëve ose materialet mësimore. Për këtë arsye fakulteti dhe departamenti plotësohen nga studenti, dhe ndihma për studimin kufizohet te lista e lëndëve, ECTS dhe këshilla të përgjithshme të shënuara si të tilla. Te Streamlit Community Cloud skedari SQLite fshihet kur serveri rinis, kështu që llogaritë nuk janë të qëndrueshme aty pa një bazë të jashtme.

## Asistenti i studentit aktual dhe Study Mode

- **Lëndët identifikohen vetë** nga plani zyrtar (`data/curriculum.json`): programi, specializimi, viti dhe semestri. Specializimet nxirren nga titujt e grupeve C te tabela (p.sh. "IT e biznesit"). Për programet ku tabela nuk i lidh lëndët me specializim, shfaqen vetëm lëndët e përbashkëta.
- **Study Mode:** studenti zgjedh një lëndë të semestrit dhe mund të kërkojë shpjegim, pyetje, quiz, ushtrime, flashcards, provim prove, plan përsëritjeje ose përmbledhje. Kërkimi RAG merr fillimisht materialet e lëndës (`data/lendet/`), pastaj dokumentet zyrtare të programit.
- **Materialet e lëndëve** nuk ekzistojnë ende te projekti. Shih `data/lendet/README.md` për mënyrën e shtimit. Pa materiale, sistemi thotë hapur që nuk i ka dhe përdor vetëm faktet e planit. Kur studenti emërton një temë, jep shpjegim të përgjithshëm të shënuar qartë si jo nga materialet e lëndës. Studenti mund të ngjitë edhe materialin e vet.
- **Konteksti i bisedës:** pyetja rishkruhet si pyetje e plotë me historinë ("Po REST?" pas "Çfarë është API?") para kërkimit.
- **Studenti aktual nuk merr rekomandime për programe të tjera**, përveç kur i kërkon qartë. Kërkimi i tij kufizohet te programi i vet.
- **Moduli:** `src/student_assistant.py`, i pavarur nga rruga e studentit potencial (`src/advisor.py`).
