from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import pdfplumber
import httpx
import re
import io
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = "claude-sonnet-4-20250514"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _chiedi_claude(system: str, prompt: str, max_tokens: int = 1500) -> str:
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ------------------------------------------------------------------ #
# 📌 /ai_personalizzato
# ------------------------------------------------------------------ #
class UserData(BaseModel):
    categoria: str
    eta: Optional[Any] = None
    reddito: Optional[Any] = None
    classe_bonus_malus: Optional[Any] = None
    numero_sinistri: Optional[Any] = None
    fumatore: Optional[Any] = None
    reddito_annuale: Optional[Any] = None
    durata_contratto: Optional[Any] = None
    destinazione: Optional[Any] = None
    durata_giorni: Optional[Any] = None
    eta_viaggiatore: Optional[Any] = None
    metri_quadri: Optional[Any] = None
    valore_immobile: Optional[Any] = None
    allarme_presente: Optional[Any] = None
    condizioni_preesistenti: Optional[Any] = None
    tipo_lavoro: Optional[Any] = None
    numero_figli: Optional[Any] = None
    contributi_versati: Optional[Any] = None
    profilo_di_rischio: Optional[Any] = None
    importo_da_investire: Optional[Any] = None
    capitale_gia_versato: Optional[Any] = None
    eta_pensionamento: Optional[Any] = None
    tfr: Optional[Any] = None
    tipo_veicolo: Optional[Any] = None
    valore_veicolo: Optional[Any] = None
    km_annui_percorsi: Optional[Any] = None
    zona_di_residenza: Optional[Any] = None
    anno_immatricolazione: Optional[Any] = None
    tipologia_immobile: Optional[Any] = None
    zona_geografica: Optional[Any] = None
    anno_di_costruzione: Optional[Any] = None
    inquilini: Optional[Any] = None
    hai_una_polizza_sanitaria: Optional[Any] = None
    frequenza_visite_mediche: Optional[Any] = None
    farmaci_abituali: Optional[Any] = None
    copertura_desiderata: Optional[Any] = None
    stato_di_salute_generale: Optional[Any] = None
    attivita_sportive_praticate: Optional[Any] = None
    capitale_da_assicurare: Optional[Any] = None
    orizzonte_temporale: Optional[Any] = None
    obiettivo: Optional[Any] = None
    hai_gia_altri_investimenti: Optional[Any] = None
    tipo_contratto_lavorativo: Optional[Any] = None
    fondo_pensione_attuale: Optional[Any] = None
    versamento_mensile_desiderato: Optional[Any] = None
    attivita_sportive: Optional[Any] = None
    massimale_desiderato: Optional[Any] = None
    tipo_viaggio: Optional[Any] = None
    attivita_rischiose_previste: Optional[Any] = None
    numero_di_viaggiatori: Optional[Any] = None
    richiesta_approfondimento: Optional[bool] = None
    includi_stima_prezzo: Optional[bool] = None
    approfondimento: Optional[str] = None
    richiesta: Optional[str] = None
    documento_riferimento: Optional[str] = None
    domanda: Optional[str] = None
    testo_documento: Optional[str] = None


@app.post("/ai_personalizzato")
async def ai_personalizzato(data: UserData):

    # Caso: approfondimento termine
    if data.approfondimento and data.richiesta:
        try:
            return {"valutazione": _chiedi_claude(
                system="Sei un consulente assicurativo esperto che spiega concetti in modo semplice e concreto.",
                prompt=data.richiesta, max_tokens=800)}
        except Exception as e:
            return {"errore": str(e)}

    # Caso: analisi profilo utente
    campi = data.model_dump(exclude={
        "categoria", "approfondimento", "richiesta", "documento_riferimento",
        "domanda", "testo_documento", "richiesta_approfondimento", "includi_stima_prezzo"
    })
    righe = [
        f"- {k.replace('_', ' ').capitalize()}: {v}"
        for k, v in campi.items()
        if v not in (None, "", "0", 0, False)
    ]
    dettagli = "\n".join(righe) if righe else "Nessun dettaglio fornito."

    approfondimento = data.richiesta_approfondimento or False
    stima_prezzo = data.includi_stima_prezzo or approfondimento

    istruzioni_prezzo = ""
    if stima_prezzo:
        istruzioni_prezzo = """

SEZIONE OBBLIGATORIA — aggiungi sempre alla fine:

💰 STIMA INDICATIVA DEL PREMIO
- Fornisci un range di prezzo realistico per il mercato italiano (es. €300-600/anno)
- Scrivi che è una stima indicativa e il prezzo finale dipende dalla compagnia scelta
- Elenca 2-4 consigli PRATICI e SPECIFICI per abbassare il premio, ad esempio:
  * Polizza Casa: "Installare un sistema d'allarme certificato può ridurre il premio del 10-20%"
  * Polizza Auto: "Passare alla classe 13 ridurrebbe il premio di circa €X"  
  * Polizza Vita: "Smettere di fumare abbassa il premio del 20-30% dopo 12 mesi"
  * Polizza Salute: "Scegliere una franchigia più alta riduce il premio annuo"
- Se mancano dati per la stima, indica ESATTAMENTE quali dati servirebbero"""

    # Avviso speciale per auto vecchie
    avviso_auto = ""
    if data.categoria == "Auto" and data.anno_immatricolazione:
        try:
            anno = int(str(data.anno_immatricolazione))
            eta_veicolo = 2025 - anno
            if eta_veicolo > 15:
                avviso_auto = f"\nATTENZIONE: il veicolo ha {eta_veicolo} anni. Per veicoli molto datati, coperture come Furto e Incendio o Kasko raramente convengono economicamente (il valore del veicolo è probabilmente inferiore al costo delle coperture). Segnalalo chiaramente all'utente."
        except Exception:
            pass

    prompt = f"""L'utente cerca una polizza assicurativa di tipo: {data.categoria}.

Dati forniti:
{dettagli}
{avviso_auto}

{"Analisi APPROFONDITA richiesta:" if approfondimento else "Analisi richiesta:"}
1. Coperture consigliate e motivazione
2. Punti di attenzione specifici per questo profilo
3. {"Confronto tra 2 tipologie di prodotto" if approfondimento else "Suggerimenti pratici"}
4. Spiegazione semplice di ogni termine tecnico
{istruzioni_prezzo}

Usa linguaggio semplice. Non terminare con frasi vaghe come "contatta un consulente" — 
se servono più dati, specifica ESATTAMENTE quali e PERCHÉ influenzerebbero la risposta."""

    try:
        return {"valutazione": _chiedi_claude(
            system="Sei un consulente assicurativo italiano esperto, pratico e diretto. Fornisci sempre stime di prezzo concrete quando richiesto.",
            prompt=prompt, max_tokens=1500)}
    except Exception as e:
        return {"errore": str(e)}


# ------------------------------------------------------------------ #
# ❓ /ai_domanda_documento — domanda contestuale su documento
# ------------------------------------------------------------------ #
class DomandaDoc(BaseModel):
    domanda: str
    testo_documento: str


@app.post("/ai_domanda_documento")
async def ai_domanda_documento(data: DomandaDoc):
    if not data.testo_documento.strip() or not data.domanda.strip():
        return {"errore": "Mancano testo documento o domanda."}

    prompt = f"""Hai davanti il testo completo di un documento assicurativo:

--- INIZIO DOCUMENTO ---
{data.testo_documento[:6000]}
--- FINE DOCUMENTO ---

L'utente fa questa domanda specifica:
"{data.domanda}"

Istruzioni:
- Rispondi SOLO basandoti sul contenuto del documento sopra
- Se l'informazione è presente nel documento, cita la parte rilevante
- Se l'informazione NON è nel documento, dillo chiaramente: "Questa informazione non è presente nel documento analizzato"
- Usa linguaggio semplice e comprensibile
- Sii preciso e conciso"""

    try:
        return {"valutazione": _chiedi_claude(
            system="Sei un consulente assicurativo che risponde a domande precise su documenti assicurativi specifici.",
            prompt=prompt, max_tokens=800)}
    except Exception as e:
        return {"errore": str(e)}


# ------------------------------------------------------------------ #
# 📎 /ai_pdf
# ------------------------------------------------------------------ #
@app.post("/ai_pdf")
async def ai_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            return {"errore": "Il file ricevuto è vuoto."}

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            testo = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())

        if not testo.strip():
            return {"errore": "Nessun testo rilevato nel PDF."}

        prompt = (
            "Analizza questo documento assicurativo e fornisci:\n"
            "1. Caratteristiche principali del prodotto\n"
            "2. Vantaggi e punti di forza\n"
            "3. Criticità e limitazioni\n"
            "4. A chi è adatto questo prodotto\n"
            "5. Punteggio da 1 a 5 con motivazione\n"
            "6. 💰 Stima indicativa del premio/costo se deducibile dal documento, "
            "con consigli per abbassarlo\n\n"
            f"{testo[:5000]}"
        )

        valutazione = _chiedi_claude(
            system="Sei un consulente assicurativo esperto.",
            prompt=prompt)
        return {"valutazione": valutazione, "testo_documento": testo[:6000]}
    except Exception as e:
        return {"errore": str(e)}


# ------------------------------------------------------------------ #
# 🌐 /ai_url
# ------------------------------------------------------------------ #
class URLData(BaseModel):
    url: str


@app.post("/ai_url")
async def ai_url(data: URLData):
    url = data.url.strip()
    if not url.startswith("http"):
        return {"errore": "URL non valido."}

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as hc:
            risposta = await hc.get(url)

        if risposta.status_code != 200:
            return {"errore": f"Impossibile scaricare il documento (HTTP {risposta.status_code})."}

        content_type = risposta.headers.get("content-type", "")
        raw = risposta.content

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                testo = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
        elif "html" in content_type:
            testo = re.sub(r"<[^>]+>", " ", risposta.text)
            testo = re.sub(r"\s+", " ", testo).strip()
        else:
            testo = risposta.text

        if not testo.strip():
            return {"errore": "Nessun testo estraibile dal documento."}

        prompt = (
            f"Analizza questo documento assicurativo (URL: {url}):\n\n"
            "1. Caratteristiche principali\n"
            "2. Vantaggi e punti di forza\n"
            "3. Criticità e limitazioni\n"
            "4. A chi è adatto\n"
            "5. Punteggio da 1 a 5 con motivazione\n"
            "6. 💰 Stima indicativa del premio/costo se deducibile, "
            "con consigli per abbassarlo\n\n"
            f"Documento:\n{testo[:5000]}"
        )

        valutazione = _chiedi_claude(
            system="Sei un consulente assicurativo esperto.",
            prompt=prompt)
        return {"valutazione": valutazione, "testo_documento": testo[:6000]}
    except httpx.TimeoutException:
        return {"errore": "Timeout: il server non ha risposto in tempo."}
    except Exception as e:
        return {"errore": str(e)}
