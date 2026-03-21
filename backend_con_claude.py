from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import pdfplumber
import httpx
import io
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ✅ Client Anthropic (sostituisce OpenAI)
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


# ------------------------------------------------------------------ #
# 📌 Modello dati utente — ora include TUTTI i campi del Flutter
# (fix: prima molti campi venivano ignorati silenziosamente)
# ------------------------------------------------------------------ #
class UserData(BaseModel):
    categoria: str
    # Campi comuni
    eta: Optional[Any] = None
    reddito: Optional[Any] = None
    # Auto
    classe_bonus_malus: Optional[Any] = None
    numero_sinistri: Optional[Any] = None
    # Vita / Prodotti Vita
    fumatore: Optional[Any] = None
    reddito_annuale: Optional[Any] = None
    durata_contratto: Optional[Any] = None
    # Viaggio
    destinazione: Optional[Any] = None
    durata_giorni: Optional[Any] = None
    eta_viaggiatore: Optional[Any] = None
    # Casa
    metri_quadri: Optional[Any] = None
    valore_immobile: Optional[Any] = None
    allarme_presente: Optional[Any] = None
    # Salute
    condizioni_preesistenti: Optional[Any] = None
    # Infortuni
    tipo_lavoro: Optional[Any] = None
    # Caso Morte
    numero_figli: Optional[Any] = None
    # Pensionistici
    contributi_versati: Optional[Any] = None
    # Investimento
    profilo_di_rischio: Optional[Any] = None
    importo_da_investire: Optional[Any] = None


def _chiedi_claude(system: str, prompt: str) -> str:
    """Helper unico per chiamare Claude — sostituisce openai.ChatCompletion.create"""
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ------------------------------------------------------------------ #
# 🤖 /ai_personalizzato — analisi profilo utente
# ------------------------------------------------------------------ #
@app.post("/ai_personalizzato")
async def ai_personalizzato(data: UserData):
    # Costruiamo il riepilogo solo con i campi effettivamente compilati
    campi = data.model_dump(exclude={"categoria"})
    righe = [
        f"- {chiave.replace('_', ' ').capitalize()}: {valore}"
        for chiave, valore in campi.items()
        if valore not in (None, "", "0", 0)
    ]
    dettagli = "\n".join(righe) if righe else "Nessun dettaglio fornito."

    prompt = f"""L'utente cerca una polizza assicurativa di tipo: {data.categoria}.

Dati forniti:
{dettagli}

Suggerisci una tipologia di copertura adatta e motiva brevemente la scelta."""

    try:
        valutazione = _chiedi_claude(
            system="Sei un consulente assicurativo esperto e obiettivo.",
            prompt=prompt,
        )
        return {"valutazione": valutazione}
    except Exception as e:
        return {"errore": f"Errore AI: {str(e)}"}


# ------------------------------------------------------------------ #
# 📎 /ai_pdf — analisi di un PDF caricato localmente
# ------------------------------------------------------------------ #
@app.post("/ai_pdf")
async def ai_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()

        if not content:
            return {"errore": "Il file ricevuto è vuoto."}

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            testo = "\n".join(
                pagina.extract_text()
                for pagina in pdf.pages
                if pagina.extract_text()
            )

        if not testo.strip():
            return {"errore": "Nessun testo rilevato nel PDF. Potrebbe essere un PDF scansionato o protetto."}

        prompt = (
            f"Analizza sinteticamente questo prodotto assicurativo. "
            f"Evidenzia: caratteristiche principali, vantaggi, criticità e un punteggio da 1 a 5.\n\n"
            f"{testo[:4000]}"
        )

        valutazione = _chiedi_claude(
            system="Sei un consulente assicurativo esperto.",
            prompt=prompt,
        )
        return {"valutazione": valutazione}

    except Exception as e:
        return {"errore": f"Errore nell'elaborazione del PDF: {str(e)}"}


# ------------------------------------------------------------------ #
# 🌐 /ai_url — analisi di un documento assicurativo via URL
# FIX CRITICO: ora il PDF viene REALMENTE scaricato e letto,
# invece di passare solo l'URL al modello (che non ha internet)
# ------------------------------------------------------------------ #
class URLData(BaseModel):
    url: str


@app.post("/ai_url")
async def ai_url(data: URLData):
    url = data.url.strip()

    if not url.startswith("http"):
        return {"errore": "URL non valido. Deve iniziare con http:// o https://"}

    try:
        # 1. Scarica il contenuto dall'URL
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http_client:
            risposta = await http_client.get(url)

        if risposta.status_code != 200:
            return {"errore": f"Impossibile scaricare il documento (HTTP {risposta.status_code})."}

        content_type = risposta.headers.get("content-type", "")
        raw = risposta.content

        # 2. Estrai testo in base al tipo di contenuto
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                testo = "\n".join(
                    p.extract_text() for p in pdf.pages if p.extract_text()
                )
        elif "html" in content_type:
            import re
            testo = re.sub(r"<[^>]+>", " ", risposta.text)
            testo = re.sub(r"\s+", " ", testo).strip()
        else:
            testo = risposta.text

        if not testo.strip():
            return {"errore": "Nessun testo estraibile dal documento raggiunto all'URL."}

        prompt = (
            f"Analizza il seguente prodotto/documento assicurativo scaricato da: {url}\n\n"
            f"Rispondi con:\n"
            f"- validita_dati: completi / incompleti / non chiari\n"
            f"- caratteristiche_principali\n"
            f"- vantaggi\n"
            f"- criticita\n"
            f"- punteggio: da 1 a 5\n"
            f"- motivazione del punteggio\n"
            f"- consigli\n"
            f"- alert (punti di attenzione)\n\n"
            f"Documento:\n{testo[:5000]}"
        )

        valutazione = _chiedi_claude(
            system="Sei un consulente assicurativo esperto.",
            prompt=prompt,
        )
        return {"valutazione": valutazione}

    except httpx.TimeoutException:
        return {"errore": "Timeout: il server del documento non ha risposto in tempo."}
    except Exception as e:
        return {"errore": f"Errore durante l'elaborazione del link: {str(e)}"}
