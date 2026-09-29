import os
import re

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="AI Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)

GEMINI_MODEL = "gemini-3.8-flash"  

_configured = False


def ensure_configured() -> None:
    """Lazily configure the Gemini client so the app can still start (and /health
    can still respond) even before GEMINI_API_KEY is set."""
    global _configured
    if not _configured:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        _configured = True


class SummarizeRequest(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content must not be empty")
        return v.strip()


class SummarizeResponse(BaseModel):
    summary: str
    source_type: str  


def fetch_article_text(url: str) -> str:
    """Fetch a URL and pull out the readable text from the page."""
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SummarizerBot/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    
    container = soup.find("article") or soup.find("body")
    if container is None:
        raise HTTPException(status_code=400, detail="Could not find readable content on that page.")

    text = " ".join(container.get_text(separator=" ").split())

    if len(text) < 200:
        raise HTTPException(status_code=400, detail="That page didn't have enough readable text to summarize.")

    
    return text[:15000]


def summarize_text(text: str) -> str:
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Server is missing GEMINI_API_KEY. Set it in backend/.env.",
        )

    ensure_configured()
    model = genai.GenerativeModel(GEMINI_MODEL)

    try:
        response = model.generate_content(
            "Summarize the following content into 4-6 concise bullet points. "
            "Capture only the key ideas, no preamble, no closing remarks.\n\n"
            f"{text}"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    if not response.text:
        raise HTTPException(status_code=502, detail="Gemini returned an empty response.")

    return response.text.strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest):
    content = payload.content

    if URL_PATTERN.match(content):
        article_text = fetch_article_text(content)
        summary = summarize_text(article_text)
        return SummarizeResponse(summary=summary, source_type="url")

    summary = summarize_text(content)
    return SummarizeResponse(summary=summary, source_type="text")
