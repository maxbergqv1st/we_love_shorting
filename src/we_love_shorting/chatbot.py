"""Chatbot: explain the model's predictions/evaluation in plain language via
OpenRouter (free tier). API key is injected by the caller (app.py reads it
from st.secrets), so this module stays pure and testable — no Streamlit import
here."""

import json
import logging
import urllib.error
import urllib.request

from .retry import retry

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODEL = "openai/gpt-5.6-luna"  # tried if MODEL fails after retries

SYSTEM_PROMPT = (
    "Du är en assistent som förklarar resultat från en ML-modell som "
    "förutsäger aktie-/råvarupriser och nyhetston. Svara kort och konkret "
    "på svenska, baserat på kontexten nedan när frågan rör projektet. Om "
    "frågan inte handlar om projektet: du får ändå svara, men inled då "
    "svaret med en lekfull kommentar om att du bara svarar för att "
    "utvecklarna tillåtit det (t.ex. 'Utanför ämnet, men eftersom "
    "utvecklarna gav mig lov...').\n\n"
)


class BadResponse(Exception):
    """OpenRouter returned 200 OK but no usable completion (seen on the free
    tier when the upstream provider fails silently, e.g. an empty or missing
    `choices` list)."""


@retry(times=3, exceptions=(urllib.error.HTTPError, BadResponse))
def _post(api_key: str, model: str, messages: list[dict]) -> dict:
    body = json.dumps({"model": model, "messages": messages}).encode()
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:  # free tier can be slow
        data = json.load(r)
    if not data.get("choices"):
        raise BadResponse(data.get("error", data))
    return data


def ask(api_key: str, context: str, question: str) -> str:
    """Answer `question` about the model/data, grounded in `context`.

    Tries MODEL first; if it still fails after its own retries (e.g. the free
    tier is overloaded or silently returns no choices), falls back to
    FALLBACK_MODEL once.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + context},
        {"role": "user", "content": question},
    ]
    try:
        data = _post(api_key, MODEL, messages)
    except (urllib.error.HTTPError, BadResponse) as e:
        log.warning(
            "primary model %s failed (%s), falling back to %s", MODEL, e, FALLBACK_MODEL
        )
        data = _post(api_key, FALLBACK_MODEL, messages)
    return data["choices"][0]["message"]["content"]
