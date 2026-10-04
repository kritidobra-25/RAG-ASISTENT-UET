"""
Të gjitha thirrjet drejt OpenAI API janë këtu, në një vend të vetëm.
Kështu, nëse ndryshon ofruesi ose modeli, ndryshon vetëm ky skedar.
"""

import os
from functools import lru_cache

from openai import OpenAI

import config


class MissingApiKeyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-xxxx"):
        raise MissingApiKeyError(
            "Mungon çelësi OPENAI_API_KEY. Hap skedarin .env në dosjen kryesore "
            "dhe shkruaj çelësin tënd real, p.sh. OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Kthen një vektor (embedding) për çdo tekst në listë."""
    response = _client().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    """Dërgon një pyetje te modeli gjuhësor dhe kthen tekstin e përgjigjes."""
    response = _client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=config.TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()
