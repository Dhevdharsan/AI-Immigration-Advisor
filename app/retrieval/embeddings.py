"""
Thin wrapper around OpenAI's embeddings API -- turns chunk text (at
ingestion time) or a user's question (at query time) into the vector
pgvector compares against.
"""

from openai import OpenAI

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    if not texts:
        return []
    client = client or OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_text(text: str, client: OpenAI | None = None) -> list[float]:
    return embed_texts([text], client=client)[0]
