"""The OpenAI embeddings transport. This is the only kaful module that imports `openai`.

The client is created on first use (reading OPENAI_API_KEY from .env then, without
overriding a variable already set), so building an embedder needs no key or network.
"""

from collections.abc import Sequence

import openai
from dotenv import load_dotenv

from kaful.config import REPO_ROOT, LLMSettings
from kaful.llm.transport import TransportError


class OpenAIEmbeddingTransport:
    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            load_dotenv(REPO_ROOT / ".env", override=False)
            self._client = openai.OpenAI(timeout=self._settings.request_timeout_s,
                                         max_retries=self._settings.sdk_max_retries)
        return self._client

    def embed(self, texts: Sequence[str], model: str, dimensions: int) -> tuple[list[list[float]], int]:
        try:
            response = self.client.embeddings.create(model=model, input=list(texts), dimensions=dimensions)
        except openai.OpenAIError as exc:
            raise TransportError(f"embeddings: {type(exc).__name__}: {exc}") from exc
        rows = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in rows], int(response.usage.prompt_tokens)
