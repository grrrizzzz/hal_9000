"""Token counting with the vendored ``tiktoken`` encodings (plan §4, §21).

Chunk sizes, context budgets and embedding batches are all measured in model
tokens. ``text-embedding-3-*`` uses ``cl100k_base``; the GPT-4o/GPT-5 chat
families use ``o200k_base``. Both encodings are vendored, so nothing here
downloads (``hal9000.knowledge`` sets ``TIKTOKEN_CACHE_DIR`` on import).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import hal9000.knowledge  # noqa: F401  (sets TIKTOKEN_CACHE_DIR before tiktoken loads)

if TYPE_CHECKING:  # pragma: no cover
    import tiktoken

__all__ = ["EMBEDDING_ENCODING", "CHAT_ENCODING", "count_tokens", "encoding", "encoding_for_model"]

EMBEDDING_ENCODING = "cl100k_base"
CHAT_ENCODING = "o200k_base"


@lru_cache(maxsize=4)
def encoding(name: str = EMBEDDING_ENCODING) -> tiktoken.Encoding:
    import tiktoken

    return tiktoken.get_encoding(name)


def encoding_for_model(model: str) -> str:
    """The encoding name for a LiteLLM model string (``openai/text-embedding-3-small``).

    ``fake/*`` models (the offline test embedder) count like the embedding models.
    """
    if model.startswith("fake/"):
        return EMBEDDING_ENCODING
    name = model.split("/", 1)[-1]
    legacy_gpt4 = name.startswith("gpt-4") and not name.startswith(("gpt-4o", "gpt-4.1"))
    if name.startswith(("text-embedding-", "gpt-3.5")) or legacy_gpt4:
        return EMBEDDING_ENCODING
    return CHAT_ENCODING


def count_tokens(text: str, encoding_name: str = EMBEDDING_ENCODING) -> int:
    """Number of tokens in ``text``. Special-token text is counted as ordinary text."""
    return len(encoding(encoding_name).encode(text, disallowed_special=()))
