"""HAL 9000's house knowledge base (docs/KNOWLEDGE_PLAN.md).

Importing this package must stay cheap and offline:

- it never imports Docling, fastembed or PyTorch (those live in optional ``uv``
  groups and are imported lazily by the modules that need them; a test checks
  ``sys.modules``);
- it points ``tiktoken`` at the encodings vendored in ``vendor/tiktoken/``, so
  counting tokens never downloads anything (plan §21). An existing
  ``TIKTOKEN_CACHE_DIR`` wins, which is how the container image supplies its
  baked-in copy.

This must run before the first ``import tiktoken`` anywhere, so every knowledge
module that counts tokens imports this package first (Python does that for any
``hal9000.knowledge.*`` import).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["TIKTOKEN_VENDOR_DIR", "configure_tiktoken"]

#: ``<repo>/vendor/tiktoken``: files named ``sha1(blob URL)``, as tiktoken caches them.
TIKTOKEN_VENDOR_DIR = Path(__file__).resolve().parents[3] / "vendor" / "tiktoken"


def configure_tiktoken() -> str | None:
    """Set ``TIKTOKEN_CACHE_DIR`` to the vendored encodings unless it is already set.

    Returns the directory tiktoken will use, or ``None`` when neither is available.
    """
    current = os.environ.get("TIKTOKEN_CACHE_DIR", "").strip()
    if current:
        return current
    if TIKTOKEN_VENDOR_DIR.is_dir():
        os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_VENDOR_DIR)
        return str(TIKTOKEN_VENDOR_DIR)
    return None


configure_tiktoken()
