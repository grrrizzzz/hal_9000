#!/usr/bin/env python3
"""Fallback only: (re)download the vendored tiktoken encodings into ``vendor/tiktoken/``.

The encodings are already committed (KWP0), so this is normally never needed.
It exists for a host that must refresh them, e.g. if tiktoken ever changes an
encoding's expected hash. Run it where openaipublic.blob.core.windows.net is
reachable (the Mac), then commit the files:

    uv run python scripts/kb_fetch_tiktoken.py

Each file is checked against the sha256 tiktoken itself expects and saved
under tiktoken's cache name, ``sha1(url)``, which is what ``TIKTOKEN_CACHE_DIR``
lookups use.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "tiktoken"
BASE = "https://openaipublic.blob.core.windows.net/encodings/"
#: The hashes in tiktoken_ext/openai_public.py.
EXPECTED_SHA256 = {
    "cl100k_base": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    "o200k_base": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
}


def cache_name(encoding: str) -> str:
    return hashlib.sha1(f"{BASE}{encoding}.tiktoken".encode()).hexdigest()


def main() -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)
    for encoding, expected in EXPECTED_SHA256.items():
        url = f"{BASE}{encoding}.tiktoken"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (fixed https URL)
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            print(f"FAIL: {encoding} sha256 {digest} != expected {expected}", file=sys.stderr)
            return 1
        target = VENDOR / cache_name(encoding)
        target.write_bytes(data)
        print(f"{encoding}: {len(data):,} bytes -> {target.relative_to(VENDOR.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
