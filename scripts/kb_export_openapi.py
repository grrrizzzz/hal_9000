#!/usr/bin/env python3
"""Write the knowledge API's OpenAPI schema to ``web/src/api/openapi.json``.

    uv run python scripts/kb_export_openapi.py           # regenerate
    uv run python scripts/kb_export_openapi.py --check   # exit 1 if the file is stale

The committed file is the contract between the backend and the UI (plan §11);
``tests/knowledge/test_kb_openapi.py`` fails when it drifts from the app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "web" / "src" / "api" / "openapi.json"


def render() -> str:
    """The schema exactly as it is committed: two-space JSON, trailing newline."""
    from hal9000.knowledge.api import create_app

    return json.dumps(create_app().openapi(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    parser.add_argument("--out", type=Path, default=TARGET)
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = args.out.read_text() if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is out of date; run scripts/kb_export_openapi.py", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
