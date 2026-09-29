"""``hal9000 kb check``: is this machine ready to run the knowledge base? (plan §17)

Offline checks, always:

- ``knowledge.db`` opens and is migrated to this code's schema version;
- the sqlite-vec extension loads and answers a cosine KNN with a metadata filter;
- FTS5 with the ``porter unicode61 remove_diacritics 2`` tokenizer works;
- the vendored ``tiktoken`` encodings load without the network;
- ``config/knowledge.yaml`` parses, and its active profile name (if any) is valid.

Online checks, **only when ``OPENAI_API_KEY`` is set**: one embedding with the
configured model and one tiny fast-tier chat call, both metered with
``agent="knowledge"`` (a small fraction of a cent). Without a key they are
reported as skipped, which is not a failure.

Prints one line per check and exits 1 if any check failed.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

__all__ = ["run"]


def _db() -> str:
    from hal9000.knowledge.config import settings
    from hal9000.knowledge.storage import SCHEMA_VERSION, open_db, schema_version

    path = settings().db_path
    conn = open_db(path)
    try:
        version = schema_version(conn)
    finally:
        conn.close()
    if version != SCHEMA_VERSION:  # pragma: no cover - open_db migrates or refuses
        raise RuntimeError(f"schema version {version}, expected {SCHEMA_VERSION}")
    return f"{path} at schema version {version}"


def _sqlite_vec() -> str:
    import struct

    from hal9000.knowledge.storage import connect

    conn = connect(":memory:")
    try:
        version = conn.execute("SELECT vec_version()").fetchone()[0]
        conn.execute(
            "CREATE VIRTUAL TABLE v USING vec0(chunk_id integer primary key, "
            "embedding float[2] distance_metric=cosine, doc_type text)"
        )
        for chunk_id, vector, doc_type in ((1, (1.0, 0.0), "a"), (2, (0.0, 1.0), "b")):
            conn.execute(
                "INSERT INTO v VALUES (?, ?, ?)", (chunk_id, struct.pack("2f", *vector), doc_type)
            )
        rows = conn.execute(
            "SELECT chunk_id FROM v WHERE embedding MATCH ? AND k = 2 AND doc_type = 'b'",
            (struct.pack("2f", 1.0, 0.0),),
        ).fetchall()
        if [row[0] for row in rows] != [2]:
            raise RuntimeError(f"filtered KNN returned {rows!r}")
    finally:
        conn.close()
    return f"version {version}; cosine KNN with a metadata filter works"


def _fts5() -> str:
    import sqlite3

    from hal9000.knowledge.storage import FTS_TOKENIZER

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(f"CREATE VIRTUAL TABLE t USING fts5(text, tokenize = '{FTS_TOKENIZER}')")
        conn.execute("INSERT INTO t VALUES ('The relief valves were dripping')")
        hits = conn.execute('SELECT COUNT(*) FROM t WHERE t MATCH \'"valve" OR "drip"\'')
        if hits.fetchone()[0] != 1:
            raise RuntimeError("the porter stemmer did not match 'valves'")
    finally:
        conn.close()
    return f"tokenizer '{FTS_TOKENIZER}' works"


def _tiktoken() -> str:
    from hal9000.knowledge.tokens import CHAT_ENCODING, EMBEDDING_ENCODING, count_tokens

    for name in (EMBEDDING_ENCODING, CHAT_ENCODING):
        if count_tokens("house knowledge", name) < 1:  # pragma: no cover
            raise RuntimeError(f"{name} counted no tokens")
    return f"{EMBEDDING_ENCODING} and {CHAT_ENCODING} load offline"


def _config() -> str:
    from hal9000.knowledge.config import load_knowledge_file
    from hal9000.knowledge.index.profiles import configured_profile

    knowledge = load_knowledge_file()
    profile = configured_profile(knowledge.active_profile)
    return f"active_profile: {profile.name}" if profile else "no active_profile set yet"


def _embed_model() -> str:
    from hal9000.knowledge.config import load_knowledge_file, settings
    from hal9000.knowledge.index.embedding import is_fake_model
    from hal9000.knowledge.index.profiles import configured_profile

    profile = configured_profile(load_knowledge_file().active_profile)
    if profile is not None and not is_fake_model(profile.embed_model):
        return profile.embed_model
    return settings().embed_model


def _meter():
    from hal9000 import config as hal_config
    from hal9000.cost.metering import Meter
    from hal9000.knowledge.config import monthly_caps
    from hal9000.storage.db import Database

    ledger = Database(hal_config.settings().data_dir / "hal9000.db")
    return ledger, Meter(ledger, "kb-check", 0.05, 0.05, monthly_caps=monthly_caps())


def _openai_embed() -> str:
    from hal9000 import llm
    from hal9000.knowledge.config import AGENT

    model = _embed_model()
    ledger, meter = _meter()
    try:
        result = meter.embed(llm.embed, ["hal9000 kb check"], model, agent=AGENT)
    finally:
        ledger.close()
    return f"{model} returned a {len(result.vectors[0])}-dimension vector (${meter.spent:.6f})"


def _openai_chat() -> str:
    from hal9000 import config as hal_config
    from hal9000 import llm
    from hal9000.core.prompt import Prompt
    from hal9000.knowledge.config import AGENT, settings

    tier = settings().chat_tier
    ledger, meter = _meter()
    try:
        reply = meter.wrap(llm.generate_response, agent=AGENT)(
            Prompt(messages=[{"role": "user", "content": "Reply with the word OK."}]), tier
        )
    finally:
        ledger.close()
    return f"{hal_config.model_for(tier)} replied {reply.strip()[:20]!r} (${meter.spent:.6f})"


def run(args: argparse.Namespace) -> int:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("knowledge.db migrated", _db),
        ("sqlite-vec loads", _sqlite_vec),
        ("FTS5 tokenizer", _fts5),
        ("tiktoken offline", _tiktoken),
        ("config/knowledge.yaml", _config),
    ]
    online = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if online:
        checks += [("OpenAI embedding", _openai_embed), ("OpenAI chat", _openai_chat)]
    failures = 0
    for name, check in checks:
        try:
            detail = check()
        except Exception as error:  # every failure is reported, then the next check runs
            failures += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok    {name}: {detail}")
    if not online:
        print(
            "skip  OpenAI embedding and chat: OPENAI_API_KEY is not set (checked only with a key)"
        )
    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    return 1 if failures else 0
