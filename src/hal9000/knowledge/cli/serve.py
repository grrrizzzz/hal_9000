"""``hal9000 kb serve [--host --port] [--fake]``: the knowledge-base web service (plan §15, §17).

Builds the app (:func:`hal9000.knowledge.api.create_app`) over the real,
metered services, starts the one ingest worker, and serves the API, ``/health``,
``/metrics`` and the web UI (``web/dist``, when built) with uvicorn.

``--fake`` is for the UI's end-to-end tests and demos: fake mode is switched
on explicitly, the chat model and embedder are the offline fakes (nothing is
paid, no key is needed), and a temporary data directory is seeded with the
synthetic fixture library (:func:`hal9000.knowledge.api.seed.fake_corpus`)
and evaluation data (:func:`~hal9000.knowledge.api.seed.seed_evaluation`)
before the server starts. The passcode is ``KB_PASSCODE``, or the mock default
``pod-bay-doors`` documented in ``web/README.md``.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

__all__ = ["FAKE_PASSCODE", "FAKE_PROFILE", "fake_settings", "run"]

FAKE_PASSCODE = "pod-bay-doors"
#: Page-sized structure chunks over the fake 64-d embedder: citations land on the right page.
FAKE_PROFILE = "pdfium.structure-64.hdr.fake-64"
#: A second built profile, so the Evaluation page can switch to it.
FAKE_OTHER_PROFILE = "pdfium.structure-64.nohdr.fake-64"


def fake_settings(root: Path):
    """Settings for ``--fake``: a temp data dir and config (policy copied, fake profile active)."""
    from hal9000.knowledge.config import settings as kb_settings

    current = kb_settings()
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    policy = current.policy_path
    if policy.is_file():
        shutil.copy(policy, config / "knowledge_policy.txt")
    (config / "knowledge.yaml").write_text(f"active_profile: {FAKE_PROFILE}\n")
    return replace(
        current,
        data_dir=root / "data",
        config_dir=config,
        passcode=current.passcode or FAKE_PASSCODE,
    )


def _exit_on_sigterm(signum, frame) -> None:  # pragma: no cover - exercised by a real signal
    raise SystemExit(0)


def _temp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="kb-fake-"))


def _fake_services(root: Path):
    from hal9000.knowledge.api.seed import fake_corpus, seed_evaluation, seed_library
    from hal9000.knowledge.api.services import build_services

    settings = fake_settings(root)
    services = build_services(settings, fake=True, start_worker=False)
    conn = services.connect()
    try:
        seed_library(conn, services.ingest, fake_corpus())
        seed_evaluation(
            conn, settings.eval_dir, pipeline=services.ingest, extra_profile=FAKE_OTHER_PROFILE
        )
    finally:
        conn.close()
    if services.worker is not None:
        services.worker.start()
    return services


def run(args: argparse.Namespace) -> int:
    import uvicorn

    from hal9000 import config as hal_config
    from hal9000.knowledge.api import create_app
    from hal9000.knowledge.config import settings as kb_settings

    root: Path | None = None
    try:
        if args.fake:
            root = _temp_root()
            services = _fake_services(root)
        else:
            from hal9000.knowledge.api.services import build_services

            services = build_services(kb_settings())
    except hal_config.ConfigError as error:
        print(f"hal9000 kb serve: {error}", file=sys.stderr)
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
        return 1
    port = args.port or services.settings.port
    mode = " (fake mode: offline fakes, synthetic library)" if args.fake else ""
    print(f"House knowledge base on http://{args.host}:{port}{mode}", flush=True)
    if args.fake and services.settings.passcode == FAKE_PASSCODE:
        print(f"Passcode: {FAKE_PASSCODE} (set KB_PASSCODE to change it)", flush=True)
    # uvicorn re-raises SIGTERM after a graceful shutdown; with Python's default
    # handler the process would die before the cleanup below, so exit instead.
    previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        uvicorn.run(create_app(services), host=args.host, port=port, log_level="info")
    except SystemExit:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous)
        services.close()
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
    return 0
