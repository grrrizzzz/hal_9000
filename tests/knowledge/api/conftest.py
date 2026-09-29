"""An API over the synthetic answer-test library, in fake mode, with a signed-in client."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hal9000.knowledge.answer.fake import fake_deps
from hal9000.knowledge.api import create_app
from hal9000.knowledge.api.seed import seed_library
from hal9000.knowledge.api.services import Services, build_services
from hal9000.knowledge.config import load_knowledge_file
from hal9000.knowledge.index.fake import FAKE_ENV, FakeEmbedder, enable_fake_mode
from hal9000.knowledge.ingest.jobs import ProgressBus
from hal9000.knowledge.ingest.pipeline import IngestPipeline
from tests.knowledge.answer.conftest import CORPUS, make_settings

PASSCODE = "open-the-pod-bay-doors"
#: what a browser sends from the app's own pages (the request guard's CSRF check)
SAME_ORIGIN = {"sec-fetch-site": "same-origin"}


def client_for(app) -> TestClient:
    return TestClient(app, headers=SAME_ORIGIN)


@dataclass
class Api:
    client: TestClient
    services: Services
    bus: ProgressBus

    def sign_in(self, client: TestClient | None = None) -> TestClient:
        target = client or self.client
        response = target.post("/api/session", json={"passcode": PASSCODE})
        assert response.status_code == 200, response.text
        return target

    def run_jobs(self) -> list:
        conn = self.services.connect()
        try:
            return self.services.worker.run_pending(conn)
        finally:
            conn.close()

    def conn(self):
        return self.services.connect()


@pytest.fixture(autouse=True)
def fake_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(FAKE_ENV, "1")
    yield
    enable_fake_mode(False)  # build_services(fake=True) turns it on for the process


def make_api(tmp_path: Path, *, seed: bool = True, chat_limit: int | None = None) -> Api:
    settings = make_settings(tmp_path)
    bus = ProgressBus()
    ingest = IngestPipeline(
        settings.files_dir,
        embed_fn=FakeEmbedder(),
        use_llm=False,
        bus=bus,
        isolate_parse=False,
        knowledge_file=load_knowledge_file(settings.yaml_path),
    )
    services = build_services(
        settings,
        fake=True,
        start_worker=False,
        answer_deps=fake_deps(delay_s=0),
        ingest=ingest,
        chat_limit=chat_limit,
    )
    services.bus = bus
    if seed:
        conn = services.connect()
        try:
            seed_library(conn, ingest, CORPUS)
        finally:
            conn.close()
    return Api(client_for(create_app(services, web_dist=False)), services, bus)


@pytest.fixture
def api(tmp_path: Path) -> Iterator[Api]:
    made = make_api(tmp_path)
    yield made
    made.client.close()
    made.services.close()


@pytest.fixture
def signed(api: Api) -> TestClient:
    return api.sign_in()
