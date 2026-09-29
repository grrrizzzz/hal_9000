"""Static checks of the Pi deployment files (handoff KWP11, plan §15).

``deploy/docker/compose.knowledge.yaml``, ``deploy/docker/Dockerfile.kb`` and
the root ``.dockerignore`` are checked for the plan's structure (services,
memory limits, ports from ``.env``, mounts) and for the properties that keep the
image safe and offline (a non-root user, a healthcheck, pinned images, the
vendored tiktoken encodings, no inline secrets). Nothing here needs Docker.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
DOCKER = REPO / "deploy" / "docker"
COMPOSE = DOCKER / "compose.knowledge.yaml"
DOCKERFILE = DOCKER / "Dockerfile.kb"
DOCKERIGNORE = REPO / ".dockerignore"
GRAFANA = REPO / "deploy" / "grafana"
PROMETHEUS = REPO / "deploy" / "prometheus" / "prometheus.yml"

SERVICES = {"hal-kb", "prometheus", "grafana"}
SECRET_NAME = re.compile(r"PASSWORD|PASSCODE|TOKEN|SECRET|API_KEY|_KEY$", re.I)
INTERPOLATED = re.compile(r"^\$\{[A-Z_][A-Z0-9_]*(:?[-?][^}]*)?\}$")


# ------------------------------------------------------------------ helpers
@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


@pytest.fixture(scope="module")
def services(compose) -> dict:
    return compose["services"]


def _resolve(path: str) -> Path:
    """A compose-relative host path (relative to the compose file's folder)."""
    return (COMPOSE.parent / path).resolve()


def _mounts(service: dict) -> list[tuple[str, str, str]]:
    """``(source, target, mode)`` for each short-syntax volume."""
    out = []
    for entry in service.get("volumes", []):
        parts = entry.split(":")
        source, target = parts[0], parts[1]
        out.append((source, target, parts[2] if len(parts) > 2 else "rw"))
    return out


def _mount_for(service: dict, target: str) -> tuple[str, str]:
    for source, tgt, mode in _mounts(service):
        if tgt == target:
            return source, mode
    raise AssertionError(f"no volume mounted at {target}")


def _instructions(text: str) -> list[tuple[str, str]]:
    """Dockerfile instructions as ``(KEYWORD, rest)``, continuations joined, comments dropped."""
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        if stripped.startswith("#"):
            continue  # a comment inside a continued instruction
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        lines.append(pending + stripped)
        pending = ""
    out = []
    for line in lines:
        keyword, _, rest = line.partition(" ")
        out.append((keyword.upper(), rest.strip()))
    return out


@pytest.fixture(scope="module")
def dockerfile() -> list[tuple[str, str]]:
    return _instructions(DOCKERFILE.read_text())


@pytest.fixture(scope="module")
def runtime_stage(dockerfile) -> list[tuple[str, str]]:
    """The instructions of the last stage (the image that runs)."""
    last_from = max(i for i, (kw, _) in enumerate(dockerfile) if kw == "FROM")
    return dockerfile[last_from:]


def _global_args(dockerfile) -> dict[str, str]:
    args = {}
    for keyword, rest in dockerfile:
        if keyword == "FROM":
            break
        if keyword == "ARG" and "=" in rest:
            name, _, value = rest.partition("=")
            args[name] = value
    return args


# ------------------------------------------------------------------ compose
def test_compose_has_exactly_the_three_services(services):
    assert set(services) == SERVICES  # Home Assistant is never part of this file


@pytest.mark.parametrize(
    ("name", "limit"), [("hal-kb", "2g"), ("prometheus", "512m"), ("grafana", "512m")]
)
def test_memory_limits_follow_the_plan(services, name, limit):
    assert services[name]["mem_limit"] == limit


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_every_service_restarts_and_has_a_healthcheck(services, name):
    service = services[name]
    assert service["restart"] == "unless-stopped"
    test = service["healthcheck"]["test"]
    assert test and test[0] in ("CMD", "CMD-SHELL")


@pytest.mark.parametrize(
    ("name", "variable", "default", "container_port"),
    [
        ("hal-kb", "KB_PORT", "8910", "8910"),
        ("prometheus", "PROMETHEUS_PORT", "9090", "9090"),
        ("grafana", "GRAFANA_PORT", "3001", "3001"),
    ],
)
def test_host_ports_come_from_env(services, name, variable, default, container_port):
    assert services[name]["ports"] == [f"${{{variable}:-{default}}}:{container_port}"]


def test_hal_kb_is_built_from_the_repo_with_dockerfile_kb(services):
    build = services["hal-kb"]["build"]
    assert _resolve(build["context"]) == REPO
    assert (REPO / build["dockerfile"]).resolve() == DOCKERFILE
    assert set(build["args"]) >= {"WITH_DOCLING", "WITH_RERANK", "APP_UID", "APP_GID"}
    # :local by default; KB_IMAGE_TAG names a variant (S6 rerankers) without replacing it
    assert services["hal-kb"]["image"] == "hal9000/hal-kb:${KB_IMAGE_TAG:-local}"


def test_hal_kb_reads_env_file_and_mounts_data_read_write(services):
    hal = services["hal-kb"]
    env_files = [e["path"] if isinstance(e, dict) else e for e in hal["env_file"]]
    assert [_resolve(p) for p in env_files] == [REPO / ".env"]
    source, mode = _mount_for(hal, "/app/data")
    assert _resolve(source) == REPO / "data" and mode == "rw"
    source, mode = _mount_for(hal, "/app/config")
    assert _resolve(source) == REPO / "config"  # read-write: `kb eval choose` writes it
    assert hal["environment"]["KB_DATA_DIR"] == hal["environment"]["HAL_DATA_DIR"] == "/app/data"


def test_grafana_mounts_follow_the_kwp8_provisioning(services):
    grafana = services["grafana"]
    for source, target in [
        ("grafana.ini", "/etc/grafana/grafana.ini"),
        ("provisioning", "/etc/grafana/provisioning"),
        ("dashboards", "/etc/grafana/dashboards"),
    ]:
        mounted, mode = _mount_for(grafana, target)
        assert _resolve(mounted) == GRAFANA / source and mode == "ro"
    # the dashboards provider reads from where the dashboards are mounted
    provider = yaml.safe_load((GRAFANA / "provisioning/dashboards/knowledge.yaml").read_text())
    assert provider["providers"][0]["options"]["path"] == "/etc/grafana/dashboards"


def test_grafana_mounts_the_data_directory_read_only(services):
    grafana = services["grafana"]
    assert "../../data:/data:ro" in grafana["volumes"]
    source, mode = _mount_for(grafana, "/data")
    assert _resolve(source) == REPO / "data"  # the directory, so -wal and -shm are visible
    assert mode == "ro"  # it can never create -wal/-shm files hal-kb can't write
    datasources = yaml.safe_load((GRAFANA / "provisioning/datasources/knowledge.yaml").read_text())[
        "datasources"
    ]
    sqlite = next(d for d in datasources if d["type"] == "frser-sqlite-datasource")
    assert sqlite["jsonData"]["path"] == "/data/knowledge.db"
    assert "mode=ro" in sqlite["jsonData"]["pathOptions"]


def test_grafana_runs_as_its_own_user(services):
    assert "user" not in services["grafana"]  # the image's uid 472, not hal-kb's
    assert services["hal-kb"]["user"] == "${KB_UID:-1000}:${KB_GID:-1000}"


def test_grafana_requires_a_login_with_the_password_from_env(services):
    env = services["grafana"]["environment"]
    assert env["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert env["GF_SECURITY_ADMIN_PASSWORD"].startswith("${GRAFANA_ADMIN_PASSWORD:?")


def test_prometheus_keeps_90_days_and_scrapes_hal_kb(services):
    prom = services["prometheus"]
    command = prom["command"]
    assert "--storage.tsdb.retention.time=90d" in command
    config = next(c.split("=", 1)[1] for c in command if c.startswith("--config.file="))
    source, mode = _mount_for(prom, config)
    assert _resolve(source) == PROMETHEUS and mode == "ro"
    scrape = yaml.safe_load(PROMETHEUS.read_text())["scrape_configs"]
    targets = [t for job in scrape for sc in job["static_configs"] for t in sc["targets"]]
    assert "hal-kb:8910" in targets  # the service name and the container port


def test_images_are_pinned(services):
    for name, service in services.items():
        if "image" not in service or "build" in service:
            continue
        repo, _, tag = service["image"].rpartition(":")
        assert repo and tag and tag != "latest", name
        assert re.search(r"\d+\.\d+\.\d+", tag), f"{name}: pin a full version, not {tag!r}"


def test_no_secret_is_written_inline(services):
    for name, service in services.items():
        env = service.get("environment", {})
        items = env.items() if isinstance(env, dict) else (e.split("=", 1) for e in env)
        for key, value in items:
            if SECRET_NAME.search(key):
                assert INTERPOLATED.match(str(value)), f"{name}.{key} must come from .env"
    text = COMPOSE.read_text()
    assert not re.search(r"sk-[A-Za-z0-9_-]{16,}", text)
    assert not re.search(r"^\s*(OPENAI_API_KEY|KB_PASSCODE)\s*[:=]", text, re.M)


# --------------------------------------------------------------- Dockerfile
def test_runtime_stage_runs_as_a_non_root_user(runtime_stage):
    users = [rest for keyword, rest in runtime_stage if keyword == "USER"]
    assert users, "the runtime stage has no USER"
    user = users[-1].split(":")[0]
    assert user not in ("root", "0", "")
    args = dict(rest.partition("=")[::2] for kw, rest in runtime_stage if kw == "ARG")
    if user.startswith("${"):
        assert args[user.strip("${}")] not in ("0", "")


def test_runtime_stage_has_a_healthcheck_on_health(runtime_stage):
    checks = [rest for keyword, rest in runtime_stage if keyword == "HEALTHCHECK"]
    assert len(checks) == 1
    assert "http://127.0.0.1:8910/health" in checks[0]


def test_the_service_listens_on_8910_on_all_interfaces(runtime_stage):
    cmd = next(rest for keyword, rest in runtime_stage if keyword == "CMD")
    assert cmd == '["hal9000", "kb", "serve", "--host", "0.0.0.0", "--port", "8910"]'
    assert ("EXPOSE", "8910") in runtime_stage
    assert ("VOLUME", '["/app/data"]') in runtime_stage


def test_no_latest_tags(dockerfile):
    args = _global_args(dockerfile)
    assert args["BASE_IMAGE"] == "python:3.12-slim"
    assert args["NODE_IMAGE"] == "node:22-slim"
    for keyword, rest in dockerfile:
        if keyword != "FROM":
            continue
        image = rest.split()[-3] if " AS " in rest else rest.split()[-1]
        for name, value in args.items():
            image = image.replace(f"${{{name}}}", value)
        assert not image.endswith(":latest"), rest
        if image not in ("scratch",) and not image.startswith("web-"):
            assert ":" in image, f"untagged base image: {rest}"


def test_dependencies_come_from_the_lock_without_dev_tools(dockerfile):
    runs = " ".join(rest for keyword, rest in dockerfile if keyword == "RUN")
    assert "uv sync --frozen --no-dev" in runs
    assert 'groups="--group knowledge"' in runs
    assert "--all-groups" not in runs and "--group dev" not in runs
    assert re.search(r"pip install \"uv==\$\{UV_VERSION\}\"", runs)


def test_every_uv_sync_is_frozen_and_the_code_is_byte_compiled(dockerfile):
    runs = [rest for keyword, rest in dockerfile if keyword == "RUN"]
    syncs = [cmd for run in runs for cmd in run.split("&&") if "uv sync" in cmd]
    assert len(syncs) >= 2  # dependencies, then the project
    for cmd in syncs:
        assert "--frozen" in cmd and "--no-dev" in cmd, cmd
    assert any("python -m compileall" in run and "/app/src" in run for run in runs)


def test_the_vendored_tiktoken_encodings_are_baked_in(dockerfile, runtime_stage):
    copies = [rest for keyword, rest in dockerfile if keyword == "COPY"]
    assert "vendor/tiktoken/ vendor/tiktoken/" in copies
    env = " ".join(rest for keyword, rest in runtime_stage if keyword == "ENV")
    assert "TIKTOKEN_CACHE_DIR=/app/vendor/tiktoken" in env
    # the build checks the files tiktoken will look for: sha1 of each blob URL
    text = DOCKERFILE.read_text()
    for name in ("cl100k_base", "o200k_base"):
        url = f"https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken"
        cached = hashlib.sha1(url.encode()).hexdigest()
        assert f"test -s vendor/tiktoken/{cached}" in text
        assert (REPO / "vendor" / "tiktoken" / cached).stat().st_size > 1_000_000


def test_nothing_is_downloaded_at_run_time(runtime_stage):
    env = " ".join(rest for keyword, rest in runtime_stage if keyword == "ENV")
    for setting in (
        "LITELLM_LOCAL_MODEL_COST_MAP=True",
        "HF_HUB_OFFLINE=1",
        # the models downloaded at build time are where the loaders look
        "FASTEMBED_CACHE_PATH=/opt/models/fastembed",
        "DOCLING_ARTIFACTS_PATH=/opt/models/docling",
    ):
        assert setting in env
    copies = [rest for keyword, rest in runtime_stage if keyword == "COPY"]
    assert any(c.endswith("/opt/models /opt/models") for c in copies)


def test_the_ui_build_is_copied_into_the_runtime(runtime_stage):
    assert ("COPY", "--from=web /web/dist /app/web/dist") in runtime_stage


def test_dockerignore_keeps_house_data_and_build_junk_out():
    patterns = {
        line.strip()
        for line in DOCKERIGNORE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    for required in (".env", "data/", ".git", ".venv", "**/node_modules", "tests/", "web/dist"):
        assert required in patterns, required


def test_env_example_documents_the_deployment_keys():
    text = (REPO / ".env.example").read_text()
    for name in (
        "KB_PASSCODE",
        "KB_UID",
        "KB_GID",
        "GRAFANA_ADMIN_PASSWORD",
        "PROMETHEUS_PORT",
        "GRAFANA_PORT",
        "KB_PORT",
        "KB_WITH_DOCLING",
        "KB_WITH_RERANK",
        "KB_MONTHLY_CAP_USD",
        "KB_EVAL_CAP_USD",
        "KB_DENYLIST",
    ):
        line = next((ln for ln in text.splitlines() if re.match(rf"#? ?{name}=", ln)), None)
        assert line is not None, name
        assert "#" in line.split("=", 1)[1] or line.startswith("#"), f"{name}: no comment"
    # placeholders only: the secrets have no value
    for name in ("GRAFANA_ADMIN_PASSWORD", "KB_PASSCODE"):
        assert re.search(rf"^{name}=\s*(#|$)", text, re.M), name


# ---------------------------------------------------- the committed default
def test_the_committed_default_profile_is_a_valid_profile_name():
    from hal9000.knowledge.chunking.profiles import parse_profile_name
    from hal9000.knowledge.config import load_knowledge_file

    name = load_knowledge_file(REPO / "config" / "knowledge.yaml").active_profile
    assert name == "pdfium.structure-768.hdr.te3s-1536"  # the plan's baseline (G-K1)
    profile = parse_profile_name(name)
    assert profile.name == name  # canonical spelling
    assert (profile.parser, profile.chunker, profile.size, profile.header) == (
        "pdfium",
        "structure",
        768,
        True,
    )
    assert (profile.embed_model, profile.dimensions) == ("openai/text-embedding-3-small", 1536)


def test_the_first_ingest_builds_and_activates_the_committed_default(tmp_path):
    """G-K1 §2.4: `kb ingest` into an empty library leaves the default profile active,
    so `kb gold generate` has a real retriever. The OpenAI embedder is replaced by the
    offline fake (same dimensions); the profile and its name are the committed ones."""
    from hal9000.cost.metering import Meter
    from hal9000.knowledge.config import load_knowledge_file
    from hal9000.knowledge.index.fake import FakeEmbedder
    from hal9000.knowledge.index.profiles import active_profile, profile_status
    from hal9000.knowledge.ingest.pipeline import IngestPipeline
    from hal9000.knowledge.storage import open_db
    from hal9000.storage.db import Database

    knowledge_file = load_knowledge_file(REPO / "config" / "knowledge.yaml")
    note = tmp_path / "boiler.md"
    note.write_text("# Boiler\n\nService the boiler every autumn and bleed the radiators.\n")
    conn = open_db(tmp_path / "knowledge.db")
    ledger = Database(tmp_path / "hal9000.db")
    pipeline = IngestPipeline(
        tmp_path / "files",
        embed_fn=FakeEmbedder(),
        meter=Meter(ledger, "kb-deploy-test", 10, 10),  # a paid model is always metered (K9)
        use_llm=False,
        bus=None,
        knowledge_file=knowledge_file,
    )
    try:
        assert active_profile(conn) is None
        result = pipeline.submit(conn, note, "boiler.md")
        job = pipeline.worker(":unused:").run_job(conn, result.job_id)
        assert job.status == "done", job.error
        active = active_profile(conn)
        assert active is not None and active.name == knowledge_file.active_profile
        assert profile_status(conn, active.name) == "ready"
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert chunks >= 1
    finally:
        pipeline.close()
        conn.close()
        ledger.close()
