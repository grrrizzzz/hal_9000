# Deployment

| Folder | What |
|---|---|
| `docker/` | The house knowledge base's image (`Dockerfile.kb`) and its Compose file (`compose.knowledge.yaml`) |
| `grafana/` | Grafana settings, provisioning, the three knowledge dashboards and alert rules ([`grafana/VERIFY.md`](grafana/VERIFY.md) checks them on the Pi) |
| `prometheus/` | Prometheus scrape config |

The solar agent still runs from a checkout (`uv run hal9000 run`); only the
knowledge base has containers so far.

---

## House knowledge base on the Pi

Three containers run next to Home Assistant on the Raspberry Pi 5 (plan §15).
Home Assistant's container is not part of the Compose file and is never
touched by it; the three services get their own Compose network.

| Service | Image | Memory | Port (`.env`) | What |
|---|---|---|---|---|
| `hal-kb` | built here from `docker/Dockerfile.kb` (`hal9000/hal-kb:local`) | 2 GB | `KB_PORT` (8910) | the web app, API, `/health`, `/metrics` |
| `prometheus` | `prom/prometheus:v3.5.5` (the 3.5 LTS line) | 512 MB | `PROMETHEUS_PORT` (9090) | scrapes `hal-kb` every 30 s, keeps 90 days |
| `grafana` | `grafana/grafana:12.4.12` (OSS) | 512 MB | `GRAFANA_PORT` (3001) | the dashboards; its own login, no anonymous access |

`grafana/grafana-oss` stopped receiving updates at Grafana 12.4.0; the
`grafana/grafana` repository publishes the same OSS build.

All three restart unless stopped, have healthchecks, and rotate their logs
(3 × 10 MB each, to spare the SD card). The ports are for the home network
only: don't forward them on the router.

The step-by-step first deployment is in
[`docs/KNOWLEDGE_RUNBOOK.md`](../docs/KNOWLEDGE_RUNBOOK.md#4-pi-deployment).
This section is the reference.

### Running Compose

Always from the repository root, and always with the root `.env`:

```sh
C="docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml"
$C up -d --build        # build hal-kb if needed, start all three
$C ps                   # all three should say (healthy) after a minute
$C logs -f hal-kb
$C exec hal-kb hal9000 kb stats
$C down                 # stop and remove the containers (data stays)
```

Without `--env-file .env`, Compose looks for `.env` next to the Compose file
and stops with "required variable GRAFANA_ADMIN_PASSWORD is missing a value".
Paths inside the Compose file are relative to it, so `../../data` is the
repository's `data/` folder.

### `.env` keys

`hal-kb` reads the whole `.env` (the same file as on the Mac, see
[`.env.example`](../.env.example)); Compose reads a few more for the stack.
`.env` is git-ignored: never commit it, and keep it `chmod 600`.

| Key | Needed | What |
|---|---|---|
| `KB_PASSCODE` | yes | the household passcode for the web app |
| `OPENAI_API_KEY` | yes | embeddings and answers |
| `GRAFANA_ADMIN_PASSWORD` | yes | Grafana's `admin` password, set on Grafana's first start (it is kept in the `grafana-data` volume after that) |
| `KB_UID`, `KB_GID` | yes on the Pi | the owner of `data/`: the output of `id -u` and `id -g` (default 1000) |
| `KB_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT` | if taken | host ports (8910, 9090, 3001) |
| `KB_WITH_DOCLING` | no | `1` builds Docling and its layout and table models into the image (PyTorch: gigabytes; only if the evaluation chose it) |
| `KB_WITH_RERANK` | no | `none`, `minilm` or `bge-base`: the cross-encoder chosen at G-K2, downloaded at build time |
| `KB_MONTHLY_CAP_USD`, `KB_EVAL_CAP_USD`, `KB_CHAT_TIER`, ... | no | as on the Mac |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | no | only for Grafana's optional Telegram alerts |

The Compose file sets `HAL_DATA_DIR`, `KB_DATA_DIR` (`/app/data`) and
`HAL_CONFIG_DIR` (`/app/config`) itself, so paths in `.env` never point the
container at the wrong place.

After changing `.env`, run `$C up -d` (not `restart`: a restart keeps the old
environment). After changing `KB_WITH_*` or `KB_UID`, run `$C up -d --build`.

### What is mounted

| Host | Container | Mode | Why |
|---|---|---|---|
| `data/` | `hal-kb:/app/data` | read-write | `knowledge.db`, `knowledge/files/` (originals by hash), `knowledge/eval/`, `backups/`, `hal9000.db` (the shared cost ledger) |
| `config/` | `hal-kb:/app/config` | read-write | `knowledge.yaml` etc. from the checkout: a `git pull` plus `$C up -d` applies it without a rebuild, and `kb eval choose` writes it at G-K2 |
| `data/` | `grafana:/data` | read-write | the SQLite datasource opens `/data/knowledge.db` with `mode=ro` |
| `deploy/grafana/{grafana.ini,provisioning,dashboards}` | `grafana:/etc/grafana/...` | read-only | KWP8's provisioning |
| `deploy/prometheus/prometheus.yml` | `prometheus:/etc/prometheus/prometheus.yml` | read-only | scrape config |
| volumes `prometheus-data`, `grafana-data` | | | the time series and Grafana's own database |

**Why Grafana mounts `data/` read-write and runs as `hal-kb`'s user.**
`knowledge.db` is in WAL mode. A reader has to be able to create and map
`knowledge.db-shm`, which a read-only mount prevents whenever `hal-kb` isn't
holding the file open, and it mounts the directory rather than the single
file so the `-wal` and `-shm` files are visible at all. Grafana normally runs
as uid 472; if it created those two files while `hal-kb` was stopped, `hal-kb`
(a different user) could not write its own database afterwards. So both
containers run as `KB_UID:KB_GID`, the owner of `data/`: whichever creates the
side files, the other can use them. The datasource's `mode=ro` (and the
plugin's `query_only`) still stops Grafana from changing anything.

### The image

`deploy/docker/Dockerfile.kb`, built from the repository root (the root
`.dockerignore` keeps `.env`, `data/`, `.git`, `.venv`, `node_modules` and the
tests out of the build context):

1. **web**: `node:22-slim` runs `npm ci` and `npm run build` in `web/`. Node
   is only used here.
2. **build**: `python:3.12-slim` installs `uv` from PyPI and runs
   `uv sync --frozen --no-dev --group knowledge` (plus `knowledge-docling` and
   `knowledge-rerank` when chosen) into `/app/.venv`. It removes the
   third-party test suites, then fails the build if sqlite-vec can't run its
   cosine KNN (`scripts/kb_sqlite_vec_check.py`, gate G-K0b), if the vendored
   `tiktoken` encodings are missing, or if the service doesn't import. With a
   reranker it downloads the model from Hugging Face and scores one pair; with
   Docling it downloads the layout and table models and converts a blank page
   offline. Models live in `/opt/models`.
3. **runtime**: `python:3.12-slim` with `/app` (code, venv, `vendor/tiktoken`,
   `web/dist`), running as the non-root user `hal` (`APP_UID`/`APP_GID`, from
   `KB_UID`/`KB_GID`). `/app/data` is its only writable place. The
   healthcheck calls `/health` (200 when `ok` or `degraded`). The command is
   `hal9000 kb serve --host 0.0.0.0 --port 8910`.

At run time it downloads nothing: the encodings are baked in,
`LITELLM_LOCAL_MODEL_COST_MAP=True` uses LiteLLM's bundled price map, and
`HF_HUB_OFFLINE=1` keeps the reranker and Docling from looking for models
online. Only OpenAI is
called.

| Build argument | Values | Default |
|---|---|---|
| `WITH_DOCLING` | `0`, `1` | `0` |
| `WITH_RERANK` | `none`, `minilm`, `bge-base` | `none` |
| `APP_UID`, `APP_GID` | numbers | `1000` |
| `BASE_IMAGE`, `NODE_IMAGE` | image references | `python:3.12-slim`, `node:22-slim` |
| `SKIP_WEB_BUILD` | `0`, `1` | `0` (1 is for offline tests only; see below) |

Building on the Pi needs Docker Hub, PyPI and the npm registry, and
huggingface.co when `WITH_RERANK` isn't `none` or `WITH_DOCLING` is `1`. The first build takes
10–20 minutes on a Pi 5; later builds reuse the cached dependency layers
unless `pyproject.toml`, `uv.lock` or `web/package-lock.json` change.

**Size.** The target is under 900 MB with `WITH_DOCLING=0 WITH_RERANK=none`.
Measured in the build workspace on x86_64 with a stand-in base (below): 580 MB
for `/app`, plus the base image (python:3.12-slim is about 120 MB unpacked).
Record the arm64 sizes with `docker image ls hal9000/hal-kb` after building
on the Pi (runbook §4.9).

### Backups

`hal9000 kb backup` makes an online, consistent copy of `knowledge.db` (with
SQLite's backup API, so it is safe while the service runs) and a tarball of
`knowledge/files/` and `knowledge/eval/`, with a checksummed manifest, in
`data/backups/knowledge-<UTC time>/`:

```sh
cd ~/hal_9000
C="docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml"
$C exec -T hal-kb hal9000 kb backup
ls data/backups/
```

Nightly at 03:17, keeping two weeks, from your user's crontab (`crontab -e`):

```cron
17 3 * * * cd $HOME/hal_9000 && mkdir -p data/backups && docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml exec -T hal-kb hal9000 kb backup >> data/backups/cron.log 2>&1 && find data/backups -maxdepth 1 -name 'knowledge-*' -mtime +14 -exec rm -rf {} +
```

`data/backups` is on the same SD card as the library, so copy it off the Pi
now and then (to the Mac, from the Mac):

```sh
rsync -av pi@<pi-address>:hal_9000/data/backups/ ~/hal-kb-backups/
```

### Restore

`kb restore` checks the backup's checksums, refuses to overwrite an existing
library unless `--force` (which moves the old one aside to
`data/knowledge-replaced-<time>/`, never deletes it), leaves `hal9000.db`
alone, and refuses while the service holds the library lock. So stop
`hal-kb` first:

```sh
cd ~/hal_9000
C="docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml"
ls data/backups/                                   # pick one
$C stop hal-kb
$C run --rm --no-deps hal-kb hal9000 kb restore data/backups/knowledge-<time> --force
$C up -d hal-kb
curl -s http://localhost:8910/health
$C exec hal-kb hal9000 kb stats
```

A backup copied back from the Mac goes into `data/backups/` first (the
container only sees `data/`).

### Upgrade

```sh
cd ~/hal_9000
C="docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml"
$C exec -T hal-kb hal9000 kb backup            # a restore point
docker tag hal9000/hal-kb:local hal9000/hal-kb:previous
git rev-parse --short HEAD > data/backups/previous-commit
git pull
$C up -d --build
$C ps                                          # healthy after a minute
curl -s http://localhost:8910/health
```

### Rollback

Back to the previous image and code:

```sh
cd ~/hal_9000
C="docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml"
git checkout "$(cat data/backups/previous-commit)"
docker tag hal9000/hal-kb:previous hal9000/hal-kb:local
$C up -d --no-build hal-kb
$C exec hal-kb hal9000 kb check
```

If the upgrade migrated `knowledge.db` to a newer schema, the older code
refuses to open it (`kb check` says so). Then also restore the backup made
just before the upgrade (Restore, above). Go back to the branch with
`git checkout knowledge` once the problem is fixed.

### Testing the image without a registry

The build workspace can't reach Docker Hub, so `python:3.12-slim` and
`node:22-slim` can't be pulled there. `deploy/docker/make-standin-base.sh`
builds stand-ins from the host's own files with `docker import`
(`hal-kb-standin:py312`: python 3.12, pip, a shell and coreutils, `useradd`,
the host's CA bundle; `hal-kb-standin:node22` adds the host's Node 22). Then:

```sh
deploy/docker/make-standin-base.sh
docker buildx build --load -f deploy/docker/Dockerfile.kb \
  --build-arg BASE_IMAGE=hal-kb-standin:py312 \
  --build-arg NODE_IMAGE=hal-kb-standin:node22 \
  -t hal-kb:standin .
# or, with a web/dist built on the host instead of the Node stage:
docker buildx build --load -f deploy/docker/Dockerfile.kb \
  --build-arg BASE_IMAGE=hal-kb-standin:py312 --build-arg SKIP_WEB_BUILD=1 \
  --build-context webdist=web/dist -t hal-kb:standin .
docker run --rm --network none hal-kb:standin hal9000 kb serve --fake
```

The stand-in is x86_64 only and is never used on the Pi.
`tests/knowledge/test_kb_deploy.py` checks the Compose file and the
Dockerfile statically in the normal test suite.
