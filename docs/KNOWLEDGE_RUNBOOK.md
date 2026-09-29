# House knowledge base runbook

Copy-paste steps for every gate of the knowledge base build, then for running
it day to day. Each step says where it runs (the **Mac** or the **Pi**), what
it costs, and what a pass looks like. Stop at the first step that doesn't look
right; the troubleshooting table in §7 covers the likely causes.

When you report back, paste command output, never question text, answers or
document text. `kb eval export` files are safe to share: they hold aggregates
and per-question scores only.

| § | Gate | Where | Time | Cost |
|---|---|---|---|---|
| 2 | G-K1: ingest, house test, question set, review | Mac | about 1 hour, can be split | about $1 |
| 3 | G-K1b: sweep S1a–S4, judge labels, S5, export | Mac | 15 min + unattended runs | capped at $10 in all |
| 4 | Pi deployment | Pi (and the Mac for the copy) | about 1 hour, mostly building | a cent for the smoke test |
| 5 | S6: latency on the Pi | Pi | about 30 min | a few cents |
| 6 | G-K2: choose the configuration | Pi | 15 min | none, unless a profile must be built |
| 7 | Daily use and troubleshooting | | | |
| 8 | Backup and restore | Pi | | |

## 1. Conventions

On the **Mac**, every command runs from the repository folder:

```bash
cd ~/AI/hal_9000
```

On the **Pi**, every command runs from the repository folder, with Compose
always given the root `.env` (see [`deploy/README.md`](../deploy/README.md)).
Set this once per terminal session:

```sh
cd ~/hal_9000
C="docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml"
```

On the Mac, set the Pi's SSH login once per terminal session (for the copy
steps):

```bash
PI=<you>@<pi-address>        # e.g. mike@192.168.1.50
```

`RUN` or `SWEEP` below means the sweep ID that `kb eval sweep` printed.

---

## 2. G-K1: the question set (Mac)

### 2.1 Update and install (5 minutes, no cost)

```bash
git fetch && git checkout knowledge && git pull
uv sync --all-groups
```

`.env` needs `OPENAI_API_KEY` and `KB_PASSCODE` (the household passcode for
the web page).

### 2.2 Put the inspection bundle in the inbox

```bash
mkdir -p data/knowledge/inbox
cp ~/Downloads/Home_Inspection.pdf data/knowledge/inbox/Home_Inspection.pdf
```

(Use wherever the PDF actually is. `data/` is git-ignored; the PDF never goes
into git.)

### 2.3 Check (a fraction of a cent)

```bash
uv run hal9000 kb check
```

Pass: `ok` on every line and `All checks passed.` (the index profile line may
say there is no active profile yet).

### 2.4 Ingest (about a cent)

```bash
uv run hal9000 kb ingest data/knowledge/inbox/Home_Inspection.pdf
```

It prints an estimate and asks before spending. Pass: the file is `indexed`
(or stored and parsed, if there is no profile yet) with no failures.

### 2.5 The house test (no cost)

```bash
uv run pytest -m house --no-cov
```

Pass: every `house` test passes (89 pages, the eleven sub-documents, no header
noise in chunks).

### 2.6 Generate the draft question set (about $1)

```bash
uv run hal9000 kb gold generate --yes
```

Pass: a summary of about 120 drafts by category, with any quota shortfall
listed. If it stops because unanswerable drafts can't be checked against an
index, tell the orchestrator rather than adding `--allow-keyword-check`.

### 2.7 Review (about 45 minutes, can be split)

```bash
uv run hal9000 kb serve
```

Open <http://localhost:8910>, sign in with the passcode, and go to
**Evaluation → Question set**. Approve, fix or reject each item; your work is
saved as you go, so you can stop and come back. The page shows the counts.
Done when at least 100 are approved, including at least 20 unanswerable, with
the category and paraphrase quotas met (or you accept the shortfall). Stop the
server with Ctrl-C.

---

## 3. G-K1b: sweep, judge calibration, S5 (Mac)

Stop `kb serve` first (Ctrl-C): the sweep works on the same database.

### 3.1 Sweep S1a–S4 (unattended; within the $10 evaluation cap)

```bash
uv run hal9000 kb eval sweep --stages S1a,S1b,S2,S3,S4 --yes
```

It prints each stage's estimate before running it and the sweep ID at the
start: note it as `SWEEP`. It stops at `KB_EVAL_CAP_USD` and keeps its partial
results (exit code 4). If it is interrupted, resume it:

```bash
uv run hal9000 kb eval sweep --stages S1a,S1b,S2,S3,S4 --resume SWEEP --yes
```

### 3.2 Answers for judge calibration (a few cents)

```bash
uv run hal9000 kb eval judge-prep SWEEP --yes
```

### 3.3 Label the 20 answers (about 15 minutes)

```bash
uv run hal9000 kb serve
```

**Evaluation → Judge calibration**: label all 20. Then Ctrl-C.

### 3.4 S5 (unattended)

```bash
uv run hal9000 kb eval sweep --stages S5 --resume SWEEP --yes
```

It refuses to start until the judge agrees with at least 85% of your labels.
If it refuses, send the orchestrator the agreement it printed (the rubric gets
one revision).

### 3.5 Export and share

```bash
uv run hal9000 kb eval export SWEEP
```

It writes `data/knowledge/eval/runs/SWEEP/export.json`. Share that file. The
orchestrator summarises the results and names the finalists for S6.

---

## 4. Pi deployment

Reference: [`deploy/README.md`](../deploy/README.md). Home Assistant's
container is not touched by any of this.

### 4.1 Prerequisites (Pi, 5 minutes)

```sh
uname -m                                  # aarch64
docker version --format '{{.Server.Version}}'
docker compose version                    # v2.24 or newer
id -nG | grep -qw docker && echo "in the docker group"
free -h                                   # the stack is limited to 3 GB in all
df -h ~                                   # 5 GB free for images and build cache
ss -ltnp | grep -E ':(8910|9090|3001)\b' || echo "ports free"
curl -sI https://registry-1.docker.io/v2/ | head -1   # any HTTP status: Docker Hub reachable
curl -sI https://pypi.org/simple/ | head -1
curl -sI https://registry.npmjs.org/ | head -1
```

Pass: `aarch64`, a Docker version, Compose v2.24+, "in the docker group",
"ports free", and an HTTP status line from each of the three registries.

- No `docker compose`: install Docker's `docker-compose-plugin` package.
- Not in the docker group: `sudo usermod -aG docker $USER`, then log out and in.
- A port is taken: choose another and set `KB_PORT`, `PROMETHEUS_PORT` or
  `GRAFANA_PORT` in `.env` in step 4.3.

### 4.2 Get the code (Pi)

```sh
git clone https://github.com/grrrizzzz/hal_9000.git ~/hal_9000   # same credentials as on the Mac
cd ~/hal_9000
git checkout knowledge
git pull
```

(If the checkout exists from gate G-K0b: `cd ~/hal_9000 && git fetch && git checkout knowledge && git pull`.)

### 4.3 `.env` (Mac, then Pi)

Copy the Mac's `.env` (it has `OPENAI_API_KEY` and `KB_PASSCODE`). On the Mac:

```bash
scp .env "$PI:hal_9000/.env"
```

On the Pi, add the Pi-only keys. The password prompt doesn't echo:

```sh
cd ~/hal_9000
chmod 600 .env
read -rsp "Grafana admin password: " GP; echo
printf '\n# Pi: knowledge stack (deploy/README.md)\nKB_UID=%s\nKB_GID=%s\nGRAFANA_ADMIN_PASSWORD=%s\n' \
  "$(id -u)" "$(id -g)" "$GP" >> .env
unset GP
grep -oE '^(OPENAI_API_KEY|KB_PASSCODE|GRAFANA_ADMIN_PASSWORD|KB_UID|KB_GID)=.' .env | cut -d= -f1
```

Pass: the last command lists all five names (it prints no values). `.env`
stays on the Pi and the Mac only; it is git-ignored.

### 4.4 Build (Pi, 10–20 minutes the first time)

```sh
cd ~/hal_9000
C="docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml"
mkdir -p data/knowledge
$C build hal-kb
$C pull prometheus grafana
```

`mkdir` first, as your user, so Docker doesn't create `data/` as root. Pass:
the build ends without an error, and its log includes
`PASS: sqlite-vec loads and cosine KNN matches brute force.`

### 4.5 Copy the library from the Mac

This reuses the Mac's embeddings instead of recomputing them. Nothing is
running on the Pi yet, so its library is empty.

On the **Mac**, stop `kb serve` (Ctrl-C) and anything else using the library,
then fold the write-ahead log into `knowledge.db` so the single file is
complete:

```bash
uv run python -c "import sqlite3; c = sqlite3.connect('data/knowledge.db'); print(c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()); c.close()"
ls data/knowledge.db*
```

Pass: `(0, 0, 0)`, and only `data/knowledge.db` is listed (a leftover
`-wal` of 0 bytes is fine). A first number of 1 means something still has the
database open.

Still on the Mac, copy the database, the originals and the evaluation data
(not `session.key`: the Pi makes its own, so everyone signs in again):

```bash
ssh "$PI" 'mkdir -p hal_9000/data/knowledge && rm -f hal_9000/data/knowledge.db-wal hal_9000/data/knowledge.db-shm'
rsync -av data/knowledge.db "$PI:hal_9000/data/"
rsync -av data/knowledge/files/ "$PI:hal_9000/data/knowledge/files/"
rsync -av data/knowledge/eval/ "$PI:hal_9000/data/knowledge/eval/"
```

On the Pi, check that everything belongs to your user:

```sh
cd ~/hal_9000
ls -ln data/knowledge.db
find data ! -user "$(id -u)" | head
```

Pass: `knowledge.db` has your uid, and `find` prints nothing.

(Alternative, once `kb backup` is in your checkout: `uv run hal9000 kb backup`
on the Mac, `rsync -av data/backups/knowledge-<time> "$PI:hal_9000/data/backups/"`,
then the restore in §8 on the Pi. It is checksummed end to end.)

### 4.6 Start (Pi)

```sh
$C up -d
sleep 60
$C ps
```

Pass: `hal-kb`, `hal-kb-prometheus` and `hal-kb-grafana` are all `Up ... (healthy)`.

### 4.7 Smoke tests (Pi; about a cent)

```sh
curl -s http://localhost:8910/health; echo
$C exec hal-kb hal9000 kb check
$C exec hal-kb hal9000 kb stats
$C exec hal-kb hal9000 kb profile list
```

Pass: `/health` returns JSON with `"status"`; `kb check` ends with
`All checks passed.`; `kb stats` shows the same document and chunk counts as
`uv run hal9000 kb stats` on the Mac.

Before G-K2, `/health` may say `"status":"degraded"` with
`"active_profile":null`: no profile serves the chat yet. For the smoke test,
activate the leading configuration's profile for now (the orchestrator names
it from the S5 results; G-K2 sets the final one):

```sh
$C exec hal-kb hal9000 kb profile activate <PROFILE>
```

Then one question from the terminal, inside the container:

```sh
$C exec hal-kb hal9000 kb ask "When should the air filter be replaced?"
```

Pass: an answer with `[S1]`-style markers and a Sources list, or a clear
"couldn't find that" with the closest sources; then `Outcome:` and `Cost:`
lines. An `error` outcome means the key or network is wrong (§7).

Then from the Mac's browser: open `http://<pi-address>:8910`, sign in with the
passcode, ask a question, and open a citation (it opens at the cited page).

Data survives a restart:

```sh
$C restart hal-kb
sleep 45
curl -s http://localhost:8910/health; echo
```

Pass: healthy again, and the Library page still lists every document.

Prometheus scrapes hal-kb:

```sh
curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"[a-z]*"'
```

Pass: `"health":"up"`.

### 4.8 Grafana: first login and verification (Pi, 10 minutes)

Open `http://<pi-address>:3001` and sign in as `admin` with
`GRAFANA_ADMIN_PASSWORD`. **Dashboards → Knowledge** holds three dashboards:
Knowledge: Ingestion, Knowledge: Live and Knowledge: Quality.

Then follow [`deploy/grafana/VERIFY.md`](../deploy/grafana/VERIFY.md) step by
step, with these two lines in place of its step 0 shell setup (this stack's
password key is `GRAFANA_ADMIN_PASSWORD`, and Compose needs `--env-file`):

```sh
C="docker compose --env-file .env -f deploy/docker/compose.knowledge.yaml"
PW=$(grep '^GRAFANA_ADMIN_PASSWORD=' .env | cut -d= -f2-)
G="http://localhost:3001"
```

It checks the Prometheus config, that Grafana logs no provisioning errors, the
pinned SQLite plugin, that anonymous access is off, the three dashboards, that
a fresh trace appears within 60 seconds, and the four alert rules.

### 4.9 Record the image size (Pi)

The unpacked size, in MB (with Docker's containerd image store,
`docker image ls` adds the compressed download to it, so it reads higher):

```sh
size() { docker run --rm --network none --entrypoint sh "$1" -c 'set -- $(du -sxm / 2>/dev/null); echo $1'; }
echo "local: $(size hal9000/hal-kb:local) MB"
```

Pass: under 900. Send the number. For the build record, the size with each
reranker (optional; about 10 minutes each, needs huggingface.co):

```sh
for r in minilm bge-base; do
  docker build -f deploy/docker/Dockerfile.kb --build-arg WITH_RERANK=$r \
    --build-arg APP_UID="$(id -u)" --build-arg APP_GID="$(id -g)" -t hal9000/hal-kb:$r .
  echo "$r: $(size hal9000/hal-kb:$r) MB"
done
```

These extra images don't change the running stack (it uses `:local`). Keep
the one S6 needs, if any (§5), and remove the rest:
`docker image rm hal9000/hal-kb:minilm hal9000/hal-kb:bge-base`.

### 4.10 Nightly backups (Pi)

Add the cron line from [`deploy/README.md`](../deploy/README.md#backups)
with `crontab -e`, then make one backup now:

```sh
$C exec -T hal-kb hal9000 kb backup
ls data/backups/
```

---

## 5. S6: latency on the Pi

Needs §4 done. The finalists' profiles came across with the library; the
orchestrator gives you their names (`A,B,C` below) and the sweep ID.

**If a finalist uses a reranker**, the image must include that model (nothing
is downloaded at run time). Set `KB_WITH_RERANK=minilm` (or `bge-base`) in
`.env`, run `$C up -d --build`, and run S6 for that finalist; set it back
afterwards if G-K2 doesn't choose it.

The scratch-ingest test times a fresh ingestion of the bundle with an empty
cache. It needs the bundle's SHA-256; on the Mac:

```bash
shasum -a 256 data/knowledge/inbox/Home_Inspection.pdf | cut -c1-64
```

On the Pi, check it is there (`ls data/knowledge/files/<SHA>`), keep the Pi
otherwise idle (no uploads), and run:

```sh
$C exec hal-kb hal9000 kb eval latency --profiles A,B,C --n 200 --end-to-end 30 --ingest-test <SHA>
```

It prints an estimate (the live end-to-end questions and the scratch ingest's
embeddings, metered as `knowledge-eval`) and asks before spending. Pass: a
`Stored <run id> (<profile>, host ...)` line per profile and the spend. Exit
code 4 means it stopped at the evaluation cap with partial results.

Write the report (it goes to `docs/KNOWLEDGE_EVAL_REPORT.md` in the checkout,
aggregates only) and the export:

```sh
$C run --rm --no-deps -v "$PWD/docs:/app/docs" hal-kb hal9000 kb eval report SWEEP
$C exec hal-kb hal9000 kb eval export SWEEP
```

Copy both to the Mac and share them:

```bash
scp "$PI:hal_9000/docs/KNOWLEDGE_EVAL_REPORT.md" "$PI:hal_9000/data/knowledge/eval/runs/SWEEP/export.json" ~/Downloads/
```

---

## 6. G-K2: choose the configuration (Pi)

The orchestrator presents the report's recommendation and the alternatives.
Pick one (`NAME`, as the report prints it), then on the Pi:

```sh
$C exec hal-kb hal9000 kb eval choose SWEEP --config NAME
```

It builds the profile first if it isn't built (it shows the embedding cost and
asks), activates it, stores its abstain threshold, writes
`config/knowledge.yaml`, and says whether `KB_CHAT_TIER` should change.

1. If it says so, set `KB_CHAT_TIER` in `.env`.
2. If the chosen configuration has a reranker, set `KB_WITH_RERANK` in `.env`
   to it (and back to `none` if not).
3. Apply: `$C up -d --build` (it only rebuilds if `KB_WITH_RERANK` changed).
4. Check: `curl -s http://localhost:8910/health; echo` shows `"status":"ok"`
   and the chosen profile.
5. Commit the choice, on the Pi or by copying the file to the Mac:

   ```sh
   git diff config/knowledge.yaml
   git add config/knowledge.yaml && git commit -m "G-K2: choose NAME"
   git push
   ```

Tell the orchestrator which configuration you chose; it records the decision
in `docs/KNOWLEDGE_BUILD_STATUS.md`. From now on the Pi's library is the one
in use; the Mac's copy is only the evaluation's.

---

## 7. Daily use and troubleshooting

**Use it**: open `http://<pi-address>:8910` on any device on the home Wi-Fi and
sign in with the household passcode. Add documents on the Library page (drop
files or a whole folder, or use the folder picker). Ask in the chat; every
answer cites its sources, and a citation opens the document at that page.

**Watch it**: `http://<pi-address>:3001` (Grafana) for questions, latency,
cost against the monthly cap and ingestion; `$C exec hal-kb hal9000 kb stats`
for a summary in the terminal.

**From the Pi's terminal:**

```sh
$C ps                                             # all three healthy?
$C logs --tail 100 hal-kb
$C exec hal-kb hal9000 kb ask "question"          # one question, with sources
$C exec hal-kb hal9000 kb ingest data/knowledge/inbox/<file>   # after copying a file there
$C exec hal-kb hal9000 kb eval regression         # recall on the approved question set, retrieval only
```

| Symptom | Likely cause and fix |
|---|---|
| Compose: `required variable GRAFANA_ADMIN_PASSWORD is missing a value` | `--env-file .env` was left out (use `$C` from §1), or the key isn't in `.env` (§4.3). |
| Compose: `env file .../.env not found` | Run from the repository root; `.env` must be there (§4.3). |
| `bind: address already in use` | Another program has the port. Set `KB_PORT`, `PROMETHEUS_PORT` or `GRAFANA_PORT` in `.env`, then `$C up -d`. |
| The build fails at `kb_sqlite_vec_check.py` | sqlite-vec doesn't work in this image on this Pi (the G-K0b risk). Stop and send the orchestrator the output. |
| The build fails downloading the reranker or Docling's models | The Pi can't reach huggingface.co. Build with `KB_WITH_RERANK=none` / `KB_WITH_DOCLING=0`, or retry later. |
| `hal-kb` keeps restarting, log says `KB_PASSCODE is not set` | Add it to `.env`, then `$C up -d`. |
| `/health`: `"active_profile":null` | No profile serves the chat: §4.7 (for now) or §6 (for good). |
| `/health`: `"retrieval_only":true` | The month's spend reached `KB_MONTHLY_CAP_USD`: the chat shows sources without writing answers. Raise the cap in `.env` and `$C up -d`, or wait for next month. |
| Answers fail with an `error` outcome | The OpenAI key or the Pi's internet connection. `$C exec hal-kb hal9000 kb check` checks both. |
| Log: `attempt to write a readonly database` or `Permission denied` under `/app/data` | `data/` isn't owned by `KB_UID:KB_GID`. `sudo chown -R "$(id -u):$(id -g)" data`, check `KB_UID`/`KB_GID` in `.env`, then `$C up -d`. |
| Grafana's database panels say `unable to open database file` | The same ownership problem (Grafana runs as `KB_UID:KB_GID` too): fix as above. |
| Grafana rejects the admin password | The password is set on Grafana's first start only. Reset it: `$C exec grafana grafana cli admin reset-admin-password '<new>'`, and put the new one in `.env`. |
| Prometheus target `down` | `hal-kb` isn't healthy: `$C logs --tail 100 hal-kb`. |
| `hal-kb` was killed during a big upload (`docker inspect hal-kb --format '{{.State.OOMKilled}}'` says `true`) | It hit its 2 GB limit. Upload large documents one at a time; tell the orchestrator if it repeats. |
| A setting changed in `.env` has no effect | `restart` keeps the old environment. Use `$C up -d`. |

---

## 8. Backup and restore

Details and the cron line are in [`deploy/README.md`](../deploy/README.md#backups).

```sh
$C exec -T hal-kb hal9000 kb backup            # data/backups/knowledge-<UTC time>/
```

Restore (it refuses while the service runs, and moves the current library
aside rather than deleting it):

```sh
ls data/backups/
$C stop hal-kb
$C run --rm --no-deps hal-kb hal9000 kb restore data/backups/knowledge-<time> --force
$C up -d hal-kb
curl -s http://localhost:8910/health; echo
$C exec hal-kb hal9000 kb stats
```

Before an upgrade, make a backup; to go back to the previous version, follow
Upgrade and Rollback in [`deploy/README.md`](../deploy/README.md#upgrade).
