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
| 2 | G-K1: ingest, house test, question set, review | Mac | about 1 hour of your time, plus 60–90 min unattended generation | about $2.50 |
| 3 | G-K1b: sweep S1a–S4, judge labels, S5, export | Mac | 15 min + unattended runs | capped at $10 in all |
| 4 | Pi deployment | Pi (and the Mac for the copy) | about 1 hour, mostly building | a cent for the smoke test |
| 5 | S6: latency on the Pi | Pi | about 30 min | a few cents |
| 6 | G-K2: choose, then the one-time test split | Pi | 30 min | the test split's judged answers (estimate shown first) |
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

**Values you fill in are shell variables.** Set each one once per terminal
session, replacing the example value after `=` (with nothing else around it:
zsh on the Mac reads `<` and `>` as redirections, so never type them). The
commands then use `$SWEEP`, `$NAME` and so on as they are.

| Variable | Where | What | Set it like |
|---|---|---|---|
| `PI` | Mac | the Pi's SSH login | `PI=mike@192.168.1.50` |
| `SWEEP` | Mac and Pi | the sweep ID `kb eval sweep` printed (§3.1) | `SWEEP=20261002T0915-a1b2` |
| `SHA` | Pi | the inspection bundle's SHA-256 (§5) | `SHA=0f3c...` (all 64 characters) |
| `PROFILE` | Pi | a profile name, from the orchestrator (§4.7) | `PROFILE=pdfium.structure-768.hdr.te3s-1536` |
| `NAME` | Pi | the configuration you choose at G-K2 (§6) | `NAME=s4-07` |
| `BACKUP` | Pi | a backup folder (§8) | `BACKUP=data/backups/knowledge-20261003T031700Z` |

Passwords and the passcode: use only letters, digits, `-` and `_`. Quotes,
spaces, `#` and `$` are read differently by Compose, the shell and `.env`
parsers.

---

## 2. G-K1: the question set (Mac)

### 2.1 Update and install (5 minutes, no cost)

```bash
git fetch && git checkout knowledge && git pull
uv sync --all-groups
(cd web && npm ci && npm run build)
```

The last line builds the web page that `kb serve` shows (it needs Node 22 or
newer: `node -v`; on a Mac, `brew install node`). Re-run it after any pull that
changes `web/`. Without it, <http://localhost:8910> only explains how to build it.

`.env` needs `OPENAI_API_KEY` and `KB_PASSCODE` (the household passcode for
the web page: letters, digits, `-` and `_` only).

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

Pass: `ok` on every line and `All checks passed.` Before the first ingest the
index profile line reads `no active profile yet; default for the first ingest:
pdfium.structure-768.hdr.te3s-1536`: the plan's baseline, set in
`config/knowledge.yaml` until G-K2 replaces it.

### 2.4 Ingest (about a cent)

```bash
uv run hal9000 kb ingest data/knowledge/inbox/Home_Inspection.pdf
```

It prints an estimate (embeddings and segmentation) and asks before
spending. Because the library has no active profile yet, the ingestion also
builds the default profile from `config/knowledge.yaml` and activates it, so
the real retriever exists for the next steps. Pass: `added
Home_Inspection.pdf: ...`, then `1 added, 0 duplicate, 0 skipped, 0 failed.`
with the cost. Then:

```bash
uv run hal9000 kb check
```

Pass: the index profile line now reads `active profile
pdfium.structure-768.hdr.te3s-1536; ...`.

### 2.5 The house test (no cost)

```bash
uv run pytest -m house --no-cov
```

Pass: every `house` test passes (89 pages, the eleven sub-documents, no header
noise in chunks).

### 2.6 Generate the draft question set (about $2.50, 60–90 minutes unattended)

```bash
uv run hal9000 kb gold generate --yes
```

It prints the plan (documents, pages, generation tasks, unanswerable drafts
to check), then the estimate and its basis, for example:

```
Estimate: about 75 calls to openai/gpt-5 (35 generation + 4 repair reserve + 36 unanswerable checks): ... estimated $2.58, taking about 1 h 28 min (cap --max-cost $3.00).
Basis: ~3,190 completion tokens per call incl. reasoning, the mean of 38 earlier gold-generation calls to openai/gpt-5; ~70 s per call from their ledger timestamps.
```

GPT-5 thinks before it answers, and its reasoning tokens are billed as
output: a call takes about 70 s and costs about 3.4 cents, so the whole run
takes 60–90 minutes and costs about $2.50. The estimate counts that (from
the cost ledger once there are earlier calls, else a default allowance), the
unanswerable checks and a reserve for repair calls. If the estimate exceeds
`--max-cost` (default $3.00) it refuses and says which `--max-cost` to pass.

While it runs, one progress line per LLM call appears, e.g.
`[12/75] window 9/29 (Home inspection report, pp. 20–22): 34 items so far; $0.43 spent; ~70 s/call; ~74 min left`.
Nothing is lost if it stops: each finished call is saved to a checkpoint
under `data/knowledge/eval/gold/`. Ctrl-C lets the call in flight finish
(it is paid for either way; press Ctrl-C again to abandon it), then stores the
drafts collected so far and prints the exact command to continue, e.g.
`uv run hal9000 kb gold generate --resume gold-20261001-181502-3fa0 --max-cost 1.50`.
A stop at the `--max-cost` cap (or `KB_EVAL_CAP_USD`) does the same and exits
with code 4; Ctrl-C exits 130, SIGTERM 143. `--resume` on its own continues
the latest unfinished run; finished calls are never paid for again, nothing
is stored twice, and a checkpoint another terminal is running is refused.

Pass: a summary of about 120 drafts by category, with any quota shortfall
listed. Unanswerable drafts are checked against the profile §2.4 activated;
don't add `--allow-keyword-check`. If it stops for want of an index, §2.4's
`kb check` line didn't show the active profile: send the orchestrator both
outputs.

**After a run that stopped at the cap in the unanswerable checks** (the
first run, before checkpoints existed, did: it stored 24 unanswerable drafts,
16 of them "FLAGGED: could not be checked (retrieval failed:
BudgetExceeded)"), do these three steps in order:

1. Check the drafts that were never checked:

   ```bash
   uv run hal9000 kb gold recheck --yes
   ```

   It lists how many unreviewed unanswerable drafts say they could not be
   checked, prints the estimate, then checks each against the whole library
   with the real index: about 16 checks, about $0.55 and 20 minutes at the
   observed rates. Each result is saved as it comes: the note becomes
   "Checked: not found in the documents." or
   "FLAGGED: retrieval found it answered: …" (or "partly answered"). Ctrl-C
   loses nothing; run it again to finish (exit code 4 if it stopped at the
   cap). Drafts you have already approved, rejected or edited are never
   touched.
2. Review (§2.7), rejecting every unanswerable draft flagged as answered.
3. Refill exactly what you rejected:

   ```bash
   uv run hal9000 kb gold generate --only unanswerable --yes
   ```

**Topping up a shortfall.** `--only` fills whatever a category is short of
its quota (for example after the review, or when the summary lists a
SHORTFALL):

```bash
uv run hal9000 kb gold generate --only unanswerable --yes
```

It prints the shortfall it is filling (the quota minus what is stored and
not rejected), drafts about 50% extra to survive flagging and de-duplication,
checks each, and stores the new drafts with dev/test splits consistent with
the stored set. Each draft costs one check call (about 70 s and 3.5 cents):
a dozen missing unanswerables take about 20 minutes and $0.70.
`--only` takes several categories (`--only list,cross_document`) and
`--count N` asks for N items of each instead of the shortfall. It works on a
set generated before checkpoints existed.

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
start: set it with `SWEEP=...` (§1). It stops at `KB_EVAL_CAP_USD` and keeps its partial
results (exit code 4). If it is interrupted, resume it:

```bash
uv run hal9000 kb eval sweep --stages S1a,S1b,S2,S3,S4 --resume "$SWEEP" --yes
```

### 3.2 Answers for judge calibration (a few cents)

```bash
uv run hal9000 kb eval judge-prep "$SWEEP" --yes
```

### 3.3 Label the 20 answers (about 15 minutes)

```bash
uv run hal9000 kb serve
```

**Evaluation → Judge calibration**: label all 20. Then Ctrl-C.

### 3.4 S5 (unattended)

```bash
uv run hal9000 kb eval sweep --stages S5 --resume "$SWEEP" --yes
```

It refuses to start until the judge agrees with at least 85% of your labels.
If it refuses, send the orchestrator the agreement it printed (the rubric gets
one revision).

### 3.5 Export and share

```bash
uv run hal9000 kb eval export "$SWEEP"
ls "data/knowledge/eval/runs/$SWEEP/export.json"
```

Share that file. The
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
free -m | awk '/^Mem:/ {print ($7 >= 3072 ? "memory ok: " : "memory LOW: ") $7 " MB available"}'
df -h ~                                   # 5 GB free for images and build cache
jq --version
ss -ltnp | grep -E ':(8910|9090|3001)\b' || echo "ports free"
curl -sI https://registry-1.docker.io/v2/ | head -1   # any HTTP status: Docker Hub reachable
curl -sI https://pypi.org/simple/ | head -1
curl -sI https://registry.npmjs.org/ | head -1
curl -sI https://grafana.com/api/plugins/frser-sqlite-datasource | head -1   # Grafana's plugin, fetched at its first start
```

Pass: `aarch64`; a Docker version; Compose v2.24+; "in the docker group";
"memory ok" (at least 3 GB available with Home Assistant running: the three
containers are limited to 3 GB in all); 5 GB free on disk; a `jq` version;
"ports free"; and an HTTP status line from each of the four sites.

- No `docker compose`: install Docker's `docker-compose-plugin` package.
- No `jq` (`deploy/grafana/VERIFY.md` uses it): `sudo apt-get install -y jq`.
- "memory LOW": stop what else is running on the Pi, or tell the orchestrator.
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
sed -i '/^\(KB_UID\|KB_GID\|GRAFANA_ADMIN_PASSWORD\)=/d' .env   # any earlier (empty) lines
read -rsp "Grafana admin password: " GP; echo
printf '\n# Pi: knowledge stack (deploy/README.md)\nKB_UID=%s\nKB_GID=%s\nGRAFANA_ADMIN_PASSWORD=%s\n' \
  "$(id -u)" "$(id -g)" "$GP" >> .env
unset GP
grep -oE '^(OPENAI_API_KEY|KB_PASSCODE|GRAFANA_ADMIN_PASSWORD|KB_UID|KB_GID)=[^[:space:]#]' .env | cut -d= -f1
```

Use only letters, digits, `-` and `_` in the Grafana password. Pass: the last
command lists all five names (it prints no values; a name missing from the
list has an empty value). `.env` stays on the Pi and the Mac only; it is
git-ignored.

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
`-wal` of 0 bytes is fine). A `knowledge.db-shm` file, or a first number of 1,
means something still has the database open: find it (`lsof data/knowledge.db`),
stop it, and run the two commands again.

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

(Alternative, checksummed end to end: `uv run hal9000 kb backup` on the Mac
writes a folder `data/backups/knowledge-` plus the UTC time, holding
`knowledge.db`, `library.tar.gz` and `manifest.json`; copy it with
`rsync -av data/backups/knowledge-2026... "$PI:hal_9000/data/backups/"` (the
folder's real name), then restore it on the Pi as in §8, without `--force`
since the Pi's library is empty.)

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

`/health` should show `"status":"ok"` and the active profile that came
across with the library (the baseline from §2.4, unless the sweep changed it).
If it says `"active_profile":null`, activate the profile the orchestrator
names for now (G-K2 sets the final one):

```sh
PROFILE=pdfium.structure-768.hdr.te3s-1536    # or the name the orchestrator gives
$C exec hal-kb hal9000 kb profile activate "$PROFILE"
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
step, starting with its shell setup. It checks the Prometheus config, that Grafana logs no provisioning errors, the
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
  KB_WITH_RERANK=$r KB_IMAGE_TAG=rerank-$r $C build hal-kb
  echo "$r: $(size hal9000/hal-kb:rerank-$r) MB"
done
```

`KB_IMAGE_TAG` gives each variant its own tag (`hal9000/hal-kb:rerank-minilm`,
`hal9000/hal-kb:rerank-bge-base`), so the running stack's `:local` is not
replaced. Keep the ones S6 needs (§5); remove the rest with
`docker image rm hal9000/hal-kb:rerank-minilm` (or `-bge-base`).

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
orchestrator gives you their names, which reranker each uses, and the sweep
ID (set `SWEEP`, §1).

### 5.1 The evaluation cap on the Pi

`KB_EVAL_CAP_USD` is counted against each machine's own ledger, and the Mac's
sweep spend isn't in the Pi's. So give the Pi only what is left. On the Mac:

```bash
uv run hal9000 kb stats
```

The spend line shows `$X of $Y (evaluation)`. On the Pi, set the cap to
Y − X (for example 3.60), then apply it:

```sh
sed -i '/^KB_EVAL_CAP_USD=/d' .env
echo "KB_EVAL_CAP_USD=3.60" >> .env          # your Y - X
$C up -d
```

### 5.2 The bundle's SHA-256

The scratch-ingest test times a fresh ingestion of the bundle with an empty
cache. On the Mac:

```bash
shasum -a 256 data/knowledge/inbox/Home_Inspection.pdf | cut -c1-64
```

On the Pi, set `SHA` to that value (§1) and check the file is there:

```sh
ls -l "data/knowledge/files/$SHA"
```

### 5.3 One latency run per image

Nothing is downloaded at run time, so a finalist with a reranker must run in
an image that has that model. Group the finalists by reranker and run
`kb eval latency` once per group, in that group's image: `local` for no
reranker, `rerank-minilm` or `rerank-bge-base` otherwise. Build the reranker
images you need first, each under its own tag so `:local` is untouched (skip
any §4.9 already built):

```sh
KB_WITH_RERANK=minilm KB_IMAGE_TAG=rerank-minilm $C build hal-kb
KB_WITH_RERANK=bge-base KB_IMAGE_TAG=rerank-bge-base $C build hal-kb
```

Keep the Pi otherwise idle (no uploads) and run each group, with its own
finalists after `--profiles` (comma-separated, no spaces). For example, two
finalists without a reranker and one with minilm:

```sh
KB_IMAGE_TAG=local $C run --rm --no-deps hal-kb \
  hal9000 kb eval latency --profiles pdfium.structure-768.hdr.te3s-1536,pdfium.structure-512.hdr.te3s-1536 \
  --n 200 --end-to-end 30 --ingest-test "$SHA"
KB_IMAGE_TAG=rerank-minilm $C run --rm --no-deps hal-kb \
  hal9000 kb eval latency --profiles pdfium.structure-1024.hdr.te3s-1536 \
  --n 200 --end-to-end 30 --ingest-test "$SHA"
```

(Use the orchestrator's names; the ones above are only the shape.) Each run
prints an estimate (the live end-to-end questions and the scratch ingest's
embeddings, metered as `knowledge-eval`) and asks before spending. Pass: a
`Stored ... (profile, host ...)` line per profile and the spend. Exit code 4
means it stopped at the evaluation cap with partial results.

### 5.4 Report and export

The report goes to `docs/KNOWLEDGE_EVAL_REPORT.md` in the Pi's checkout
(aggregates only); the export to the data folder:

```sh
$C run --rm --no-deps -v "$PWD/docs:/app/docs" hal-kb hal9000 kb eval report "$SWEEP"
$C exec hal-kb hal9000 kb eval export "$SWEEP"
```

Copy both to the Mac and share them (on the Mac):

```bash
scp "$PI:hal_9000/docs/KNOWLEDGE_EVAL_REPORT.md" "$PI:hal_9000/data/knowledge/eval/runs/$SWEEP/export.json" ~/Downloads/
```

Nothing is committed on the Pi; §6 commits the final report from the Mac.

---

## 6. G-K2: choose the configuration (Pi, then the Mac)

The orchestrator presents the report's recommendation and the alternatives.
Pick one and set `NAME` to it, as the report prints it (§1). On the Pi:

```sh
$C exec hal-kb hal9000 kb eval choose "$SWEEP" --config "$NAME"
```

It builds the profile first if it isn't built (it shows the embedding cost and
asks), activates it, stores its abstain threshold, and writes the choice
(profile, reranker, query transform and chat tier) to `config/knowledge.yaml`.

Then score the chosen configuration once on the held-back test split (answers
are judged; it prints an estimate and asks), and rewrite the report with it:

```sh
$C exec hal-kb hal9000 kb eval test "$SWEEP" --config "$NAME"
$C run --rm --no-deps -v "$PWD/docs:/app/docs" hal-kb hal9000 kb eval report "$SWEEP"
```

The report now has the test-split scores against the plan §1 targets. This is
a one-time measurement: don't re-run it to chase a better number (it refuses
without `--force`, which is logged).

Apply the choice on the Pi:

1. `KB_CHAT_TIER` in `.env` overrides the chat tier `choose` wrote: delete
   that line from `.env` unless you mean to override it.
2. If the chosen configuration has a reranker, set `KB_WITH_RERANK` in `.env`
   to it (and back to `none` if not).
3. Apply: `$C up -d --build` (it only rebuilds if `KB_WITH_RERANK` changed).
4. Check: `curl -s http://localhost:8910/health; echo` shows `"status":"ok"`
   and the chosen profile.

Commit from the Mac, never on the Pi. On the Mac:

```bash
git pull
scp "$PI:hal_9000/config/knowledge.yaml" config/knowledge.yaml
scp "$PI:hal_9000/docs/KNOWLEDGE_EVAL_REPORT.md" docs/KNOWLEDGE_EVAL_REPORT.md
git diff --stat
git add config/knowledge.yaml docs/KNOWLEDGE_EVAL_REPORT.md
git commit -m "G-K2: choose $NAME"
git push
```

Then put the Pi's checkout back in step with git (the Pi's report file is
untracked there, and would block the pull):

```sh
rm docs/KNOWLEDGE_EVAL_REPORT.md
git checkout -- config/knowledge.yaml
git pull
git status --short                            # prints nothing
$C up -d
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
$C exec hal-kb hal9000 kb ingest data/knowledge/inbox   # after copying files there
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
| Grafana's SQLite panels (traces, jobs, regression, sweep) show an error, the Prometheus ones don't | Expected while `hal-kb` is stopped: Grafana reads `knowledge.db` through a read-only mount, which works only while `hal-kb` holds the database open. `$C up -d hal-kb`; the panels recover on their next refresh. |
| Grafana's SQLite panels say `unable to open database file` while `hal-kb` is running | Grafana (uid 472) can't read `data/`. `chmod o+rx data && chmod o+r data/knowledge.db*`. |
| Grafana rejects the admin password | The password is set on Grafana's first start only. Pick a new one (letters, digits, `-` and `_` only), put it on the `GRAFANA_ADMIN_PASSWORD=` line in `.env`, then: `read -rsp "New Grafana password: " GP; echo; $C exec grafana grafana cli admin reset-admin-password "$GP"; unset GP`. |
| Prometheus target `down` | `hal-kb` isn't healthy: `$C logs --tail 100 hal-kb`. |
| `hal-kb` was killed during a big upload (`docker inspect hal-kb --format '{{.State.OOMKilled}}'` says `true`) | It hit its 2 GB limit. Upload large documents one at a time; tell the orchestrator if it repeats. |
| A setting changed in `.env` has no effect | `restart` keeps the old environment. Use `$C up -d`. |

---

## 8. Backup and restore

Details and the cron line are in [`deploy/README.md`](../deploy/README.md#backups).

```sh
$C exec -T hal-kb hal9000 kb backup            # or: kb backup --to DIR (DIR under data/)
ls data/backups/knowledge-*/                   # knowledge.db  library.tar.gz  manifest.json
```

Restore with `kb restore FROM [--into DIR] [--force]`. It checks the
manifest's checksums, refuses while the service holds the library lock, and
with `--force` moves the current library aside to
`data/knowledge-replaced-<time>/` rather than deleting it:

```sh
ls data/backups/                               # pick one, then set BACKUP (§1)
BACKUP=data/backups/knowledge-20261003T031700Z
$C stop hal-kb
$C run --rm --no-deps hal-kb hal9000 kb restore "$BACKUP" --force
$C up -d hal-kb
curl -s http://localhost:8910/health; echo
$C exec hal-kb hal9000 kb stats
```

Before an upgrade, make a backup; to go back to the previous version, follow
Upgrade and Rollback in [`deploy/README.md`](../deploy/README.md#upgrade).
