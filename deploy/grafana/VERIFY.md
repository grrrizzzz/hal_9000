# Verifying Prometheus and Grafana on the Pi

Mike runs these steps on the Pi after the first `docker compose up` and after
any change under `deploy/grafana/` or `deploy/prometheus/`. KWP11 includes this
section in `docs/KNOWLEDGE_RUNBOOK.md`. The build workspace can't reach a
container registry, so none of this has run in a container yet. What could
run without containers has: `tests/knowledge/obs/test_live.py` checks every
panel query against a live `kb serve --fake` and its `knowledge.db`, and
`tests/knowledge/obs/test_prometheus_real.py` (opt-in, with a Prometheus 3.5
release in `KB_PROMETHEUS_DIR`) runs `promtool check config` and evaluates
every dashboard and alert expression in a real Prometheus scraping
`kb serve --fake`. Grafana itself, the SQLite plugin and the mounts are what
these steps add.

It takes about 10 minutes. Report each step as pass or fail with the output
asked for. Never paste question text, answers or document names.

## What the compose file must provide (KWP11)

- Services `hal-kb`, `prometheus` and `grafana`, on ports 8910, 9090 and 3001
  by default (the commands below use the defaults; use your `.env` values if
  you changed them).
- `prometheus`: `deploy/prometheus/prometheus.yml` mounted at
  `/etc/prometheus/prometheus.yml`, and `--storage.tsdb.retention.time=90d`.
- `grafana`: `deploy/grafana/grafana.ini` at `/etc/grafana/grafana.ini`,
  `deploy/grafana/provisioning` at `/etc/grafana/provisioning`,
  `deploy/grafana/dashboards` at `/etc/grafana/dashboards`, and `./data` at
  `/data` **read-only** (`./data:/data:ro`). `GF_SECURITY_ADMIN_PASSWORD` comes
  from `.env`.

Why read-only: `knowledge.db` is in WAL mode. While `hal-kb` runs, its job
worker keeps a connection open, so `knowledge.db-wal` and `-shm` exist and a
read-only reader sees every row. With a read-write mount, a Grafana query made
while `hal-kb` is stopped would create those two files as Grafana's user (uid
472), and `hal-kb` could not write to its database when it started again.

## 0. Set up the shell

From the repository root on the Pi:

```sh
C="docker compose -f deploy/docker/compose.knowledge.yaml"
PW=$(grep '^GF_SECURITY_ADMIN_PASSWORD=' .env | cut -d= -f2-)
G="http://localhost:3001"
```

## 1. Prometheus configuration (promtool)

```sh
$C run --rm --no-deps --entrypoint promtool prometheus \
  check config /etc/prometheus/prometheus.yml
```

Pass: `SUCCESS: /etc/prometheus/prometheus.yml is valid prometheus config file syntax`.
If the compose file isn't ready yet, run the same check straight from the image:

```sh
docker run --rm -v "$PWD/deploy/prometheus:/etc/prometheus:ro" \
  --entrypoint promtool prom/prometheus check config /etc/prometheus/prometheus.yml
```

## 2. Start the stack and check Prometheus scrapes hal-kb

```sh
$C up -d hal-kb prometheus grafana
sleep 60
curl -s http://localhost:9090/api/v1/targets \
  | jq -r '.data.activeTargets[] | [.labels.job, .health, .lastError] | @tsv'
```

Pass: one line `hal-kb  up` with an empty error.

## 3. Grafana logs: no provisioning errors

```sh
$C logs grafana 2>&1 | grep -E 'level=(error|crit)'
$C logs grafana 2>&1 | grep -iE 'provision' | grep -iE 'fail|error|invalid'
```

Pass: both print nothing. If anything prints, paste those lines.

## 4. The SQLite plugin is installed and pinned

```sh
$C logs grafana 2>&1 | grep -i 'frser-sqlite-datasource' | head -5
curl -su "admin:$PW" "$G/api/plugins/frser-sqlite-datasource/settings" | jq -r .info.version
curl -su "admin:$PW" "$G/api/datasources/uid/kb-sqlite/health" | jq -r .status
curl -su "admin:$PW" "$G/api/datasources/uid/kb-prometheus/health" | jq -r .status
```

Pass: the version is `4.0.6` (from `preinstall_sync` in `grafana.ini`) and
both datasources report `OK`. The plugin is installed before provisioning
runs, so a missing plugin shows up as a datasource error in step 3.

## 5. Anonymous access is off and the mount is read-only

```sh
curl -s -o /dev/null -w '%{http_code}\n' "$G/api/search"
$C exec grafana sh -c 'touch /data/.probe'
ls -ln data/knowledge.db*
```

Pass: `401`; the `touch` fails with `Read-only file system`; `knowledge.db`,
`knowledge.db-wal` and `knowledge.db-shm` are all owned by hal-kb's user, not
by uid 472.

## 6. The three dashboards are provisioned

```sh
curl -su "admin:$PW" "$G/api/search?type=dash-db&folderUIDs=kb" | jq -r '.[].title'
```

Pass: `Knowledge: Ingestion`, `Knowledge: Live` and `Knowledge: Quality`.

## 7. A fresh trace reaches Grafana within 60 seconds

Note the time, then ask one question in the web app (`http://<pi-address>:8910`):

```sh
ASKED=$(date -u +%Y-%m-%dT%H:%M:%S)
```

Then, straight away:

```sh
Q='{"from":"now-1h","to":"now","queries":[{"refId":"A","datasource":{"uid":"kb-sqlite"},"queryType":"table","timeColumns":[],"queryText":"SELECT MAX(created_at) AS newest FROM traces WHERE kind = '"'"'question'"'"'"}]}'
for i in $(seq 1 12); do
  NEWEST=$(curl -su "admin:$PW" -H 'Content-Type: application/json' "$G/api/ds/query" -d "$Q" \
    | jq -r '.results.A.frames[0].data.values[0][0]')
  if [[ "$NEWEST" > "$ASKED" ]]; then echo "visible after about $((i * 5)) s"; break; fi
  sleep 5
done
```

Pass: `visible after about N s` with N at most 60. Then open
`$G/d/kb-live?var-kb_url=http://<pi-address>:8910`. Within 60 seconds (the
dashboard refreshes every 30 s) the question is the top row of **Latest 50
traces**, with time, outcome, latency and cost and no question text. Click its
ID: the Query Inspector opens in a new tab (after the passcode if needed).
Bookmark this dashboard URL: `kb_url` defaults to `http://hal-kb.local:8910`,
which only works if the Pi answers to that name.

## 8. The dashboards render with data

After a few questions (one that the documents can't answer, so it abstains)
and one upload:

- **Knowledge: Live**: every panel draws without an error triangle.
  Questions per hour, end-to-end latency, the retrieval stage latencies,
  tokens, the spend-vs-cap gauge and the traces table have data. Grounding
  failures, errors and feedback can be empty.
- **Knowledge: Ingestion**: files by status, pages, parse and embed time per
  page, chunks, documents and database size have data. Failed jobs can be
  empty.
- **Knowledge: Quality**: the regression panels have data once
  `hal9000 kb eval regression` has run, and the sweep panels once the Mac's
  sweep results are in `knowledge.db`. Before that they are empty, not in error.

Pass: no panel shows an error. Report which panels are empty.

## 9. The alert rules are loaded

```sh
curl -su "admin:$PW" "$G/api/v1/provisioning/alert-rules" | jq -r '.[].title'
sleep 120
curl -su "admin:$PW" "$G/api/prometheus/grafana/api/v1/rules" \
  | jq -r '.data.groups[].rules[] | [.name, .health, .state] | @tsv'
```

Pass: four rules (error rate over 10%, answer p95 over 15 s, month-to-date
spend over 80% of the cap, regression recall down after an ingestion), each
with health `ok` and state `inactive` (or `normal`).

Optional Telegram: mount `deploy/grafana/optional/telegram.yaml` into
`/etc/grafana/provisioning/alerting/`, pass `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` to the grafana container, restart it, repeat step 3, then
use **Test** on the `hal-telegram` contact point in Grafana and check that the
message arrives.

## If a step fails

- Datasource `kb-sqlite` reports `attempt to write a readonly database`:
  `hal-kb` isn't running (step 2), so there are no `-wal`/`-shm` files.
- `unable to open database file`: uid 472 can't read `data/` or
  `data/knowledge.db`. Make both readable by others (`chmod o+rx data`,
  `chmod o+r data/knowledge.db*`).
- The plugin isn't installed: the Pi couldn't reach grafana.com when Grafana
  started. Paste the plugin lines from `$C logs grafana`.
- `-wal` or `-shm` owned by uid 472: the mount was read-write, and `hal-kb`
  can't write its database. Stop the stack, make the mount `:ro`, give the two
  files back to hal-kb's user (`sudo chown <uid>:<gid> data/knowledge.db-wal
  data/knowledge.db-shm`, with the owner of `data/knowledge.db`), and start again.
