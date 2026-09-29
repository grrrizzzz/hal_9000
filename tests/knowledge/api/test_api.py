"""The plan §11 API over the synthetic library (handoff KWP6 acceptance)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hal9000.knowledge.api import ROUTES, create_app
from hal9000.knowledge.auth import COOKIE
from hal9000.knowledge.ingest.jobs import create_job, update_job
from tests.knowledge.api.conftest import PASSCODE, Api, client_for, make_api
from tests.knowledge.fixtures import corpus

SHA = "ab" * 32
RUBRIC = {"correctness": "correct", "faithful": True, "citations_correct": True}
BODIES = {
    ("POST", "/api/uploads"): dict(
        files=[("files", ("a.txt", b"hello", "text/plain"))], data={"relative_path": "A/a.txt"}
    ),
    ("PATCH", "/api/documents/{id}"): dict(json={"title": "New"}),
    ("POST", "/api/chat"): dict(json={"message": "How old is the furnace?"}),
    ("POST", "/api/feedback"): dict(json={"message_id": 1, "rating": 1}),
    ("GET", "/api/search"): dict(params={"q": "furnace"}),
    ("PATCH", "/api/eval/gold/{id}"): dict(json={"status": "approved"}),
    ("PUT", "/api/eval/judge-labels/{id}"): dict(json={"human": RUBRIC}),
    ("PUT", "/api/profile"): dict(json={"name": "p"}),
}


def sse(text: str) -> list[tuple[str, dict]]:
    """``(event, data)`` pairs of an SSE body."""
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        event, data = "message", []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            out.append((event, json.loads("\n".join(data))))
    return out


def chat(client: TestClient, message: str, **body) -> list[tuple[str, dict]]:
    response = client.post("/api/chat", json={"message": message, **body})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return sse(response.text)


def done(events) -> dict:
    return next(data for name, data in events if name == "done")["answer"]


# ---------------------------------------------------------------- sessions
@pytest.mark.parametrize(("method", "path"), [r for r in ROUTES if r[1].startswith("/api/")])
def test_every_api_route_but_sign_in_needs_a_session(api: Api, method, path):
    url = path.replace("{id}", "1").replace("{sha}", SHA).replace("{n}", "1")
    response = api.client.request(method, url, **BODIES.get((method, path), {}))
    if (method, path) == ("POST", "/api/session"):
        assert response.status_code == 422  # no body: validation, not auth
        return
    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Not signed in"


def test_signing_in_sets_a_strict_http_only_cookie_for_30_days(api: Api):
    response = api.client.post("/api/session", json={"passcode": PASSCODE})
    assert response.status_code == 200 and response.json()["ok"] is True
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE}=")
    lowered = cookie.lower()
    assert "httponly" in lowered and "samesite=strict" in lowered and "path=/" in lowered
    assert "max-age=2592000" in lowered
    assert api.client.get("/api/jobs").status_code == 200
    # signing out clears it
    assert api.client.delete("/api/session").status_code == 204
    assert api.client.get("/api/jobs").status_code == 401


def test_a_forged_or_foreign_cookie_is_refused(api: Api, tmp_path: Path):
    api.client.cookies.set(COOKIE, "forged.value.here")
    assert api.client.get("/api/jobs").status_code == 401
    other = make_api(tmp_path / "other", seed=False)
    token = other.sign_in().cookies.get(COOKIE)
    other.services.close()
    api.client.cookies.set(COOKIE, token)  # signed with another library's secret
    assert api.client.get("/api/jobs").status_code == 401


def test_sign_in_is_rate_limited_to_ten_failures_a_minute(api: Api):
    for _ in range(10):
        response = api.client.post("/api/session", json={"passcode": "wrong"})
        assert response.status_code == 401
        assert response.json()["detail"] == "That passcode isn't right."
    blocked = api.client.post("/api/session", json={"passcode": PASSCODE})
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) >= 1


# -------------------------------------------------------------------- chat
def test_chat_streams_events_in_order_and_stores_the_conversation(api: Api, signed):
    events = chat(signed, "How old is the furnace?")
    names = [name for name, _ in events]
    assert names[0] == "status" and names[-1] == "done"
    assert names.index("sources") < names.index("token")
    assert names.index("verify") > max(i for i, n in enumerate(names) if n == "token")
    assert all(data["event"] == name for name, data in events)  # data is the envelope's event
    answer = done(events)
    assert answer["outcome"] == "answered" and answer["grounding"] == "verified"
    assert "2009" in answer["text"] and answer["message_id"] and answer["conversation_id"]
    assert answer["trace_id"]

    conversations = signed.get("/api/conversations").json()
    assert [c["id"] for c in conversations] == [answer["conversation_id"]]
    assert conversations[0]["title"] == "How old is the furnace?"
    detail = signed.get(f"/api/conversations/{answer['conversation_id']}").json()
    user, assistant = detail["messages"]
    assert user["role"] == "user" and user["content"] == "How old is the furnace?"
    assert assistant["id"] == answer["message_id"] and assistant["grounding"] == "verified"
    assert assistant["trace_id"] == answer["trace_id"] and assistant["citations"]
    assert assistant["issues"] == []

    trace = signed.get(f"/api/traces/{answer['trace_id']}").json()
    assert trace["question"] == "How old is the furnace?" and trace["outcome"] == "answered"
    listed = signed.get("/api/traces").json()
    assert listed[0]["id"] == answer["trace_id"] and listed[0]["question"] == trace["question"]


def test_a_forced_reset_streams_reset_between_the_drafts(signed):
    events = chat(signed, "How much was the termite repair? (force a reset)")
    names = [name for name, _ in events]
    reset = names.index("reset")
    assert "token" in names[:reset] and "token" in names[reset:]
    assert names[-2:] == ["verify", "done"]
    assert "Invoice" in done(events)["text"] and "$1,234.00" in done(events)["text"]


def test_an_unverified_answer_carries_its_issues_into_the_conversation(signed):
    answer = done(chat(signed, "When was the furnace installed? (unverified)"))
    assert answer["grounding"] == "failed" and answer["issues"]
    detail = signed.get(f"/api/conversations/{answer['conversation_id']}").json()
    assert detail["messages"][-1]["grounding"] == "failed"
    assert detail["messages"][-1]["issues"] == answer["issues"]


def test_a_follow_up_continues_the_conversation(signed):
    first = done(chat(signed, "How old is the furnace?"))
    second = done(
        chat(signed, "Where is the relief valve?", conversation_id=first["conversation_id"])
    )
    assert second["conversation_id"] == first["conversation_id"]
    trace = signed.get(f"/api/traces/{second['trace_id']}").json()
    assert "condense" in {stage["stage"] for stage in trace["stages"]}
    messages = signed.get(f"/api/conversations/{first['conversation_id']}").json()["messages"]
    assert len(messages) == 4


def test_retrieval_only_and_embedding_down_over_the_api(signed):
    events = chat(signed, "Where is the furnace? (retrieval only)")
    sources = [data for name, data in events if name == "sources"]
    assert sources[0]["retrieval_only"] and sources[0]["banner"]
    assert "token" not in [name for name, _ in events]
    assert done(events)["outcome"] == "retrieval_only"
    events = chat(signed, "Where is the furnace? (embedding down)")
    banner = next(data for name, data in events if name == "sources")["banner"]
    assert "keyword search" in banner


def test_conversations_are_private_to_their_session(api: Api, signed, tmp_path):
    answer = done(chat(signed, "How old is the furnace?"))
    other = client_for(api.client.app)
    api.sign_in(other)
    assert other.get("/api/conversations").json() == []
    assert other.get(f"/api/conversations/{answer['conversation_id']}").status_code == 404
    feedback = {"message_id": answer["message_id"], "rating": 1}
    assert other.post("/api/feedback", json=feedback).status_code == 404
    reply = other.post(
        "/api/chat", json={"message": "Hi?", "conversation_id": answer["conversation_id"]}
    )
    assert reply.status_code == 404


def test_deleting_conversations(signed):
    first = done(chat(signed, "How old is the furnace?"))
    done(chat(signed, "Where is the relief valve?"))
    assert signed.delete(f"/api/conversations/{first['conversation_id']}").status_code == 204
    assert len(signed.get("/api/conversations").json()) == 1
    assert signed.delete(f"/api/conversations/{first['conversation_id']}").status_code == 404
    assert signed.delete("/api/conversations").status_code == 204
    assert signed.get("/api/conversations").json() == []


def test_questions_are_limited_per_session(tmp_path):
    api = make_api(tmp_path, chat_limit=2)
    try:
        client = api.sign_in()
        chat(client, "How old is the furnace?")
        chat(client, "How old is the furnace?")
        refused = client.post("/api/chat", json={"message": "How old is the furnace?"})
        assert refused.status_code == 429 and refused.headers["retry-after"]
    finally:
        api.services.close()


def test_chat_without_an_active_profile_is_a_conflict(tmp_path):
    api = make_api(tmp_path, seed=False)
    try:
        client = api.sign_in()
        response = client.post("/api/chat", json={"message": "Anything?"})
        assert response.status_code == 409 and "active" in response.json()["detail"]
        assert client.get("/api/conversations").json() == []
    finally:
        api.services.close()


# ----------------------------------------------------------------- feedback
def test_feedback_is_one_row_per_message_and_adds_a_draft_gold_item(api: Api, signed):
    answer = done(chat(signed, "How old is the furnace?"))
    message = answer["message_id"]
    first = signed.post("/api/feedback", json={"message_id": message, "rating": -1}).json()
    second = signed.post(
        "/api/feedback", json={"message_id": message, "rating": -1, "comment": "Say the filter."}
    ).json()
    assert first["id"] == second["id"] and second["gold_item_id"] is None
    added = signed.post("/api/feedback", json={"message_id": message, "add_to_gold": True}).json()
    assert added["id"] == first["id"] and added["gold_item_id"]
    again = signed.post("/api/feedback", json={"message_id": message, "add_to_gold": True}).json()
    assert again["gold_item_id"] == added["gold_item_id"]  # created once

    conn = api.conn()
    try:
        rows = conn.execute("SELECT * FROM feedback").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["rating"] == -1 and rows[0]["comment"] == "Say the filter."

    item = signed.get(f"/api/eval/gold/{added['gold_item_id']}").json()
    assert item["question"] == "How old is the furnace?" and item["status"] == "draft"
    assert item["origin"] == "feedback" and item["reviewer_note"] == "Say the filter."
    assert item["reference_answer"] == answer["text"]
    assert item["evidence"] and "2009" in item["evidence"][0]["quote"]
    assert item["evidence"][0]["file_sha"] == answer["citations"][0]["file_sha"]
    drafts = signed.get("/api/eval/gold", params={"status": "draft"}).json()
    assert drafts[0]["id"] == item["id"]  # newest first, as the review queue shows them


def test_feedback_needs_a_rating_unless_it_adds_to_the_question_set(signed):
    answer = done(chat(signed, "How old is the furnace?"))
    response = signed.post("/api/feedback", json={"message_id": answer["message_id"]})
    assert response.status_code == 422


# ------------------------------------------------------------------ uploads
NESTED = {
    "House/inspection.pdf": corpus.INSPECTION.path,
    "House/Receipts/2023/invoice.pdf": corpus.INVOICE.path,
    "House/Appliances/Garage/garage_door.md": corpus.FILES_DIR / "garage_door.md",
    "House/Appliances/Heat pump/heat_pump_guide.docx": corpus.FILES_DIR / "heat_pump_guide.docx",
    "House/Yard/irrigation_notes.html": corpus.FILES_DIR / "irrigation_notes.html",
    "House/Yard/Pool/pool_log.txt": corpus.FILES_DIR / "pool_log.txt",
}
JPG = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0, 16, 0x4A, 0x46, 0x49, 0x46, 0, 1, 0xFF, 0xD9])


def _form(files: dict[str, Path | bytes]):
    parts, paths = [], []
    for rel, source in files.items():
        body = source if isinstance(source, bytes) else source.read_bytes()
        parts.append(("files", (Path(rel).name, body, "application/octet-stream")))
        paths.append(rel)
    return parts, {"relative_path": paths}


def test_uploading_a_nested_folder_with_images(api: Api, signed):
    files = {**NESTED, "House/Photos/front.jpg": JPG, "House/Photos/back.jpg": JPG}
    parts, data = _form(files)
    response = signed.post("/api/uploads", files=parts, data=data)
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    added = [r for r in results if r["status"] == "added"]
    skipped = [r for r in results if r["status"] == "skipped"]
    assert len(added) == 6 and all(r["job_id"] for r in added)
    assert len({r["job_id"] for r in added}) == 6
    assert len(skipped) == 2 and all(r["reason"] and r["job_id"] is None for r in skipped)
    by_path = {r["relative_path"]: r for r in results}
    assert by_path["House/Appliances/Heat pump/heat_pump_guide.docx"]["collection"] == (
        "House/Appliances/Heat pump"
    )
    assert by_path["House/Yard/Pool/pool_log.txt"]["collection"] == "House/Yard/Pool"
    assert by_path["House/inspection.pdf"]["collection"] == "House"

    jobs = api.run_jobs()
    assert len(jobs) == 6 and {j.status for j in jobs} == {"done"}, [j.error for j in jobs]
    listed = signed.get("/api/jobs").json()
    assert {j["id"] for j in listed} >= {r["job_id"] for r in added}
    collections = {d["collection"] for d in signed.get("/api/documents").json()}
    assert {"House", "House/Appliances/Heat pump", "House/Yard/Pool"} <= collections
    in_yard = signed.get("/api/documents", params={"collection": "House/Yard"}).json()
    assert {d["collection"] for d in in_yard} == {"House/Yard", "House/Yard/Pool"}

    # the same bytes again are a duplicate, reported, with no new job
    parts, data = _form({"Elsewhere/copy.txt": NESTED["House/Yard/Pool/pool_log.txt"]})
    again = signed.post("/api/uploads", files=parts, data=data).json()["results"][0]
    assert again["status"] == "duplicate" and "already in library" in again["reason"]


def test_uploads_need_one_path_per_file(signed):
    parts, _ = _form({"a.txt": b"synthetic text"})
    response = signed.post("/api/uploads", files=parts, data={"relative_path": ["a", "b"]})
    assert response.status_code == 422


# -------------------------------------------------------------------- files
def _upload_and_index(api: Api, client, files) -> list[dict]:
    parts, data = _form(files)
    results = client.post("/api/uploads", files=parts, data=data).json()["results"]
    api.run_jobs()
    return results


def test_pdfs_are_served_inline_with_range_support(api: Api, signed):
    (result,) = _upload_and_index(api, signed, {"Docs/inspection.pdf": corpus.INSPECTION.path})
    sha = result["sha256"]
    whole = signed.get(f"/api/files/{sha}")
    assert whole.status_code == 200 and whole.headers["content-type"] == "application/pdf"
    assert whole.headers["content-disposition"].startswith("inline")
    assert whole.headers["accept-ranges"] == "bytes"
    assert whole.content == corpus.INSPECTION.path.read_bytes()
    part = signed.get(f"/api/files/{sha}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    assert part.headers["content-range"] == f"bytes 0-99/{len(whole.content)}"
    assert part.content == whole.content[:100]


def test_an_html_original_is_only_a_sandboxed_download(api: Api, signed):
    source = corpus.FILES_DIR / "irrigation_notes.html"
    (result,) = _upload_and_index(api, signed, {"Yard/irrigation_notes.html": source})
    response = signed.get(f"/api/files/{result['sha256']}")
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.headers["content-security-policy"] == "sandbox"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-type"] == "application/octet-stream"
    assert signed.get(f"/api/files/{'cd' * 32}").status_code == 404


def test_page_text_comes_with_the_chunk_spans_on_it(api: Api, signed):
    (result,) = _upload_and_index(api, signed, {"Docs/inspection.pdf": corpus.INSPECTION.path})
    sha = result["sha256"]
    view = signed.get(f"/api/files/{sha}/pages/2").json()
    assert view["page"] == 2 and view["page_count"] == corpus.INSPECTION.page_count
    assert view["spans"]
    conn = api.conn()
    try:
        for span in view["spans"]:
            chunk = conn.execute(
                "SELECT text FROM chunks WHERE id = ?", (span["chunk_id"],)
            ).fetchone()[0]
            piece = view["text"][span["char_start"] : span["char_end"]]
            assert piece and piece in chunk  # offsets point at the chunk's text on this page
    finally:
        conn.close()
    assert signed.get(f"/api/files/{sha}/pages/999").status_code == 404


# ------------------------------------------------------------------- library
def _document(client, title: str) -> dict:
    return next(d for d in client.get("/api/documents").json() if d["document"]["title"] == title)


def test_library_lists_documents_with_collections_and_details(signed):
    docs = signed.get("/api/documents").json()
    titles = {d["document"]["title"] for d in docs}
    assert {"Synthetic home inspection", "Invoice 1001", "Garage door notes"} <= titles
    invoice = _document(signed, "Invoice 1001")
    assert invoice["collection"] == "Invoices" and invoice["status"] == "indexed"
    assert invoice["chunks"] >= 1 and invoice["pages"] == 1
    detail = signed.get(f"/api/documents/{invoice['document']['id']}").json()
    assert detail["file"]["original_name"] == "pest-invoice.txt"
    assert detail["file"]["collection"] == "Invoices"
    only = signed.get("/api/documents", params={"doc_type": "invoice"}).json()
    assert [d["document"]["title"] for d in only] == ["Invoice 1001"]
    assert signed.get("/api/documents/99999").status_code == 404


def test_a_metadata_edit_wins_and_survives_re_ingestion(api: Api, signed):
    doc = _document(signed, "Invoice 1001")["document"]
    patched = signed.patch(
        f"/api/documents/{doc['id']}",
        json={"title": "Invoice 1001, termite repair", "vendor": "Other Pest Co."},
    )
    assert patched.status_code == 200
    assert patched.json()["meta_source"] == "user" and patched.json()["vendor"] == "Other Pest Co."
    queued = [j for j in signed.get("/api/jobs").json() if j["status"] == "queued"]
    assert queued and queued[0]["file_sha"] == doc["file_sha"]
    jobs = api.run_jobs()
    assert {j.status for j in jobs} == {"done"}
    job = signed.post(f"/api/documents/{doc['id']}/reindex").json()
    assert job["status"] == "queued"
    api.run_jobs()
    after = _document(signed, "Invoice 1001, termite repair")["document"]
    assert after["vendor"] == "Other Pest Co." and after["meta_source"] == "user"
    conn = api.conn()
    try:
        header = conn.execute(
            "SELECT header FROM chunks WHERE document_id = ?", (after["id"],)
        ).fetchone()[0]
    finally:
        conn.close()
    assert "Other Pest Co." in header  # the chunks were rebuilt with the edit


def test_edits_are_validated(signed):
    doc = _document(signed, "Invoice 1001")["document"]
    assert signed.patch(f"/api/documents/{doc['id']}", json={"page_end": 5}).status_code == 422
    assert signed.patch(f"/api/documents/{doc['id']}", json={"title": " "}).status_code == 422
    assert signed.patch("/api/documents/99999", json={"title": "x"}).status_code == 404


def test_deleting_a_documents_removes_it_from_search(api: Api, signed):
    doc = _document(signed, "Invoice 1001")["document"]
    assert signed.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert "Invoice 1001" not in {
        d["document"]["title"] for d in signed.get("/api/documents").json()
    }
    assert not (api.services.settings.files_dir / doc["file_sha"]).exists()
    hits = signed.get("/api/search", params={"q": "termite ledger"}).json()["selected"]
    assert all(hit["document_id"] != doc["id"] for hit in hits)
    assert signed.delete(f"/api/documents/{doc['id']}").status_code == 404


# ---------------------------------------------------------------------- jobs
def test_a_failed_job_can_be_retried(api: Api, signed):
    conn = api.conn()
    try:
        job = create_job(conn, "ingest", file_sha=None, bus=None)
        update_job(conn, job.id, bus=None, status="failed", error="synthetic failure")
        running = create_job(conn, "ingest", file_sha=None, bus=None)
    finally:
        conn.close()
    retried = signed.post(f"/api/jobs/{job.id}/retry")
    assert retried.status_code == 200 and retried.json()["status"] == "queued"
    assert signed.post(f"/api/jobs/{running.id}/retry").status_code == 409
    assert signed.post("/api/jobs/99999/retry").status_code == 404


# -------------------------------------------------------------------- search
def test_search_returns_scores_without_answering(signed):
    result = signed.get("/api/search", params={"q": "relief valve drips", "k": 3}).json()
    assert result["selected"] and len(result["candidates"]) <= 3
    top = result["selected"][0]
    assert "relief valve" in top["text"] and top["dense_score"] is not None
    assert signed.get("/api/traces").json() == []  # search is not a question


# ---------------------------------------------------------------- evaluation
def _gold_and_run(api: Api):
    from hal9000.knowledge.contracts import EvidenceQuote, GoldItem
    from hal9000.knowledge.eval.runs import record_run
    from hal9000.knowledge.eval.store import insert_gold, insert_judge_label

    conn = api.conn()
    try:
        (item,) = insert_gold(
            conn,
            [
                GoldItem(
                    question="How old is the furnace?",
                    reference_answer="Installed in 2009.",
                    category="date",
                    evidence=[EvidenceQuote(quote="installed in 2009", page=1)],
                )
            ],
        )
        label_id = insert_judge_label(conn, item.id, "It was installed in 2009 [S1].")
        return item, label_id, record_run
    finally:
        conn.close()


def test_gold_review_and_judge_labels(api: Api, signed):
    item, label_id, _ = _gold_and_run(api)
    assert [g["id"] for g in signed.get("/api/eval/gold").json()] == [item.id]
    approved = signed.patch(f"/api/eval/gold/{item.id}", json={"status": "approved"}).json()
    assert approved["status"] == "approved"
    bad = signed.patch(f"/api/eval/gold/{item.id}", json={"evidence": []})
    assert bad.status_code == 422  # an answerable item needs evidence
    assert signed.patch("/api/eval/gold/999", json={"status": "approved"}).status_code == 404

    labels = signed.get("/api/eval/judge-labels").json()
    assert [label["id"] for label in labels] == [label_id] and labels[0]["citations"] == []
    put = signed.put(f"/api/eval/judge-labels/{label_id}", json={"human": RUBRIC}).json()
    assert put["human"]["correctness"] == "correct"
    assert signed.get(f"/api/eval/judge-labels/{label_id}").json()["human"] == put["human"]
    assert signed.get("/api/eval/judge-labels/999").status_code == 404
    assert signed.put("/api/eval/judge-labels/999", json={"human": RUBRIC}).status_code == 404


def test_eval_runs_and_paged_results(api: Api, signed, tmp_path):
    from hal9000.knowledge.contracts import EvalRun

    conn = api.conn()
    try:
        results = tmp_path / "results.jsonl"
        results.write_text(
            "\n".join(
                json.dumps(
                    {
                        "gold_item_id": i,
                        "question": f"q{i}",
                        "category": "money",
                        "recall": {"2000": 1.0},
                        "full_support": {"5": True},
                        "first_hit_rank": 1,
                        "abstain_score": 0.5,
                        "abstained": False,
                    }
                )
                for i in range(1, 8)
            )
            + "\n"
        )
        with conn:
            conn.execute(
                "INSERT INTO eval_runs (id, stage, split, config, started_at, results_path) "
                "VALUES ('run-1', 'S1a', 'dev', '{}', '2026-09-28T00:00:00+00:00', ?)",
                (str(results),),
            )
    finally:
        conn.close()
    runs = signed.get("/api/eval/runs").json()
    assert [EvalRun.model_validate(r).id for r in runs] == ["run-1"]
    assert signed.get("/api/eval/runs", params={"stage": "S2"}).json() == []
    assert signed.get("/api/eval/runs/run-1").json()["stage"] == "S1a"
    page = signed.get("/api/eval/runs/run-1/results", params={"offset": 5, "limit": 5}).json()
    assert page["total"] == 7 and [i["gold_item_id"] for i in page["items"]] == [6, 7]
    assert page["items"][0]["full_support"] is True
    assert signed.get("/api/eval/runs/nope/results").status_code == 404
    assert signed.get("/api/eval/runs/nope").status_code == 404


# -------------------------------------------------------------------- profile
def test_profile_state_and_switching_to_an_existing_profile(api: Api, signed):
    state = signed.get("/api/profile").json()
    first = state["active"]["name"]
    assert first == "pdfium.structure-64.hdr.fake-64" and state["reindex_job"] is None
    assert [p["name"] for p in state["profiles"]] == [first]
    assert signed.put("/api/profile", json={"name": "no.such-1.hdr.fake-64"}).status_code == 404

    other = "pdfium.structure-64.nohdr.fake-64"
    conn = api.conn()
    try:
        api.services.ingest.build(conn, other)  # built, not active
    finally:
        conn.close()
    switched = signed.put("/api/profile", json={"name": other}).json()
    assert switched["active"]["name"] == first  # the old one serves until the job runs
    assert switched["reindex_job"]["kind"] == "reindex"
    assert switched["reindex_job"]["status"] == "queued"
    jobs = api.run_jobs()
    assert [j.status for j in jobs] == ["done"]
    assert signed.get("/api/profile").json()["active"]["name"] == other
    assert done(chat(signed, "How old is the furnace?"))["outcome"] == "answered"

    conn = api.conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO profiles (name, parser, chunker, size, overlap, header, embed_model, "
                "dimensions, status, active, created_at) VALUES ('pdfium.fixed-128.hdr.fake-64', "
                "'pdfium', 'fixed', 128, 0, 1, 'fake/embedding-64', 64, 'building', 0, 'now')"
            )
    finally:
        conn.close()
    busy = signed.put("/api/profile", json={"name": "pdfium.fixed-128.hdr.fake-64"})
    assert busy.status_code == 409


# ------------------------------------------------------------------------ ops
def test_health_is_open_and_writes_the_health_file(api: Api, signed):
    done(chat(signed, "How old is the furnace?"))
    health = TestClient(api.client.app).get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok" and body["active_profile"] and body["retrieval_only"] is False
    written = json.loads((api.services.settings.data_dir / "knowledge_health.json").read_text())
    assert written["last_successful_question_at"] and written["last_ingestion_at"]
    assert written["month_spend_usd"] == 0.0 and "furnace" not in json.dumps(written)


def test_metrics_are_open_and_count_questions(api: Api, signed):
    done(chat(signed, "How old is the furnace?"))
    text = TestClient(api.client.app).get("/metrics").text
    assert 'kb_questions_total{outcome="answered"}' in text
    assert "furnace" not in text


def test_the_ui_is_served_with_a_history_fallback(api: Api, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>HAL</title>")
    (dist / "assets" / "app.js").write_text("console.log('synthetic')")
    client = client_for(create_app(api.services, web_dist=dist))
    assert client.get("/").text.startswith("<!doctype html>")
    assert client.get("/traces/0123abcd").text.startswith("<!doctype html>")
    assert client.get("/c/12").text.startswith("<!doctype html>")
    asset = client.get("/assets/app.js")
    assert asset.status_code == 200 and "synthetic" in asset.text
    assert "immutable" in asset.headers["cache-control"]
    assert client.get("/api/nope").status_code == 401  # not the UI, and not before sign-in
    api.sign_in(client)
    unknown = client.get("/api/nope")
    assert unknown.status_code == 404 and unknown.headers["content-type"] == "application/json"
    assert client.get("/../../etc/passwd").text.startswith("<!doctype html>")
    for raw in (b"//etc/passwd", b"/assets/../../../etc/passwd", b"/%2Fetc%2Fpasswd"):
        status, body = _raw_get(client.app, raw)
        assert status in (200, 404) and b"root:" not in body, raw
    assert client.get("/health").headers["content-type"] == "application/json"


# ------------------------------------------------------------------ services
def test_the_real_services_are_metered_and_report_the_cap(tmp_path, monkeypatch):
    from hal9000.knowledge.api.services import _meter, build_services
    from tests.knowledge.answer.conftest import make_settings

    monkeypatch.setenv("HAL_DATA_DIR", str(tmp_path / "hal"))
    settings = make_settings(tmp_path, monthly_cap_usd=0.5)
    meter = _meter(settings)
    assert meter.monthly_caps == {"knowledge": 0.5, "knowledge-eval": 10.0}
    services = build_services(settings, meter=meter, start_worker=False)
    try:
        assert services.meter is meter and services.answerer.deps.meter is meter
        assert services.ingest.meter is meter and services.fake is False
        assert services.month_spend() == 0.0 and services.retrieval_only() is False
        meter.db.add_ledger("earlier", "knowledge", {"model": "openai/gpt-5-mini"}, 0.75)
        assert services.retrieval_only() is True
        client = TestClient(create_app(services, web_dist=False))
        body = client.get("/health").json()
        assert body["status"] == "degraded" and body["retrieval_only"] is True
        assert body["month_spend_usd"] == 0.75 and body["month_cap_usd"] == 0.5
    finally:
        services.close()


def _raw_get(app, raw_path: bytes) -> tuple[int, bytes]:
    """GET with an exact raw path (an HTTP client would normalise ``//etc/passwd``)."""
    import asyncio
    from urllib.parse import unquote

    sent: dict = {"body": b""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": unquote(raw_path.decode()),
        "raw_path": raw_path,
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test")],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    asyncio.run(app(scope, receive, send))
    return sent["status"], sent["body"]
