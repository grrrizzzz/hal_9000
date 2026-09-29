"""The ingestion pipeline: upload → screen → store → parse → segment → clean → chunk → embed → index
(plan §5.1 steps 2–10).

:meth:`IngestPipeline.submit` runs the cheap, synchronous part of an upload:

2. **screen** the bytes: an allow-listed extension (PDF, DOCX, TXT, MD, HTML)
   whose content matches (magic bytes), at most ``ingest.max_upload_bytes``
   (100 MB), and SHA-256 deduplication: the same bytes again are
   ``duplicate`` ("already in library"); a different file under the same
   relative path is stored as a new version (``files.version_of``);
3. **store** the original at ``data/knowledge/files/<sha256>`` (no
   user-controlled path on disk) and record it in ``files``;

then queues an ``ingest`` job. The job (:meth:`IngestPipeline.process`, on the
one worker thread of :mod:`.jobs`) does the rest:

4. **parse** with KWP1's :func:`parse_file` (child process, time limit);
5./6. **segment** the raw pages into documents (LLM labels through the
   persistent cache, or heuristics with ``--no-llm``; documents the user
   edited are kept as they are), then **clean** the pages with those document
   boundaries (:func:`clean_for_documents`, the one adapter to KWP1's cleaning);
7. **chunk** every document with the active profile (and any extra profiles,
   for evaluation);
8. **embed** the searchable chunks through the cached, metered service;
9. **index**: documents, pages, chunks and the profile's FTS5 and ``vec0``
   rows in one transaction, so a crash never leaves half a file; then the
   metrics;
10. the **regression** hook (a no-op until KWP5's scorer plugs in).

Every step runs inside the ``stage()`` timer with its plan name. Re-running a
file (a retried job) replaces its earlier rows and pays for nothing twice:
embeddings and LLM labels are cached.

When no profile is active yet, ``active_profile`` from
``config/knowledge.yaml`` is built from the stored pages and activated first.
When none is configured either, files are stored, parsed, segmented and
cleaned, but not chunked: a profile build indexes them later.
"""

from __future__ import annotations

import io
import os
import shutil
import sqlite3
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from hal9000.knowledge.config import AGENT, KnowledgeFile, load_knowledge_file
from hal9000.knowledge.contracts import (
    DocumentMeta,
    FileRecord,
    IndexProfile,
    Job,
    Page,
    UploadResult,
)
from hal9000.knowledge.index.embedding import EmbeddingService, EmbedStats
from hal9000.knowledge.index.library import (
    DbSegmentCache,
    delete_file_index,
    documents_for,
    file_has_index,
    get_file,
    insert_documents,
    insert_file,
    latest_version,
    load_pages,
    save_pages,
    user_documents,
)
from hal9000.knowledge.index.profiles import (
    BuildReport,
    active_profile,
    build_profile,
    configured_profile,
    get_profile,
    parser_key,
)
from hal9000.knowledge.index.writer import (
    PreparedDocument,
    chunk_documents,
    embed_chunks,
    write_chunks,
)
from hal9000.knowledge.ingest.clean import CleanResult, clean_pages
from hal9000.knowledge.ingest.jobs import (
    BUS,
    JobWorker,
    ProgressBus,
    Reporter,
    create_job,
    job_params,
    retry_job,
)
from hal9000.knowledge.ingest.parsers import ParsedFile, ParseError, get_parser, parse_file
from hal9000.knowledge.ingest.segment import segment_file
from hal9000.knowledge.obs import metrics
from hal9000.knowledge.obs.timing import Stage, stage

__all__ = [
    "ALLOWED_TYPES",
    "IngestOutcome",
    "IngestPipeline",
    "Screened",
    "clean_for_documents",
    "default_collection",
    "normalize_relative_path",
    "no_regression",
    "screen",
]

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#: plan §5.1 step 2: the allow-list, by extension.
ALLOWED_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": DOCX_MIME,
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
}
_SNIFF = 8192

Regression = Callable[[sqlite3.Connection, IndexProfile], None]


def no_regression(conn: sqlite3.Connection, profile: IndexProfile) -> None:
    """The post-ingest regression check until KWP5's retrieval scorer exists."""


# ------------------------------------------------------------------- screen
@dataclass(frozen=True)
class Screened:
    """The verdict on one upload. ``status`` is ``ok``, ``skipped`` or ``failed``."""

    status: str
    relative_path: str
    reason: str | None = None
    sha256: str | None = None
    mime_type: str | None = None
    size: int = 0
    data: bytes | None = field(default=None, repr=False)


def normalize_relative_path(path: str) -> str:
    """A clean POSIX relative path (``Appliances/Furnace/manual.pdf``); never escapes upward."""
    parts = [
        part
        for part in PurePosixPath(str(path).replace("\\", "/")).parts
        if part not in ("", ".", "..", "/")
    ]
    if not parts:
        raise ValueError(f"not a file path: {path!r}")
    return "/".join(parts)


def default_collection(relative_path: str) -> str:
    """The folder part of a relative path is its collection (plan §5.1 step 1)."""
    parent = PurePosixPath(relative_path).parent.as_posix()
    return "" if parent == "." else parent


def _content_matches(suffix: str, data: bytes) -> bool:
    head = data[:_SNIFF]
    if suffix == ".pdf":
        return b"%PDF-" in data[:1024]
    if suffix == ".docx":
        if not data.startswith(b"PK\x03\x04"):
            return False
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                return "word/document.xml" in archive.namelist()
        except zipfile.BadZipFile:
            return False
    if b"\x00" in head:  # text formats: no NUL bytes
        return False
    if suffix in (".html", ".htm"):
        return b"<" in head
    return True


def screen(source: bytes | Path, relative_path: str, *, max_bytes: int) -> Screened:
    """Check one upload against the allow-list, content type and size (``stage("screen")``)."""
    with stage(Stage.SCREEN):
        import hashlib

        relative_path = normalize_relative_path(relative_path)
        suffix = PurePosixPath(relative_path).suffix.lower()
        mime = ALLOWED_TYPES.get(suffix)
        if mime is None:
            return Screened(
                "skipped",
                relative_path,
                f"{suffix or 'files without an extension'} is not a supported type "
                "(PDF, DOCX, TXT, MD, HTML)",
            )
        size = source.stat().st_size if isinstance(source, Path) else len(source)
        if size > max_bytes:
            return Screened(
                "failed",
                relative_path,
                f"too large: {size / 1e6:.1f} MB (the limit is {max_bytes / 1e6:.0f} MB)",
                size=size,
            )
        if size == 0:
            return Screened("failed", relative_path, "the file is empty", size=0)
        data = source.read_bytes() if isinstance(source, Path) else bytes(source)
        if not _content_matches(suffix, data):
            return Screened(
                "failed",
                relative_path,
                f"the content is not a {suffix[1:].upper()} file",
                size=size,
            )
        return Screened(
            "ok",
            relative_path,
            sha256=hashlib.sha256(data).hexdigest(),
            mime_type=mime,
            size=size,
            data=data,
        )


def _store_original(files_dir: Path, sha: str, data: bytes) -> Path:
    """Write ``files_dir/<sha>`` atomically (a crash never leaves a partial original)."""
    target = files_dir / sha
    if target.is_file() and target.stat().st_size == len(data):
        return target
    files_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=files_dir, prefix=".upload-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


# ------------------------------------------------------------------- clean
def clean_for_documents(
    parsed: ParsedFile,
    documents: Sequence[DocumentMeta],
    *,
    file_sha: str,
    repeat_fraction: float | None = None,
) -> CleanResult:
    """Clean a file's raw pages, given its segmented documents (parse → segment → clean → chunk).

    The one adapter between the pipeline and KWP1's cleaning.
    """
    # TODO(KWP1b): pass `documents` to clean_pages once it takes segmented document
    # boundaries (kwp/1b-house), so running lines are detected per sub-document. Until
    # then clean_pages counts repeated lines across the whole file.
    del documents
    return clean_pages(
        parsed.pages,
        file_sha=file_sha,
        parser=parsed.parser,
        repeat_fraction=repeat_fraction,
        pseudo_pages=parsed.pseudo_pages,
    )


# ----------------------------------------------------------------- pipeline
@dataclass
class IngestOutcome:
    """What one file's ingestion did (the CLI prints it; the job detail summarises it)."""

    file_sha: str
    relative_path: str
    documents: int = 0
    pages: int = 0
    pages_without_text: int = 0
    chunks: dict[str, int] = field(default_factory=dict)
    embed: EmbedStats = field(default_factory=EmbedStats)
    segmentation: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        chunks = sum(self.chunks.values())
        text = f"{self.documents} document(s), {self.pages} page(s), {chunks} chunk(s)"
        if self.embed.requested:
            text += f", {self.embed.embedded} embedded, {self.embed.cache_hits} cached"
        if not self.chunks:
            text += " (no active index profile: not chunked yet)"
        return text


@dataclass
class IngestPipeline:
    """Plan §5.1 steps 2–10, configured once per process (the service, a CLI run, a test).

    Args:
        files_dir: where originals are stored (``data/knowledge/files``).
        embed_fn: the embedding callable (default: :func:`hal9000.llm.embed`, or the
            fake embedder for ``fake/*`` models).
        meter: meters every paid call (rule K9); ``None`` only for the fake embedder.
        generate: the metered segmentation LLM, ``(prompt, tier) -> str``.
        use_llm: default for jobs that don't say (``--no-llm`` sets it per job).
        extra_profiles: also index into these (names or profiles), for evaluation.
        regression: the post-ingest hook, called with the active profile.
        isolate_parse: parse in a child process with the time limit (the default).
    """

    files_dir: Path
    embed_fn: Callable[..., Any] | None = None
    meter: Any | None = None
    agent: str = AGENT
    generate: Callable[..., str] | None = None
    use_llm: bool = True
    extra_profiles: Sequence[str | IndexProfile] = ()
    regression: Regression = no_regression
    isolate_parse: bool = True
    parse_timeout_s: float | None = None
    knowledge_file: KnowledgeFile | None = None
    bus: ProgressBus | None = BUS
    outcomes: dict[int, IngestOutcome] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.files_dir = Path(self.files_dir)
        if self.knowledge_file is None:
            self.knowledge_file = load_knowledge_file()

    # -- services ----------------------------------------------------------------
    def embeddings(self, conn: sqlite3.Connection) -> EmbeddingService:
        return EmbeddingService(conn, embed_fn=self.embed_fn, meter=self.meter, agent=self.agent)

    @property
    def max_bytes(self) -> int:
        assert self.knowledge_file is not None
        return self.knowledge_file.ingest.max_upload_bytes

    # -- 2, 3: screen and store ----------------------------------------------------
    def submit(
        self,
        conn: sqlite3.Connection,
        source: bytes | Path,
        relative_path: str,
        *,
        collection: str | None = None,
        use_llm: bool | None = None,
    ) -> UploadResult:
        """Screen and store one upload, then queue its ingestion (unless it is a duplicate)."""
        try:
            screened = screen(source, relative_path, max_bytes=self.max_bytes)
        except ValueError as error:
            metrics.record_ingest_file("skipped")
            return UploadResult(
                relative_path=str(relative_path), status="skipped", reason=str(error)
            )
        rel = screened.relative_path
        coll = collection if collection is not None else default_collection(rel)
        if screened.status != "ok":
            metrics.record_ingest_file(screened.status)
            return UploadResult(
                relative_path=rel, status=screened.status, collection=coll, reason=screened.reason
            )
        sha = screened.sha256
        assert sha is not None and screened.data is not None
        params = {
            "relative_path": rel,
            "collection": coll,
            "use_llm": self.use_llm if use_llm is None else bool(use_llm),
        }
        existing = get_file(conn, sha)
        if existing is not None:
            return self._resubmit(conn, existing, screened, params)
        with stage(Stage.STORE):
            _store_original(self.files_dir, sha, screened.data)
            record = FileRecord(
                sha256=sha,
                original_name=PurePosixPath(rel).name,
                relative_path=rel,
                collection=coll,
                mime_type=screened.mime_type or "application/octet-stream",
                bytes=screened.size,
                version_of=latest_version(conn, rel),
            )
            with conn:
                insert_file(conn, record)
        job = create_job(conn, "ingest", file_sha=sha, params=params, bus=self.bus)
        reason = "new version of an earlier file" if record.version_of else None
        return UploadResult(
            relative_path=rel, status="added", sha256=sha, collection=coll, job_id=job.id,
            reason=reason,
        )  # fmt: skip

    def _resubmit(
        self, conn: sqlite3.Connection, existing: FileRecord, screened: Screened, params: dict
    ) -> UploadResult:
        sha = existing.sha256
        rel = screened.relative_path
        active = conn.execute(
            "SELECT id FROM jobs WHERE file_sha = ? AND kind = 'ingest' AND status IN "
            "('queued', 'running') ORDER BY id DESC LIMIT 1",
            (sha,),
        ).fetchone()
        if file_has_index(conn, sha) or active is not None:
            metrics.record_ingest_file("duplicate")
            where = existing.relative_path
            return UploadResult(
                relative_path=rel, status="duplicate", sha256=sha,
                collection=existing.collection, job_id=active[0] if active else None,
                reason=f"already in library as {where}",
            )  # fmt: skip
        # stored before, but its ingestion never finished: run it again
        assert screened.data is not None
        _store_original(self.files_dir, sha, screened.data)
        failed = conn.execute(
            "SELECT id FROM jobs WHERE file_sha = ? AND kind = 'ingest' AND status = 'failed' "
            "ORDER BY id DESC LIMIT 1",
            (sha,),
        ).fetchone()
        if failed is not None:
            job = retry_job(conn, failed[0], bus=self.bus)
        else:
            job = create_job(conn, "ingest", file_sha=sha, params=params, bus=self.bus)
        return UploadResult(
            relative_path=rel, status="added", sha256=sha, collection=existing.collection,
            job_id=job.id, reason="retrying an earlier ingestion that did not finish",
        )  # fmt: skip

    # -- profiles ------------------------------------------------------------------
    def pages_for(self, conn: sqlite3.Connection, file: FileRecord, parser: str) -> list[Page]:
        """Stored cleaned pages of ``file`` from ``parser``; parsed and cleaned now if missing."""
        pages = load_pages(conn, file.sha256, parser)
        if pages:
            return pages
        parsed = self._parse(file, parser)
        cleaned = clean_for_documents(
            parsed, documents_for(conn, file.sha256), file_sha=file.sha256
        )
        with conn:
            save_pages(conn, cleaned.pages)
        return cleaned.pages

    def build(
        self,
        conn: sqlite3.Connection,
        profile: str | IndexProfile,
        *,
        activate: bool = False,
        rebuild: bool = False,
        report: Reporter | None = None,
    ) -> BuildReport:
        """Build a profile from stored pages and cached embeddings (see :func:`build_profile`)."""
        target = configured_profile(profile) if isinstance(profile, str) else profile
        assert target is not None
        return build_profile(
            conn,
            target,
            self.embeddings(conn),
            pages_for=self.pages_for,
            activate=activate,
            rebuild=rebuild,
            progress=report,
        )

    def enqueue_build(
        self, conn: sqlite3.Connection, name: str, *, activate: bool = True, rebuild: bool = False
    ) -> Job:
        """Queue a background profile build (``reindex`` job); the old profile keeps serving."""
        configured_profile(name)  # validates the name now, not in the worker
        return create_job(
            conn,
            "reindex",
            params={"profile": name, "activate": activate, "rebuild": rebuild},
            detail=f"queued: build {name}",
            bus=self.bus,
        )

    def target_profiles(self, conn: sqlite3.Connection) -> list[IndexProfile]:
        """The active profile (built from config first if needed), then the extra profiles."""
        assert self.knowledge_file is not None
        out: list[IndexProfile] = []
        active = active_profile(conn)
        if active is None and self.knowledge_file.active_profile:
            active = self.build(conn, self.knowledge_file.active_profile, activate=True).profile
        if active is not None:
            out.append(active)
        for extra in self.extra_profiles:
            name = extra if isinstance(extra, str) else extra.name
            if any(p.name == name for p in out):
                continue
            stored = get_profile(conn, name)
            if (
                stored is None
                or conn.execute("SELECT status FROM profiles WHERE name = ?", (name,)).fetchone()[0]
                != "ready"
            ):
                stored = self.build(conn, extra).profile
            out.append(stored)
        return out

    # -- 4-10: one file ------------------------------------------------------------
    def _parse(self, file: FileRecord, parser: str) -> ParsedFile:
        """Parse the stored original, seen under its original name.

        Originals are stored as bare hashes, but parsers decide details by the
        suffix (Markdown headings), and error messages should name the file, so
        the parser reads it through a link named like the upload.
        """
        assert self.knowledge_file is not None
        ingest = self.knowledge_file.ingest
        timeout = self.parse_timeout_s or ingest.parse_timeout_seconds
        stored = self.files_dir / file.sha256
        if not stored.is_file():
            raise ParseError(
                f"Could not parse {file.original_name}: the stored original is missing."
            )
        name = PurePosixPath(file.original_name).name
        if not name or len(name) > 200 or name.startswith("."):
            name = "document" + PurePosixPath(file.original_name).suffix.lower()
        with tempfile.TemporaryDirectory(prefix="kb-parse-") as tmp:
            view = Path(tmp) / name
            try:
                view.symlink_to(stored.resolve())
            except OSError:  # pragma: no cover - filesystems without symlinks
                shutil.copyfile(stored, view)
            return parse_file(
                view,
                parser=get_parser(parser, max_chars=ingest.pseudo_page_chars),
                timeout_s=timeout,
                isolate=self.isolate_parse,
                max_chars=ingest.pseudo_page_chars,
            )

    def process(
        self,
        conn: sqlite3.Connection,
        sha: str,
        *,
        use_llm: bool | None = None,
        report: Reporter | None = None,
    ) -> IngestOutcome:
        """Parse, segment, clean, chunk, embed and index one stored file (idempotent)."""
        note = report or (lambda progress, detail: None)
        file = get_file(conn, sha)
        if file is None:
            raise LookupError(f"no stored file {sha}")
        use_llm = self.use_llm if use_llm is None else use_llm
        outcome = IngestOutcome(file_sha=sha, relative_path=file.relative_path)
        profiles = self.target_profiles(conn)
        keys = list(dict.fromkeys(parser_key(file, p.parser) for p in profiles)) or [
            parser_key(file, "pdfium")
        ]

        note(0.05, "parsing")
        parsed = {key: self._parse(file, key) for key in keys}
        primary = parsed[keys[0]]

        note(0.2, "segmenting")
        kept = user_documents(conn, sha)
        if kept:
            with stage(Stage.SEGMENT):  # the user's edits win; nothing to decide
                documents = documents_for(conn, sha)
            outcome.segmentation = "user"
        else:
            result = segment_file(
                primary.pages,
                file_sha=sha,
                generate=self.generate if use_llm else None,
                use_llm=use_llm,
                cache=DbSegmentCache(conn),
                name=file.original_name,
                pseudo_pages=primary.pseudo_pages,
            )
            documents = result.documents
            outcome.segmentation = result.source
            outcome.warnings.extend(result.warnings)

        note(0.3, "cleaning")
        cleaned = {key: clean_for_documents(parsed[key], documents, file_sha=sha) for key in keys}

        note(0.4, "chunking")
        per_profile: list[tuple[IndexProfile, list[PreparedDocument], list]] = []
        for profile in profiles:
            clean = cleaned[parser_key(file, profile.parser)]
            prepared = [
                PreparedDocument(doc, clean.document(doc.page_start, doc.page_end), file.collection)
                for doc in documents
            ]
            per_profile.append((profile, prepared, chunk_documents(prepared, profile)))

        vectors = []
        embeddings = self.embeddings(conn)
        for number, (profile, _, chunked) in enumerate(per_profile):
            share = 0.45 / len(per_profile)
            start = 0.45 + number * share

            def progress(done: int, total: int, start=start, share=share, name=profile.name):
                note(start + share * done / max(total, 1), f"embedding {done}/{total} ({name})")

            note(start, f"embedding ({profile.name})")
            vectors.append(embed_chunks(embeddings, profile, chunked, progress=progress))
            outcome.embed.add(embeddings.last)

        note(0.9, "indexing")
        with stage(Stage.INDEX), conn:
            delete_file_index(conn, sha, keep_user_documents=True)
            stored: list[DocumentMeta] = []
            for doc in documents:
                if doc.meta_source == "user" and doc.id is not None:
                    stored.append(doc)
                else:
                    stored.extend(insert_documents(conn, [doc.model_copy(update={"id": None})]))
            for key in keys:
                save_pages(conn, cleaned[key].pages)
            for (profile, prepared, chunked), vecs in zip(per_profile, vectors):
                with_ids = [
                    PreparedDocument(meta, doc.paged, doc.collection)
                    for meta, doc in zip(stored, prepared)
                ]
                outcome.chunks[profile.name] = write_chunks(conn, profile, with_ids, chunked, vecs)
        outcome.documents = len(documents)
        outcome.pages = len(primary.pages)
        outcome.pages_without_text = sum(1 for page in primary.pages if not page.has_text_layer)
        metrics.record_ingest_pages(
            with_text_layer=outcome.pages - outcome.pages_without_text,
            without_text_layer=outcome.pages_without_text,
        )

        if profiles:
            note(0.97, "regression check")
            try:
                with stage(Stage.REGRESSION):
                    self.regression(conn, profiles[0])
            except Exception as error:  # a failed check never undoes an ingestion
                outcome.warnings.append(f"regression check failed: {type(error).__name__}")
        return outcome

    # -- jobs ----------------------------------------------------------------------
    def _ingest_job(self, conn: sqlite3.Connection, job: Job, report: Reporter) -> str:
        from hal9000.knowledge.obs.traces import observe

        assert job.id is not None and job.file_sha is not None
        params = job_params(conn, job.id)
        try:
            with observe(None, kind="ingest"):
                outcome = self.process(
                    conn, job.file_sha, use_llm=params.get("use_llm"), report=report
                )
        except Exception:
            metrics.record_ingest_file("failed")
            raise
        metrics.record_ingest_file("added")
        self.outcomes[job.id] = outcome
        return outcome.summary

    def _reindex_job(self, conn: sqlite3.Connection, job: Job, report: Reporter) -> str:
        assert job.id is not None
        params = job_params(conn, job.id)
        built = self.build(
            conn,
            params["profile"],
            activate=bool(params.get("activate", True)),
            rebuild=bool(params.get("rebuild", False)),
            report=report,
        )
        if built.skipped:
            return f"{built.profile.name} was already built"
        return (
            f"{built.profile.name}: {built.documents} document(s), {built.chunks} chunk(s), "
            f"{built.embedded} embedded, {built.cache_hits} cached"
        )

    def handlers(self) -> dict[str, Callable[[sqlite3.Connection, Job, Reporter], str]]:
        return {"ingest": self._ingest_job, "reindex": self._reindex_job}

    def worker(self, db_path: Path | str) -> JobWorker:
        """The one background worker for this pipeline (call ``start()`` in the service)."""
        return JobWorker(db_path, self.handlers(), bus=self.bus)
