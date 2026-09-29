"""Streaming ``multipart/form-data`` for ``POST /api/uploads`` (plan §5.1–5.2, §14).

The request body is parsed as it arrives (``python-multipart``), and each file
part is written straight to ``data/knowledge/tmp`` with its size counted as it
goes:

- a part whose file name has an unsupported extension is not written at all;
- a part that passes ``ingest.max_upload_bytes`` stops being written at the
  limit (the rest is only counted, so the reason can say how big it was) and
  its partial file is deleted;
- form fields (``relative_path``) are capped at 4 KB each.

The whole body is also capped by the security middleware (the limit plus
1 MB). Nothing is spooled to ``/tmp`` or copied before screening; the ingest
pipeline then stores each accepted file once, under its SHA-256.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from fastapi import HTTPException

__all__ = ["FIELD_MAX_BYTES", "FilePart", "UploadForm", "parse_upload_form"]

FIELD_MAX_BYTES = 4096
MAX_PARTS = 1000


@dataclass
class FilePart:
    filename: str
    size: int = 0
    path: Path | None = None  # None: not written (unsupported type, or too large)
    too_large: bool = False
    _handle: BinaryIO | None = field(default=None, repr=False)

    def discard(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self.path is not None:
            self.path.unlink(missing_ok=True)
            self.path = None


@dataclass
class UploadForm:
    files: list[FilePart] = field(default_factory=list)
    relative_paths: list[str] = field(default_factory=list)

    def discard(self) -> None:
        for part in self.files:
            part.discard()


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


async def parse_upload_form(
    content_type: str,
    stream: AsyncIterator[bytes],
    *,
    spool: Path,
    max_bytes: int,
    allowed_suffixes: set[str] | frozenset[str],
) -> UploadForm:
    """Parse the body; files land in ``spool`` (the caller deletes them after use)."""
    from python_multipart.multipart import MultipartParser, parse_options_header

    kind, options = parse_options_header(content_type)
    if kind != b"multipart/form-data" or b"boundary" not in options:
        raise _bad("send the files as multipart/form-data")
    form = UploadForm()
    state: dict = {"name": b"", "value": b"", "headers": {}, "field": None, "file": None}

    def on_part_begin() -> None:
        state.update(headers={}, field=None, file=None)

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["name"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["value"] += data[start:end]

    def on_header_end() -> None:
        state["headers"][state["name"].lower()] = state["value"]
        state["name"], state["value"] = b"", b""

    def on_headers_finished() -> None:
        if len(form.files) + len(form.relative_paths) >= MAX_PARTS:
            raise _bad(f"too many parts (at most {MAX_PARTS})")
        _, disposition = parse_options_header(state["headers"].get(b"content-disposition", b""))
        name = disposition.get(b"name", b"").decode("utf-8", "replace")
        if b"filename" in disposition:
            if name != "files":
                raise _bad(f"unexpected file field {name!r}")
            filename = disposition[b"filename"].decode("utf-8", "replace")
            part = FilePart(filename=filename)
            if PurePosixPath(filename).suffix.lower() in allowed_suffixes:
                handle = tempfile.NamedTemporaryFile(dir=spool, prefix="upload-", delete=False)
                part.path, part._handle = Path(handle.name), handle
            form.files.append(part)
            state["file"] = part
        else:
            if name != "relative_path":
                raise _bad(f"unexpected form field {name!r}")
            state["field"] = bytearray()

    def on_part_data(data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        part: FilePart | None = state["file"]
        if part is not None:
            part.size += len(chunk)
            if part.size > max_bytes and not part.too_large:
                part.too_large = True
                part.discard()  # stop writing at the limit
            if part._handle is not None:
                part._handle.write(chunk)
            return
        value: bytearray = state["field"]
        if len(value) + len(chunk) > FIELD_MAX_BYTES:
            raise _bad(f"a relative_path is longer than {FIELD_MAX_BYTES} bytes")
        value.extend(chunk)

    def on_part_end() -> None:
        part: FilePart | None = state["file"]
        if part is not None:
            if part._handle is not None:
                part._handle.close()
                part._handle = None
        elif state["field"] is not None:
            form.relative_paths.append(bytes(state["field"]).decode("utf-8", "replace"))

    parser = MultipartParser(
        options[b"boundary"],
        {
            "on_part_begin": on_part_begin,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
        },
    )
    spool.mkdir(parents=True, exist_ok=True)
    try:
        async for chunk in stream:
            parser.write(chunk)
        parser.finalize()
    except HTTPException:
        form.discard()
        raise
    except Exception as error:  # a malformed body
        form.discard()
        raise _bad(f"the multipart body could not be read ({type(error).__name__})") from None
    except BaseException:
        form.discard()
        raise
    return form
