"""The guestbook file: path confinement, monotonic ids, and safe writes."""

from __future__ import annotations

import json
import threading

import pytest

from hello_agent.storage.guestbook import GuestbookPathError, GuestbookStore

# --- where the file lives ---------------------------------------------------


def test_path_comes_from_config_when_none_is_given(guestbook_path):
    """AGENT_GUESTBOOK_PATH decides, and it is read at construction, not at import."""
    store = GuestbookStore()
    assert store.path == guestbook_path.resolve()


def test_file_is_created_on_first_write_not_before(tmp_path):
    store = GuestbookStore("guestbook.json", root=tmp_path)

    assert not store.exists()
    assert store.entries() == []

    store.append({"name": "Ada"})
    assert store.exists()
    assert json.loads(store.path.read_text())["entries"][0]["name"] == "Ada"


def test_nested_directories_are_created_on_demand(tmp_path):
    store = GuestbookStore("books/2026/guestbook.json", root=tmp_path)
    store.append({"name": "Ada"})
    assert (tmp_path / "books" / "2026" / "guestbook.json").exists()


@pytest.mark.parametrize(
    "escape",
    [
        "../escaped.json",
        "books/../../escaped.json",
        "/etc/passwd",
        "~/escaped.json",
    ],
)
def test_traversal_out_of_the_root_raises(tmp_path, escape):
    """Environmental safety: the store refuses any path outside its directory."""
    with pytest.raises(GuestbookPathError) as excinfo:
        GuestbookStore(escape, root=tmp_path)

    message = str(excinfo.value)
    assert str(tmp_path) in message  # names the directory it is confined to
    assert "Use a plain file name" in message  # and tells the caller the fix


def test_a_traversal_that_normalises_back_inside_is_allowed(tmp_path):
    store = GuestbookStore("books/../guestbook.json", root=tmp_path)
    assert store.path == (tmp_path / "guestbook.json").resolve()


def test_a_symlink_pointing_out_of_the_root_is_refused(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(GuestbookPathError):
        GuestbookStore("escape/guestbook.json", root=tmp_path)


# --- entries ----------------------------------------------------------------


def test_entry_ids_are_monotonic(tmp_path):
    store = GuestbookStore("guestbook.json", root=tmp_path)
    ids = [store.append({"name": name})["entry_id"] for name in ("Ada", "Grace", "Alan")]
    assert ids == [1, 2, 3]


def test_entry_ids_are_never_reused_after_a_removal(tmp_path):
    """The counter lives in the file, so undoing an entry does not recycle its id."""
    store = GuestbookStore("guestbook.json", root=tmp_path)
    store.append({"name": "Ada"})
    store.append({"name": "Grace"})

    assert store.remove_last()["name"] == "Grace"
    assert store.append({"name": "Alan"})["entry_id"] == 3


def test_append_stamps_a_timestamp_and_returns_the_stored_entry(tmp_path):
    store = GuestbookStore("guestbook.json", root=tmp_path)
    entry = store.append({"name": "Ada", "note": "first"})

    assert entry["entry_id"] == 1
    assert entry["name"] == "Ada"
    assert entry["signed_at"].startswith("20")
    assert store.entries() == [entry]


def test_entries_honours_a_limit_and_returns_the_most_recent(tmp_path):
    store = GuestbookStore("guestbook.json", root=tmp_path)
    for name in ("Ada", "Grace", "Alan"):
        store.append({"name": name})

    assert [e["name"] for e in store.entries(limit=2)] == ["Grace", "Alan"]
    assert store.count() == 3


def test_remove_entry_finds_by_id_and_returns_none_when_absent(tmp_path):
    store = GuestbookStore("guestbook.json", root=tmp_path)
    store.append({"name": "Ada"})
    store.append({"name": "Grace"})

    assert store.remove_entry(1)["name"] == "Ada"
    assert [e["name"] for e in store.entries()] == ["Grace"]
    assert store.remove_entry(1) is None


def test_remove_last_on_an_empty_book_returns_none(tmp_path):
    assert GuestbookStore("guestbook.json", root=tmp_path).remove_last() is None


# --- durability -------------------------------------------------------------


def test_corrupt_json_raises_an_error_that_says_what_to_do(tmp_path):
    store = GuestbookStore("guestbook.json", root=tmp_path)
    store.path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Move it aside"):
        store.read()


def test_a_file_written_without_a_counter_still_gets_monotonic_ids(tmp_path):
    """Forward compatibility: the counter is recovered from the entries."""
    store = GuestbookStore("guestbook.json", root=tmp_path)
    store.path.write_text(json.dumps({"entries": [{"entry_id": 7, "name": "Ada"}]}))

    assert store.append({"name": "Grace"})["entry_id"] == 8


def test_no_temporary_files_are_left_behind(tmp_path):
    store = GuestbookStore("guestbook.json", root=tmp_path)
    store.append({"name": "Ada"})
    assert [p.name for p in tmp_path.iterdir()] == ["guestbook.json"]


def test_interleaved_writers_never_lose_an_entry_or_duplicate_an_id(tmp_path):
    """Read-modify-write under threads: the lock is what makes this deterministic.

    Two stores, same file -- so the lock has to be keyed on the path, not held
    per instance.
    """
    stores = [GuestbookStore("guestbook.json", root=tmp_path) for _ in range(2)]
    results: list[dict] = []
    barrier = threading.Barrier(10)

    def sign(index: int) -> None:
        barrier.wait()
        results.append(stores[index % 2].append({"name": f"visitor-{index}"}))

    threads = [threading.Thread(target=sign, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(entry["entry_id"] for entry in results) == list(range(1, 11))
    assert sorted(e["entry_id"] for e in stores[0].entries()) == list(range(1, 11))
