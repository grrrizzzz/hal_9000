"""KF1 ``llm.stream_response`` and KF2 ``llm.embed`` against fake LiteLLM calls (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hal9000 import llm
from hal9000.core.prompt import Prompt


def _chunk(text=None, usage=None, choices=True):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta)] if choices else [],
        usage=usage,
    )


def _usage(prompt=120, cached=40, completion=12):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )


@pytest.fixture
def fake_stream(monkeypatch):
    import litellm

    box = {"kwargs": None, "chunks": []}

    def completion(**kwargs):
        box["kwargs"] = kwargs

        def gen():
            yield from box["chunks"]

        return gen()

    monkeypatch.setattr(litellm, "completion", completion)
    return box


# ------------------------------------------------------------------- KF1
def test_stream_yields_deltas_in_order_then_usage_once(fake_stream, openai_env):
    fake_stream["chunks"] = [
        _chunk("The "),
        _chunk(None),  # role-only first chunks carry no text
        _chunk("roof "),
        _chunk("is tile [S1]."),
        _chunk(usage=_usage(), choices=False),
    ]
    seen = []
    items = list(
        llm.stream_response(
            Prompt(messages=[{"role": "user", "content": "q"}]), "fast", on_usage=seen.append
        )
    )

    assert items[:-1] == ["The ", "roof ", "is tile [S1]."]
    final = items[-1]
    assert isinstance(final, llm.Usage)
    assert final["prompt"] == 120 and final["cached"] == 40 and final["completion"] == 12
    assert final["model"] == "openai/gpt-5-mini"
    assert seen == [final]  # the sink receives it exactly once

    kwargs = fake_stream["kwargs"]
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert "tools" not in kwargs
    assert kwargs["reasoning_effort"] == "low"


def test_stream_sink_is_not_called_before_the_stream_ends(fake_stream, openai_env):
    fake_stream["chunks"] = [_chunk("a"), _chunk("b"), _chunk(usage=_usage(), choices=False)]
    seen = []
    stream = llm.stream_response(Prompt(messages=[]), on_usage=seen.append)
    assert next(stream) == "a" and seen == []
    assert list(stream)[-1] == seen[0]


def test_stream_overrides_and_missing_usage(fake_stream, anthropic_env):
    fake_stream["chunks"] = [_chunk("x")]
    items = list(llm.stream_response(Prompt(messages=[]), "powerful", max_tokens=99))
    assert items[0] == "x" and items[-1]["model"] == "anthropic/claude-sonnet-5"
    assert fake_stream["kwargs"]["max_tokens"] == 99
    assert "reasoning_effort" not in fake_stream["kwargs"]


def test_missing_stream_usage_is_estimated_locally(fake_stream, openai_env):
    fake_stream["chunks"] = [_chunk("The roof "), _chunk("is tile.")]
    messages = [
        {"role": "system", "content": "Answer from the sources."},
        {"role": "user", "content": [{"type": "text", "text": "Roof?"}, {"type": "image"}]},
    ]
    seen = []
    items = list(
        llm.stream_response(Prompt(messages=messages), on_usage=seen.append, count_tokens=len)
    )
    usage = items[-1]
    assert usage["estimated"] is True and seen == [usage]
    assert usage["prompt"] == len("Answer from the sources.\nRoof?\n")
    assert usage["completion"] == len("The roof is tile.")


def test_missing_stream_usage_uses_the_model_tokenizer_by_default(fake_stream, openai_env):
    from hal9000.knowledge.tokens import count_tokens

    fake_stream["chunks"] = [_chunk("hello world")]
    usage = list(llm.stream_response(Prompt(messages=[{"role": "user", "content": "hi"}])))[-1]
    assert usage["completion"] == count_tokens("hello world", "o200k_base") == 2
    assert usage["prompt"] == 1 and usage["estimated"]


def test_reported_stream_usage_is_not_marked_estimated(fake_stream, openai_env):
    fake_stream["chunks"] = [_chunk("x"), _chunk(usage=_usage(), choices=False)]
    assert "estimated" not in list(llm.stream_response(Prompt(messages=[])))[-1]


def test_stream_usage_on_a_content_chunk_counts(fake_stream, openai_env):
    fake_stream["chunks"] = [_chunk("x"), _chunk("y", usage=_usage(completion=2))]
    items = list(llm.stream_response(Prompt(messages=[], metadata={"cache_key": "kb"})))
    assert items[:2] == ["x", "y"] and items[-1]["completion"] == 2
    assert fake_stream["kwargs"]["extra_body"] == {"prompt_cache_key": "kb"}


def test_generate_response_is_unchanged_by_the_refactor(monkeypatch, openai_env):
    import litellm

    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
            usage=None,
        )

    monkeypatch.setattr(litellm, "completion", completion)
    assert llm.generate_response(Prompt(messages=[])) == "ok"
    assert "stream" not in captured and captured["tools"] is None


# ------------------------------------------------------------------- KF2
@pytest.fixture
def fake_embedding(monkeypatch):
    import litellm

    calls = []

    def embedding(**kwargs):
        calls.append(kwargs)
        inputs = kwargs["input"]
        data = [
            {"index": i, "embedding": [float(len(text)), float(i)]} for i, text in enumerate(inputs)
        ]
        data.reverse()  # providers don't promise order; embed() must sort by index
        usage = SimpleNamespace(prompt_tokens=10 * len(inputs))
        return SimpleNamespace(data=data, usage=usage)

    monkeypatch.setattr(litellm, "embedding", embedding)
    return calls


def test_3000_short_inputs_make_two_calls(fake_embedding):
    texts = [f"t{i}" for i in range(3000)]
    per_request = []
    result = llm.embed(
        texts, "openai/text-embedding-3-small", on_usage=per_request.append, count_tokens=len
    )
    assert [len(c["input"]) for c in fake_embedding] == [2048, 952]
    assert result.requests == 2
    assert len(result.vectors) == 3000
    # order restored from the reversed responses
    assert result.vectors[0] == [2.0, 0.0] and result.vectors[2048] == [5.0, 0.0]
    assert result.usage["prompt"] == 30_000 == sum(u["prompt"] for u in per_request)
    assert [u["prompt"] for u in per_request] == [20_480, 9_520]
    assert "dimensions" not in fake_embedding[0]


def test_300_long_inputs_split_by_the_token_limit(fake_embedding):
    from hal9000.knowledge.tokens import count_tokens

    text = " house" * 1500
    assert count_tokens(text) == 1500  # the real cl100k_base tokenizer
    result = llm.embed([text] * 300, "openai/text-embedding-3-large", dimensions=1024)

    sizes = [len(c["input"]) for c in fake_embedding]
    assert sizes == [166, 134]  # 166 x 1,500 = 249,000 <= 250,000 < 167 x 1,500
    assert all(n * 1500 <= llm.EMBED_MAX_TOKENS for n in sizes)
    assert all(c["dimensions"] == 1024 for c in fake_embedding)
    assert result.usage["prompt"] == 3000 and result.usage["model"] == (
        "openai/text-embedding-3-large"
    )


def test_missing_usage_falls_back_to_counted_tokens(monkeypatch):
    import litellm

    monkeypatch.setattr(
        litellm,
        "embedding",
        lambda **kw: SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.5])], usage=None),
    )
    seen = []
    result = llm.embed(
        ["abcd"], "openai/text-embedding-3-small", count_tokens=lambda t: 7, on_usage=seen.append
    )
    assert result.vectors == [[0.5]] and result.usage["prompt"] == 7
    assert result.usage["estimated"] is True and seen[0]["estimated"] is True


def test_before_request_runs_before_every_batch(fake_embedding):
    fake_calls_before = []

    def before():
        fake_calls_before.append(len(fake_embedding))

    llm.embed(
        ["x"] * 3000, "openai/text-embedding-3-small", count_tokens=len, before_request=before
    )
    assert fake_calls_before == [0, 1]  # checked before request 1 and before request 2


def test_vector_count_mismatch_raises(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "embedding", lambda **kw: SimpleNamespace(data=[], usage=None))
    with pytest.raises(RuntimeError, match="0 vectors for 1 inputs"):
        llm.embed(["x"], "openai/text-embedding-3-small", count_tokens=len)


@pytest.mark.parametrize(
    ("counts", "max_inputs", "max_tokens", "expected"),
    [
        ([], 3, 10, []),
        ([1, 1, 1, 1], 3, 10, [(0, 3), (3, 4)]),
        ([4, 4, 4], 10, 8, [(0, 2), (2, 3)]),
        ([20, 1, 1], 10, 8, [(0, 1), (1, 3)]),  # an oversize input travels alone
        ([8, 8], 10, 8, [(0, 1), (1, 2)]),
    ],
)
def test_embedding_batches(counts, max_inputs, max_tokens, expected):
    batches = llm.embedding_batches(counts, max_inputs, max_tokens)
    assert [(b.start, b.stop) for b in batches] == expected
