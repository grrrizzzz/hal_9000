"""Embeddings, per-profile search indexes and retrieval primitives (plan §5.1, §6.2, §7.1, §12).

- ``embedding``: :class:`EmbeddingService`, cached by (model, dimensions, text
  SHA-256), batched, retried and metered.
- ``fake``: :class:`FakeEmbedder`, the deterministic offline embedder (tests,
  ``kb serve --fake``).
- ``vector``: the ``VectorIndex`` protocol and ``SqliteVecIndex`` (``vec_<id>``).
- ``keyword``: ``FtsIndex`` (``fts_<id>``, BM25) and ``fts_query``, the safe
  ``MATCH`` builder.
- ``fusion``: RRF (k = 60) and weighted fusion.
- ``search``: ``vector_search`` / ``keyword_search``, the timed primitives KWP4 composes.
- ``writer``: chunk → embed → write, shared by ingestion and profile builds.
- ``profiles``: profile rows, builds from stored pages, atomic activation.
- ``library``: file, document and page rows; the persistent segmentation cache.

Nothing is imported here, so importing one module stays light.
"""
