"""Ingestion (plan §5.1): parse → clean → segment → chunk → embed → index.

KWP1 provides the first three steps: :mod:`.parsers`, :mod:`.clean` and
:mod:`.segment`. Each is wrapped in the shared ``stage()`` timer (rule K7).
"""
