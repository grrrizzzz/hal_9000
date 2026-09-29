"""The knowledge-base HTTP API (plan §11): FastAPI, prefix ``/api``.

KWP0 fixes the *contract*: every route with its request and response models
(from :mod:`hal9000.knowledge.contracts`), each answering ``501 Not
Implemented``. ``scripts/kb_export_openapi.py`` writes the schema to
``web/src/api/openapi.json``, which the UI's types are generated from; a test
fails if the committed file drifts. KWP6 replaces the stub bodies, keeping the
paths, parameters and models (changing them needs the orchestrator's approval,
rule K4).
"""

from hal9000.knowledge.api.app import ROUTES, create_app

__all__ = ["ROUTES", "create_app"]
