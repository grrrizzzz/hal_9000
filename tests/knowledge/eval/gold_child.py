"""``kb gold generate`` in a child process with a slow fake LLM (for the SIGTERM test).

Usage: ``python -m tests.knowledge.eval.gold_child DELAY`` with HAL_DATA_DIR,
HAL_CONFIG_DIR and a fake OPENAI_API_KEY in the environment. Offline: the
provider is :class:`tests.knowledge.eval.fakes.RealisticProvider`.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def main() -> int:
    import hal9000.llm
    from hal9000.__main__ import main as hal_main
    from tests.knowledge.eval.fakes import GoldLLM, RealisticProvider

    hal9000.llm.generate_response = RealisticProvider(GoldLLM(), delay=float(sys.argv[1]))
    return hal_main(["kb", "gold", "generate", "--allow-keyword-check", "--yes"])


if __name__ == "__main__":
    sys.exit(main())
