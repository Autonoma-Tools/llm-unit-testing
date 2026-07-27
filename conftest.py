"""Make the repository root importable so `from summarizer...` works anywhere.

Keeps `pytest`, `pytest tests/`, and `pytest tests/test_prompt_logic_mocked.py`
all behaving the same way, with no editable install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
