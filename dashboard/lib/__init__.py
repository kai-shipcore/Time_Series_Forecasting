"""Support library for the Streamlit inventory/forecast dashboard.

The planning logic no longer lives here. It moved to ``src/planning`` so that the
FastAPI service can serve the same numbers to the Next.js page without a second
implementation of the order formula existing anywhere. Two copies of that
arithmetic would drift, and nothing would say which screen was right.

What remains under this package is rendering only:

- ui:      Streamlit rendering helpers, the stylesheet, the chart builders.

The names below are re-exported so the page scripts read the same as before:

- data:        file loaders and the inventory adapter       (src.planning.data)
- calc:        order quantities, priorities, stockout dates (src.planning.calc)
- quality:     data-quality exception checks                (src.planning.quality)
- reliability: per-SKU forecast error and tiers             (src.planning.reliability)

These are aliases, not copies. ``lib.calc is src.planning.calc`` holds, so there
is one module object and no way for the two to diverge.
"""

import sys
from pathlib import Path

# The Streamlit entrypoint runs with dashboard/ on sys.path, not the repo root,
# so `src` is not importable by default. Done here rather than in each page, so
# that importing `lib` is sufficient however the app is launched.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.planning import calc, data, quality, reliability  # noqa: E402

__all__ = ["calc", "data", "quality", "reliability", "ui"]
