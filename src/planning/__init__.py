"""Planning layer: the numbers behind the action list and the SKU detail view.

Imported by both hosts, deliberately. The Streamlit dashboard renders it directly
and the FastAPI service serves it to the Next.js page, so there is exactly one
implementation of the recommended order quantity, the coverage demand, the
stockout projection and the reliability tiers. Rendering belongs to the host;
anything that produces a number a user acts on belongs here.

Nothing in this package imports Streamlit. `_cache` resolves to `st.cache_data`
when it happens to be available and to a no-op otherwise.
"""

from src.planning import calc, data, quality, reliability

__all__ = ["calc", "data", "quality", "reliability"]
