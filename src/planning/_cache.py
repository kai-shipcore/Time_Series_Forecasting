"""Optional memoisation for the planning layer.

This package was imported by two hosts with different caching stories: the
Streamlit dashboard, where `st.cache_data` is the right mechanism and is already
wired to the session, and the FastAPI service, where Streamlit is not installed
and caching is the caller's business. The decorator resolves to whichever applies
so that neither host has to know about the other, and so that a module here can
be imported with no Streamlit present at all.

**The Streamlit dashboard was retired on 2026-08-12**, so in practice the
fallback below is now the only branch ever taken. Kept rather than simplified
because it costs nothing, it is the reason this module can be imported anywhere,
and removing it would be a change to a working caching path made on the strength
of a host that no longer exists rather than on any measurement.

Previously each module carried its own copy of this try/except. Three copies of
the same fallback is three chances for them to stop agreeing.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only under streamlit
    import streamlit as st

    cache = st.cache_data
except Exception:  # pragma: no cover - the FastAPI and plain-python paths

    def cache(*args, **kwargs):
        """No-op stand-in with the same call shapes as st.cache_data.

        Supports both `@cache` and `@cache(show_spinner=False)`, because the
        call sites use both and a decorator that only handles one of them fails
        confusingly at import time rather than at the call.
        """
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap
