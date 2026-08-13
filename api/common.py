"""Helpers shared by both API tracks.

This module exists for one reason: `api/main.py` (the LightGBM serving
endpoints) and `api/legacy.py` (the frozen statsforecast endpoints) both need
`JobLogger`, and importing it from one into the other would make the two modules
circular. Anything genuinely common to both tracks belongs here; anything used
by only one belongs in that one.

Keep this module small. It is not a dumping ground for utilities: a helper that
only the legacy track uses should live in `api/legacy.py`, so that retiring that
track removes it. `_parse_product_types`, `_cached_response` and `_data_version`
were moved into `api/legacy.py` for exactly that reason, having turned out to
have no ML-side callers.
"""

import time
import threading

from src.db import append_job_lines


class JobLogger:
    """Buffers log lines and flushes to fc_jobs in batches, so per-line
    subprocess output doesn't become one DB write per line."""
    def __init__(self, job_id: str, flush_every: int = 20, flush_secs: float = 1.0):
        self.job_id = job_id
        self.buf: list[str] = []
        self.flush_every = flush_every
        self.flush_secs = flush_secs
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self.buf.append(line)
            if (len(self.buf) >= self.flush_every
                    or time.monotonic() - self._last_flush >= self.flush_secs):
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self.buf:
            append_job_lines(self.job_id, self.buf)
            self.buf = []
        self._last_flush = time.monotonic()
