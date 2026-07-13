# Provider is configured entirely via env:
#   Gemini (free):  LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/  LLM_MODEL=gemini-2.0-flash
#   Groq (free):    LLM_BASE_URL=https://api.groq.com/openai/v1                           LLM_MODEL=llama-3.3-70b-versatile
#   Claude (paid):  LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-4-5  (OpenAI-compat layer)
# Note: Anthropic's OpenAI-compat layer is testing-oriented (no prompt caching);
# if/when Claude becomes the permanent provider, port run_chat to the native
# anthropic SDK to enable prompt caching (~90% input cost reduction).

import json
import os
import re
import time
import uuid

import openai
import pandas as pd
import requests
from openai import OpenAI

MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
MAX_TOOL_ITERATIONS = 8
MAX_TOOL_RESULT_CHARS = 20_000
API_BASE = "http://127.0.0.1:8001"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
    return _client


def _api_get(path: str, params: dict | None = None) -> dict | list:
    """Call one of our own endpoints. Includes the auth token if configured."""
    headers = {}
    token = os.getenv("FORECAST_API_TOKEN")
    if token:
        headers["x-forecast-token"] = token
    r = requests.get(f"{API_BASE}{path}", params=params or {}, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


# ── Tool implementations ─────────────────────────────────────────────────────

def tool_search_skus(query: str) -> list:
    return _api_get("/sku-search", {"q": query})[:20]


def tool_get_segment_overview(weeks: int = 10, product_type: str = "All") -> dict:
    data = _api_get("/segments", {"weeks": weeks, "product_type": product_type})
    data.pop("pareto", None)
    return data


def tool_get_segment_skus(segment: str, weeks: int = 10, product_type: str = "All",
                          sort_by: str = "yhat_total", limit: int = 25) -> dict:
    data = _api_get(f"/segment-detail/{segment}", {"weeks": weeks, "product_type": product_type})
    skus = data.get("skus", [])
    if skus and sort_by in skus[0]:
        skus = sorted(skus, key=lambda s: (s.get(sort_by) is None, s.get(sort_by) or 0), reverse=True)
    data["total_sku_count"] = len(skus)
    data["skus"] = skus[:max(1, min(limit, 50))]
    return data


def tool_get_sku_forecast(sku_id: str, weeks_history: int = 26) -> dict:
    data = _api_get(f"/forecast/{sku_id}", {"weeks": weeks_history})
    return {
        "meta":            data.get("meta"),
        "forecast_dates":  data.get("forecastDates"),
        "forecast_values": data.get("forecastValues"),
        "forecast_upper":  data.get("forecastUpper"),
        "level":           data.get("level"),
    }


def tool_get_sku_history(sku_id: str, last_n_weeks: int = 26) -> dict:
    data = _api_get(f"/history/{sku_id}")
    return {
        "sku_id": data["sku_id"],
        "weeks":  list(zip(data["dates"][-last_n_weeks:], data["values"][-last_n_weeks:])),
    }


def tool_get_all_skus_summary(weeks: int = 10, product_type: str = "All",
                              sort_by: str = "demand_total", limit: int = 25,
                              segment: str | None = None) -> dict:
    data = _api_get("/all-skus", {"weeks": weeks, "product_type": product_type})
    skus = data.get("skus", [])
    if segment:
        skus = [s for s in skus if s.get("segment") == segment]
    skus = sorted(skus, key=lambda s: (s.get(sort_by) is None, s.get(sort_by) or 0), reverse=True)
    data["total_sku_count"] = len(skus)
    data["skus"] = skus[:max(1, min(limit, 50))]
    return data


def tool_get_accuracy_history(product_type: str = "All", k: int = 4) -> dict:
    data = _api_get("/accuracy-history", {"product_type": product_type})
    series = [r for r in data.get("series", []) if r.get("k") == k]
    return {"last_complete_week": data.get("last_complete_week"), "k": k, "series": series}


def tool_run_sku_backtest(sku_id: str, cutoff: str, horizon: int = 13) -> dict:
    data = _api_get(f"/backtest/{sku_id}", {"cutoff": cutoff, "horizon": horizon})
    data.pop("actuals_context", None)
    return data


TOOL_IMPLS = {
    "search_skus":          tool_search_skus,
    "get_segment_overview": tool_get_segment_overview,
    "get_segment_skus":     tool_get_segment_skus,
    "get_sku_forecast":     tool_get_sku_forecast,
    "get_sku_history":      tool_get_sku_history,
    "get_all_skus_summary": tool_get_all_skus_summary,
    "get_accuracy_history": tool_get_accuracy_history,
    "run_sku_backtest":     tool_run_sku_backtest,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_skus",
        "description": "Find SKU ids matching a text query. Use before any per-SKU tool if the user gave a partial or uncertain SKU name.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_segment_overview",
        "description": "Aggregate view: SKU counts, demand totals, and forecast coverage per segment (smooth_full, smooth_short, intermittent) over the last N complete weeks.",
        "parameters": {"type": "object", "properties": {
            "weeks":        {"type": "integer", "description": "Lookback window in complete weeks, default 10"},
            "product_type": {"type": "string",  "description": "'All' or comma-separated: Car Cover, Seat Cover, Floor Mat"}}}}},
    {"type": "function", "function": {
        "name": "get_segment_skus",
        "description": "Per-SKU table for one segment: forecast totals (= recommended order amounts), P85 bounds, recent demand, model used, confidence, training WAPE. segment must be smooth_full, smooth_short, or intermittent.",
        "parameters": {"type": "object", "properties": {
            "segment":      {"type": "string",  "enum": ["smooth_full", "smooth_short", "intermittent"]},
            "weeks":        {"type": "integer"},
            "product_type": {"type": "string"},
            "sort_by":      {"type": "string",  "description": "Field to sort desc by, e.g. yhat_total, demand_total, train_wape"},
            "limit":        {"type": "integer", "description": "Max rows, default 25, max 50"}},
        "required": ["segment"]}}},
    {"type": "function", "function": {
        "name": "get_sku_forecast",
        "description": "Forward weekly forecast for one SKU: predicted units per week (the sum over the horizon is the recommended order amount), upper interval bound, model, bucket, confidence.",
        "parameters": {"type": "object", "properties": {
            "sku_id":        {"type": "string"},
            "weeks_history": {"type": "integer", "description": "Weeks of history context, default 26"}},
        "required": ["sku_id"]}}},
    {"type": "function", "function": {
        "name": "get_sku_history",
        "description": "Weekly sales history for one SKU. Use for intermittent SKUs (no forecast exists) or to inspect recent demand shape.",
        "parameters": {"type": "object", "properties": {
            "sku_id":       {"type": "string"},
            "last_n_weeks": {"type": "integer", "description": "default 26"}},
        "required": ["sku_id"]}}},
    {"type": "function", "function": {
        "name": "get_all_skus_summary",
        "description": "Cross-segment SKU directory with demand totals, 4-week trend %, year-over-year comparison, and forecast totals. Best tool for 'which SKUs are trending up/down' questions.",
        "parameters": {"type": "object", "properties": {
            "weeks":        {"type": "integer"},
            "product_type": {"type": "string"},
            "sort_by":      {"type": "string",  "description": "e.g. demand_total, trend_pct, forecast_total, yoy_4w"},
            "limit":        {"type": "integer"},
            "segment":      {"type": "string",  "description": "Optional filter: smooth_full, smooth_short, intermittent"}}}}},
    {"type": "function", "function": {
        "name": "get_accuracy_history",
        "description": "Historical forecast accuracy: pooled WAPE per stored forecast run per segment, evaluated over the first k completed weeks of each run. Use to answer 'how accurate is the forecast'.",
        "parameters": {"type": "object", "properties": {
            "product_type": {"type": "string"},
            "k":            {"type": "integer", "description": "Evaluation window in weeks: 1, 2, 4, 8, 10, or 13. Default 4"}}}}},
    {"type": "function", "function": {
        "name": "run_sku_backtest",
        "description": "Simulate what the model would have predicted for one SKU from a past cutoff Monday (YYYY-MM-DD), compared against what actually sold. Slow (~10-60s); use only when the user asks to validate accuracy for a specific SKU.",
        "parameters": {"type": "object", "properties": {
            "sku_id":  {"type": "string"},
            "cutoff":  {"type": "string"},
            "horizon": {"type": "integer", "description": "Weeks, default 13"}},
        "required": ["sku_id", "cutoff"]}}},
]


def _system_prompt() -> str:
    today = pd.Timestamp.today().normalize()
    last_complete = today - pd.Timedelta(days=today.dayofweek or 7)
    return f"""You are the demand forecasting assistant inside Demand Pilot, an internal tool for a vehicle accessories e-commerce company (car covers, seat covers, floor mats).

Today is {today.date()}. The last complete sales week is labeled {last_complete.date()}.

Domain facts you must apply:
- Weeks use the W-MON convention: each week is labeled by the Monday it ENDS on.
- Every SKU has a bucket: smooth (statistically forecasted), intermittent (too sporadic — no forecast, restock policy instead), low_volume. Smooth SKUs split by history: smooth_full (>=50 active weeks) and smooth_short (<50).
- The forecast total over the horizon is the recommended order amount for that SKU.
- yhat_hi (P85 upper bound) is the safety-stock-conscious order amount; ordering to yhat covers the median case.
- Pooled WAPE is the accuracy metric: total absolute error / total demand across SKUs. 0.20 means 20% error. Lower is better.
- Forecast confidence and train_wape indicate per-SKU reliability. Treat high-WAPE or low-confidence SKUs' forecasts as rough guides.

Rules:
- Always fetch data with tools before answering data questions. Never invent SKU ids or numbers.
- If a SKU id looks partial or unfamiliar, use search_skus first.
- Intermittent SKUs have NO forecast — say so and show recent history instead of guessing.
- When recommending order amounts, state the horizon (weeks) and whether you used the point forecast or the P85 upper bound, and flag low-confidence forecasts.
- Be concise. Use small tables for multi-SKU answers. State the time window of any number you quote.
- Never use LaTeX math notation ($...$). Write everything in plain text: use >=, <=, %, plain numbers, and Unicode symbols (≥ ≤ × ÷) directly. Variable names like k, n go unformatted.
- If a question is outside forecasting/inventory (HR, pricing strategy, etc.), say it's outside your scope.
- You have no memory of this company's data. ANY question about SKUs, demand, forecasts, or accuracy REQUIRES a tool call first."""


def _rate_limit_delay(exc: openai.RateLimitError, fallback: float) -> float:
    """Return Retry-After seconds from the response header, or the fallback."""
    try:
        val = exc.response.headers.get("retry-after")
        if val:
            return max(1.0, float(val))
    except Exception:
        pass
    return fallback


def _create_with_retry(client: OpenAI, **kwargs):
    """Call chat.completions.create with up to 3 retries on rate-limit errors."""
    delays = [5, 15, 30]
    last_exc: Exception | None = None
    for delay in delays + [None]:
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            last_exc = exc
            if delay is None:
                raise
            time.sleep(_rate_limit_delay(exc, delay))
        except Exception as exc:
            if "429" in str(exc):
                last_exc = exc
                if delay is None:
                    raise
                time.sleep(delay)
            else:
                raise
    raise last_exc  # type: ignore[misc]


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


class ThinkingFilter:
    """Strips <thinking>...</thinking> from a content stream chunk by chunk."""
    _OPEN  = "<thinking>"
    _CLOSE = "</thinking>"

    def __init__(self) -> None:
        self._buf   = ""
        self._state = "start"   # "start" | "thinking" | "done"

    def feed(self, chunk: str) -> str:
        if self._state == "done":
            return chunk
        self._buf += chunk

        if self._state == "start":
            if self._OPEN in self._buf:
                self._state = "thinking"
                self._buf = self._buf.split(self._OPEN, 1)[1]
                # fall through to "thinking" check below
            elif len(self._buf) >= len(self._OPEN):
                # Enough chars and no opening tag — passthrough
                self._state = "done"
                out, self._buf = self._buf, ""
                return out
            else:
                return ""  # accumulate more

        if self._state == "thinking":
            if self._CLOSE in self._buf:
                self._state = "done"
                out = self._buf.split(self._CLOSE, 1)[1].lstrip("\n")
                self._buf = ""
                return out
            # Keep the last N-1 chars so a split closing tag is never lost
            keep = len(self._CLOSE) - 1
            self._buf = self._buf[-keep:] if len(self._buf) >= keep else self._buf

        return ""

    def flush(self) -> str:
        out = self._buf if self._state != "thinking" else ""
        self._buf = ""
        return out


def _stream_with_retry(client: OpenAI, **kwargs):
    """Like _create_with_retry but returns a streaming response."""
    delays = [5, 15, 30]
    last_exc: Exception | None = None
    for delay in delays + [None]:
        try:
            return client.chat.completions.create(**kwargs, stream=True)
        except openai.RateLimitError as exc:
            last_exc = exc
            if delay is None:
                raise
            time.sleep(_rate_limit_delay(exc, delay))
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "503" in msg or "UNAVAILABLE" in msg:
                last_exc = exc
                if delay is None:
                    raise
                time.sleep(delay)
            else:
                raise
    raise last_exc  # type: ignore[misc]


def stream_chat(messages: list[dict]):
    """Generator that yields SSE strings for the full agentic loop."""
    try:
        yield from _stream_chat_inner(messages)
    except Exception as exc:
        yield _sse({"type": "error", "text": str(exc)})
        yield _sse({"type": "done", "tool_calls": []})


def _recover_tool_calls(failed_generation: str) -> list[dict]:
    """Parse tool calls out of Groq's `failed_generation` error payload.

    Llama 3.3 sometimes emits its legacy text format instead of native
    tool-call tokens — e.g. `<function=name{"arg": 1}</function>` (note the
    often-missing `>` after the name). Groq's parser rejects it, but the
    intended call is fully recoverable from the error body, costing zero
    extra API requests. Returns [] if nothing parseable is found.
    """
    recovered = []
    pattern = r'<function=([A-Za-z0-9_]+)>?\s*(\{.*?\})\s*(?:</function>)?'
    for i, m in enumerate(re.finditer(pattern, failed_generation, re.DOTALL)):
        fn, args = m.group(1), m.group(2)
        if fn not in TOOL_IMPLS:
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError:
            continue
        recovered.append({
            "id": f"recovered_{i}_{uuid.uuid4().hex[:8]}",
            "function": {"name": fn, "arguments": args},
        })
    return recovered


def _stream_chat_inner(messages: list[dict]):
    client  = get_client()
    convo   = [{"role": "system", "content": _system_prompt()}]
    convo  += [{"role": m["role"], "content": m["content"]} for m in messages]
    tool_calls_made: list[dict] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        yield _sse({"type": "status", "text": "Thinking…"})

        content_parts: list[str] = []
        raw_tcs: list[dict]      = []
        tf = ThinkingFilter()
        last_keepalive = time.monotonic()

        stream = _stream_with_retry(
            client, model=MODEL, max_tokens=4096,
            tools=TOOLS, tool_choice="auto", messages=convo,
        )
        try:
            for chunk in stream:
                now = time.monotonic()
                if now - last_keepalive > 5:
                    yield ": keepalive\n\n"
                    last_keepalive = now

                choice = chunk.choices[0]
                delta  = choice.delta

                if delta.tool_calls:
                    for tcd in delta.tool_calls:
                        idx = tcd.index
                        while len(raw_tcs) <= idx:
                            raw_tcs.append({"id": "", "function": {"name": "", "arguments": ""}})
                        if tcd.id:
                            raw_tcs[idx]["id"] = tcd.id
                        if tcd.function:
                            if tcd.function.name:
                                raw_tcs[idx]["function"]["name"] += tcd.function.name
                            if tcd.function.arguments:
                                raw_tcs[idx]["function"]["arguments"] += tcd.function.arguments

                if delta.content:
                    content_parts.append(delta.content)
                    out = tf.feed(delta.content)
                    if out:
                        yield _sse({"type": "delta", "text": out})

        except openai.APIError as exc:
            # Groq rejects the generation when Llama emits its legacy text
            # tool-call format instead of native tool-call tokens. The intended
            # call is in the error body — recover it instead of retrying.
            body   = getattr(exc, "body", None)
            failed = body.get("failed_generation") if isinstance(body, dict) else None
            if not failed:
                raise
            raw_tcs = _recover_tool_calls(failed)
            if not raw_tcs:
                # Unrecoverable garbage — answer from what we have without tools.
                stream = _stream_with_retry(
                    client, model=MODEL, max_tokens=4096, messages=convo,
                )
                for chunk in stream:
                    now = time.monotonic()
                    if now - last_keepalive > 5:
                        yield ": keepalive\n\n"
                        last_keepalive = now
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_parts.append(delta.content)
                        out = tf.feed(delta.content)
                        if out:
                            yield _sse({"type": "delta", "text": out})

        out = tf.flush()
        if out:
            yield _sse({"type": "delta", "text": out})

        full_content = "".join(content_parts)

        if raw_tcs:
            convo.append({
                "role":       "assistant",
                "content":    full_content or None,
                "tool_calls": [{"id": tc["id"], "type": "function", "function": tc["function"]} for tc in raw_tcs],
            })
            for tc in raw_tcs:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    convo.append({"role": "tool", "tool_call_id": tc["id"],
                                  "content": "Error: your tool arguments were not valid JSON. Retry the call with corrected arguments."})
                    continue
                tool_calls_made.append({"tool": fn, "input": args})

                label = fn.replace("_", " ")
                yield _sse({"type": "status", "text": f"Checking {label}…"})

                impl = TOOL_IMPLS.get(fn)
                try:
                    result  = impl(**args) if impl else {"error": f"unknown tool {fn}"}
                    payload = json.dumps(result, default=str)
                    if len(payload) > MAX_TOOL_RESULT_CHARS:
                        payload = payload[:MAX_TOOL_RESULT_CHARS] + "… [truncated]"
                except Exception as exc:
                    payload = f"Tool error: {exc}"

                convo.append({"role": "tool", "tool_call_id": tc["id"], "content": payload})
        else:
            yield _sse({"type": "done", "tool_calls": tool_calls_made})
            return

    yield _sse({"type": "done", "tool_calls": tool_calls_made})
