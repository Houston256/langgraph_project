"""Cost per model call, recorded in `usage`, and the spending limits checked before a run."""

import logging
import time
from typing import Any
from uuid import UUID

import httpx
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from assistant.config import settings
from assistant.db import Pool

log = logging.getLogger(__name__)

PRICES_URL = "https://openrouter.ai/api/v1/models"
PRICES_TTL_SECONDS = 6 * 3600


class Pricing:
    """$ per token for OpenRouter models, refreshed every few hours."""

    def __init__(self) -> None:
        self._prices: dict[str, tuple[float, float, float]] = {}
        self._loaded_at = 0.0

    async def refresh(self) -> None:
        if time.monotonic() - self._loaded_at < PRICES_TTL_SECONDS and self._prices:
            return
        try:
            async with httpx.AsyncClient(timeout=20) as http:
                data = (await http.get(PRICES_URL)).json()["data"]
        except Exception:
            log.exception("could not load model prices, keeping the old ones")
            return
        prices = {}
        for m in data:
            p = m.get("pricing") or {}
            try:
                prompt = float(p["prompt"])
                prices[m["id"]] = (
                    prompt,
                    float(p.get("input_cache_read") or prompt),
                    float(p["completion"]),
                )
            except KeyError, TypeError, ValueError:
                continue
        self._prices, self._loaded_at = prices, time.monotonic()

    def get(self, model: str) -> tuple[float, float, float] | None:
        return self._prices.get(model)

    def cost(
        self, model: str, input_tokens: int, cached: int, output: int
    ) -> float | None:
        price = self.get(model)
        if price is None:
            return None
        return (
            (input_tokens - cached) * price[0] + cached * price[1] + output * price[2]
        )


def limits_enabled() -> bool:
    return any(
        v is not None
        for v in (settings.limit_global_usd_per_month, settings.limit_user_usd_per_week)
    )


class UsageRecorder(AsyncCallbackHandler):
    """Writes one `usage` row per finished model call of a run."""

    def __init__(self, pool: Pool, pricing: Pricing, run_id: UUID, user_id: str):
        self.pool = pool
        self.pricing = pricing
        self.run_id = run_id
        self.user_id = user_id

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for g in generations:
                message = getattr(g, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    continue
                model = (message.response_metadata or {}).get("model_name") or "unknown"
                cached = (usage.get("input_token_details") or {}).get(
                    "cache_read", 0
                ) or 0
                cost = self.pricing.cost(
                    model, usage["input_tokens"], cached, usage["output_tokens"]
                )
                if cost is None:
                    log.warning(
                        "no price for model %s; usage recorded without cost", model
                    )
                async with self.pool.connection() as conn:
                    await conn.execute(
                        "INSERT INTO usage (run_id, user_id, model, input_tokens, cached_tokens, "
                        "output_tokens, cost_usd) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            self.run_id,
                            self.user_id,
                            model,
                            usage["input_tokens"],
                            cached,
                            usage["output_tokens"],
                            cost,
                        ),
                    )


async def limit_exceeded(pool: Pool, user_id: str) -> str | None:
    """Why a new run for this user is refused, or None.

    Soft limits: a call's cost counts once it ends, so runs in progress can go over.
    """
    async with pool.connection() as conn:
        if (limit := settings.limit_global_usd_per_month) is not None:
            cur = await conn.execute(
                "SELECT coalesce(sum(cost_usd), 0) AS s FROM usage "
                "WHERE created_at >= date_trunc('month', now())"
            )
            if (await cur.fetchone())["s"] >= limit:  # type: ignore[index]
                return (
                    "The assistant's monthly budget is used up. Try again next month."
                )
        if (limit := settings.limit_user_usd_per_week) is not None:
            cur = await conn.execute(
                "SELECT coalesce(sum(cost_usd), 0) AS s FROM usage "
                "WHERE user_id = %s AND created_at >= now() - interval '7 days'",
                (user_id,),
            )
            if (await cur.fetchone())["s"] >= limit:  # type: ignore[index]
                return "You've reached your weekly limit. Try again in a few days."
        if (limit := settings.limit_user_runs_per_hour) is not None:
            # From usage, not runs: runs go away with their thread.
            cur = await conn.execute(
                "SELECT count(DISTINCT run_id) AS n FROM usage "
                "WHERE user_id = %s AND created_at >= now() - interval '1 hour'",
                (user_id,),
            )
            if (await cur.fetchone())["n"] >= limit:  # type: ignore[index]
                return "Too many messages in the last hour. Try again later."
    return None
