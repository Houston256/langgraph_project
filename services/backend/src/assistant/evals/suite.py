"""Small model comparison on real shop questions.

uv run python -m assistant.evals.suite openai/gpt-6-luna [--set <name>]
Cases come from the local, gitignored evals/*.py (--dir); see evals/README.md.
Writes a JSON report with transcripts next to the summary it prints.
"""

import argparse
import asyncio
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import AsyncQdrantClient

from assistant.evals.case import Case
from assistant.evals.checks import Observed, card_payloads, check
from assistant.graphs.db_agent import create_db_agent
from assistant.models import embeddings
from assistant.search.qdrant import create_product_tools, create_qdrant_client
from assistant.usage import Pricing

PARALLEL_CASES = 4


def load_sets(directory: Path) -> dict[str, list[Case]]:
    """One set per *.py file in the directory, from its CASES list."""
    sets = {}
    for path in sorted(directory.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"evals.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        sets[path.stem] = module.CASES
    return sets


async def run_case(
    agent, qdrant: AsyncQdrantClient, pricing: Pricing, model: str, case: Case
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": f"{case.name}-{time.time()}"}}
    seen = Observed()
    error = None
    start = time.monotonic()
    try:
        known = 0
        for turn in case.turns:
            state = await agent.ainvoke({"messages": [HumanMessage(turn)]}, config)
            seen.add_turn(turn, state["messages"][known:])
            known = len(state["messages"])
    except Exception as e:  # noqa: BLE001  any failure becomes a failed case
        error = f"{type(e).__name__}: {str(e)[:300]}"
    seconds = time.monotonic() - start

    shown = await card_payloads(qdrant, seen.cards)
    cost = pricing.cost(
        model, seen.input_tokens, seen.cached_tokens, seen.output_tokens
    )
    return {
        "case": case.name,
        "seconds": round(seconds, 1),
        "cost": cost or 0.0,
        "fails": [error] if error else check(case, seen, shown),
        "shown": shown,
        **asdict(seen),
        "tools_used": sorted(seen.tools_used),
    }


async def run_model(
    model: str, pricing: Pricing, qdrant: AsyncQdrantClient, tools, cases: list[Case]
) -> list[dict]:
    agent = create_db_agent(tools, InMemorySaver(), model_name=model)
    slots = asyncio.Semaphore(PARALLEL_CASES)

    async def one(case: Case) -> dict:
        async with slots:
            return await run_case(agent, qdrant, pricing, model, case)

    return await asyncio.gather(*(one(c) for c in cases))


def print_summary(model: str, results: list[dict]) -> None:
    passed = sum(not r["fails"] for r in results)
    total_cost = sum(r["cost"] for r in results)
    avg_seconds = sum(r["seconds"] for r in results) / len(results)
    print(
        f"\n== {model}: {passed}/{len(results)} passed, "
        f"${total_cost:.4f} total, {avg_seconds:.1f}s avg per case"
    )
    for r in results:
        status = "FAIL" if r["fails"] else "ok  "
        print(
            f"  {status} {r['case']:38} {r['seconds']:5.1f}s ${r['cost']:.4f} "
            f"out={r['output_tokens']:5} calls={r['tool_calls']:2} cards={len(r['shown'])} "
            f"tools={','.join(r['tools_used'])}  {'; '.join(r['fails'])[:150]}"
        )


async def main(models: list[str], out: Path, cases: list[Case]) -> None:
    pricing = Pricing()
    await pricing.refresh()
    if unknown := [m for m in models if pricing.get(m) is None]:
        raise SystemExit(f"no OpenRouter price for {', '.join(unknown)}")
    qdrant = create_qdrant_client()
    tools = create_product_tools(qdrant, embeddings())
    report = {}
    try:
        for model in models:
            report[model] = await run_model(model, pricing, qdrant, tools, cases)
            print_summary(model, report[model])
    finally:
        await qdrant.close()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\ntranscripts: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--dir", type=Path, default=Path("evals"))
    parser.add_argument("--out", type=Path, default=Path("eval_report.json"))
    parser.add_argument("--set", default="all")
    args = parser.parse_args()
    sets = load_sets(args.dir)
    if not sets:
        raise SystemExit(f"no cases in {args.dir}, see evals/README.md")
    if args.set not in [*sets, "all"]:
        parser.error(f"--set: one of {', '.join([*sets, 'all'])}")
    cases = (
        [c for s in sets.values() for c in s] if args.set == "all" else sets[args.set]
    )
    asyncio.run(main(args.models, args.out, cases))
