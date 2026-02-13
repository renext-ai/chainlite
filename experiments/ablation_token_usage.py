"""
Ablation study: compare token consumption across truncation strategies.

Strategies:
  1. no_truncation    — no history truncation at all
  2. post_run_compaction — post-run summarization/truncation after each run
  3. in_run_compaction  — in-run compaction (compress N-1 while N executes)

Uses real LLM calls via OPENAI_API_KEY from .env.
Generates charts in a per-run output directory.
"""
import os
import sys
import asyncio
import argparse
import json
import hashlib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from chainlite import ChainLite, ChainLiteConfig
from chainlite.usage_tracker import UsageTracker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_NUM_RUNS = 5
DEFAULT_TOOL_OUTPUT_SIZE = 2000  # chars per tool output
DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_RESULTS_ROOT = Path(__file__).parent / "results"
DEFAULT_POST_THRESHOLD = 500
DEFAULT_POST_START_RUN = 1
DEFAULT_LAZY_THRESHOLD = 5000
DEFAULT_LAZY_SUMM_THRESHOLD = 500
DEFAULT_LAZY_START_ITER = 2
DEFAULT_LAZY_START_RUN = 1
DEFAULT_LAZY_MAX_CONCURRENCY = 4
DEFAULT_TOOL_CALLS_PER_RUN = 1
DEFAULT_MAX_CONTEXT_TOKENS = 128000
DEFAULT_CONTEXT_SAFETY_MARGIN = 0.8

SYSTEM_PROMPT = (
    "You are a research assistant. When the user asks a question, "
    "ALWAYS use the search_database tool to find information, then summarize the results. "
    "Always call the tool before answering."
)

DEFAULT_QUESTIONS = [
    "What are the latest advances in quantum computing and why do they matter?",
    "Summarize climate change impacts on agriculture in arid regions.",
    "What are the key developments in AI safety research this year?",
    "Explain recent breakthroughs in fusion energy commercialization.",
    "What are the most important updates in space exploration programs?",
    "Compare neutral-atom and superconducting approaches in quantum computing.",
    "How are farmers adapting to climate stress in crop planning?",
    "List practical AI safety evaluation methods used in industry.",
    "What technical blockers remain for fusion pilot plants?",
    "How have lunar missions shifted national space priorities?",
    "Give a timeline of quantum error-correction milestones.",
    "What does climate volatility mean for food price stability?",
    "What are alignment risks in multi-agent LLM systems?",
    "Compare tokamak and stellarator engineering tradeoffs.",
    "What are near-term objectives for Mars exploration?",
    "Which quantum hardware vendors are gaining momentum?",
    "What mitigation policies most improve agricultural resilience?",
    "Which AI safety proposals are least costly to deploy?",
    "How close are private companies to net-electric fusion?",
    "What are the latest small-satellite launch ecosystem trends?",
    "What are common failure modes in quantum experiments?",
    "How does heat stress affect yield and irrigation decisions?",
    "What does red-teaming reveal about model weaknesses?",
    "How do fusion supply chains constrain deployment speed?",
    "What scientific payloads are prioritized for deep-space missions?",
    "Which quantum algorithms show practical utility today?",
    "How do insurance markets react to climate crop risk?",
    "What governance frameworks exist for frontier AI safety?",
    "What economics assumptions drive fusion startup projections?",
    "How are agencies coordinating international space efforts?",
]

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------
LENGTH_BUCKETS = [500, 2000, 6000, 12000]
FORMAT_BUCKETS = ["narrative", "json", "mixed"]
NOISE_BUCKETS = [0.0, 0.1, 0.2]

TOPIC_KEYWORDS = {
    "quantum": ["quantum", "qubit", "error-correction", "superconducting"],
    "climate": ["climate", "agricultur", "crop", "irrigation", "drought"],
    "ai": ["ai safety", "alignment", "red-team", "frontier model", "llm"],
    "fusion": ["fusion", "tokamak", "stellarator", "ignition", "plasma"],
    "space": ["space", "lunar", "mars", "satellite", "jwst", "artemis"],
}


def make_fake_db_results(tool_output_size: int) -> dict[str, dict]:
    _ = tool_output_size  # keep signature stable for experiment wiring
    return {
        "quantum": {
            "title": "Quantum Computing Research Digest",
            "facts": [
                "Error-correction experiments improved logical qubit stability under repeated cycles.",
                "Hardware teams reported better gate fidelity and calibration automation.",
                "Hybrid quantum-classical workflows were prioritized for near-term utility.",
                "Compilation and routing optimizations reduced depth overhead for benchmark circuits.",
                "Cloud providers expanded managed access to neutral-atom and trapped-ion backends.",
                "Benchmark quality controls increasingly report uncertainty and reproducibility metrics.",
            ],
        },
        "climate": {
            "title": "Climate and Agriculture Impact Brief",
            "facts": [
                "Yield variability increased under compound heat and water stress conditions.",
                "Irrigation scheduling tools improved outcomes where local forecasts were reliable.",
                "Soil moisture preservation and regenerative practices helped reduce loss volatility.",
                "Insurance and credit terms were adjusted in high-risk climate-exposed regions.",
                "Seed selection diversified toward heat and drought tolerance in staple crops.",
                "Precision sensing improved input efficiency but adoption varied by farm scale.",
            ],
        },
        "ai": {
            "title": "AI Safety and Alignment Update",
            "facts": [
                "Red-teaming programs emphasized jailbreak resistance and misuse scenario coverage.",
                "Mechanistic interpretability focused on tracing high-impact internal feature circuits.",
                "Policy work advanced model release governance and incident response playbooks.",
                "Evaluation suites expanded to include deception, robustness, and long-horizon tasks.",
                "Tool-use safety research examined escalation, self-reflection failure, and oversight gaps.",
                "Post-training alignment techniques compared preference optimization tradeoffs in production.",
            ],
        },
        "fusion": {
            "title": "Fusion Energy Engineering Snapshot",
            "facts": [
                "High-field magnet performance and manufacturing reliability remained critical milestones.",
                "Materials endurance under neutron flux constrained component replacement cycles.",
                "Plasma control software improved stability windows in experimental campaigns.",
                "Fuel cycle, tritium handling, and safety compliance shaped deployment timelines.",
                "Balance-of-plant economics became a major differentiator across startup roadmaps.",
                "Grid-integration assumptions influenced total system value projections.",
            ],
        },
        "space": {
            "title": "Space Exploration Program Overview",
            "facts": [
                "Launch cadence increased for small satellites and responsive mission architectures.",
                "Lunar program plans emphasized surface logistics and sustainable mission operations.",
                "Deep-space science prioritized high-value instrument payload scheduling.",
                "Mars planning balanced sample-return complexity with near-term robotic missions.",
                "International collaboration frameworks expanded for mission interoperability.",
                "Ground-segment modernization improved data delivery and mission observability.",
            ],
        },
    }


def _repeat_to_size(text: str, target_size: int) -> str:
    if len(text) >= target_size:
        return text[:target_size]
    filler = "\n".join(
        [
            "appendix: baseline assumptions, confidence bands, and caveats.",
            "appendix: methodology notes, instrument limitations, and data revisions.",
            "appendix: comparable prior studies and unresolved disagreements.",
        ]
    )
    chunks = [text]
    while sum(len(c) for c in chunks) < target_size:
        chunks.append("\n" + filler)
    return "".join(chunks)[:target_size]


def _make_noise_block(topic: str, size: int) -> str:
    if size <= 0:
        return ""
    lines = [
        f"noise-{topic}: duplicated telemetry packet checksum mismatch resolved.",
        f"noise-{topic}: archival index references stale cache key and retries.",
        f"noise-{topic}: low-priority monitoring heartbeat and non-actionable logs.",
    ]
    return _repeat_to_size("\n".join(lines), size)


def _pick_topic(query_lower: str, fake_db_results: dict[str, dict]) -> str:
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(k in query_lower for k in keywords):
            return topic
    for topic in fake_db_results:
        if topic in query_lower:
            return topic
    return "ai"


def _format_payload(topic: str, title: str, facts: list[str], fmt: str) -> str:
    if fmt == "json":
        payload = {
            "topic": topic,
            "title": title,
            "items": [{"id": i + 1, "finding": fact} for i, fact in enumerate(facts)],
            "meta": {
                "source_quality": "mixed",
                "update_cycle": "weekly",
                "confidence": "medium",
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if fmt == "mixed":
        table_header = "| id | category | note |\n|---:|---|---|"
        table_rows = "\n".join(
            [f"| {i+1} | key-finding | {fact} |" for i, fact in enumerate(facts[:4])]
        )
        bullets = "\n".join([f"- {fact}" for fact in facts[4:]])
        return f"{title}\n\n{table_header}\n{table_rows}\n\nObservations:\n{bullets}"

    body = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])
    return f"{title}\n{body}"


def get_search_result(
    query: str, fake_db_results: dict[str, dict], tool_output_size: int
) -> str:
    """Generate query-dependent fake tool output with variable shape and size."""
    query_lower = query.lower()
    topic = _pick_topic(query_lower, fake_db_results)
    template = fake_db_results[topic]

    digest = int(hashlib.sha256(query_lower.encode("utf-8")).hexdigest(), 16)
    length_bucket = LENGTH_BUCKETS[digest % len(LENGTH_BUCKETS)]
    format_bucket = FORMAT_BUCKETS[(digest // 8) % len(FORMAT_BUCKETS)]
    noise_ratio = NOISE_BUCKETS[(digest // 64) % len(NOISE_BUCKETS)]

    scaled_target = int(length_bucket * (tool_output_size / 2000))
    target_size = max(350, scaled_target)

    core = _format_payload(
        topic=topic,
        title=template["title"],
        facts=template["facts"],
        fmt=format_bucket,
    )
    noise_size = int(target_size * noise_ratio)
    if noise_size > 0:
        core = f"{core}\n\nNoise segment:\n{_make_noise_block(topic, noise_size)}"
    return _repeat_to_size(core, target_size)


def sanitize_for_path(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in text)


def _resolve_chart_path(requested_path: Path) -> Path:
    """Return actual chart path, falling back to HTML when image export fails."""
    if requested_path.exists():
        return requested_path
    fallback = requested_path.with_suffix(".html")
    if fallback.exists():
        return fallback
    return requested_path


def _safe_tool_output_size(
    requested_size: int,
    num_runs: int,
    tool_calls_per_run: int,
    max_context_tokens: int,
    context_safety_margin: float,
) -> int:
    """Estimate a safe upper bound for tool output size (chars).

    Conservative heuristic based on no-truncation worst case:
      approx_tokens_in_history ~= (num_runs - 1) * tool_calls_per_run * (chars / 4)
    """
    if num_runs <= 1 or tool_calls_per_run <= 0:
        return requested_size

    budget_tokens = int(max_context_tokens * context_safety_margin)
    denom = (num_runs - 1) * tool_calls_per_run
    if denom <= 0:
        return requested_size

    safe_chars = int((budget_tokens * 4) / denom)
    safe_chars = max(300, safe_chars)
    return min(requested_size, safe_chars)


def build_configs(args: argparse.Namespace) -> dict[str, dict]:
    system_prompt = build_system_prompt(args.tool_calls_per_run)

    return {
        "no_truncation": {
            "llm_model_name": args.model,
            "system_prompt": system_prompt,
            "use_history": True,
            "session_id": "ablation_none",
        },
        "post_run_compaction_auto": {
            "llm_model_name": args.model,
            "system_prompt": system_prompt,
            "use_history": True,
            "session_id": "ablation_post_auto",
            "history_truncator_config": {
                "post_run_compaction": {
                    "mode": "auto",
                    "truncation_threshold": args.post_threshold,
                    "start_run": args.post_start_run,
                },
            },
        },
        "in_run_compaction_auto": {
            "llm_model_name": args.model,
            "system_prompt": system_prompt,
            "use_history": True,
            "session_id": "ablation_in_run_auto",
            "history_truncator_config": {
                "post_run_compaction": {
                    "mode": "auto",
                    "truncation_threshold": args.lazy_threshold,
                    "start_run": args.post_start_run,
                },
                "in_run_compaction": {
                    "mode": "auto",
                    "truncation_threshold": args.lazy_summ_threshold,
                    "lazy_start_iter": args.lazy_start_iter,
                    "lazy_start_run": args.lazy_start_run,
                    "max_concurrency": args.lazy_max_concurrency,
                },
            },
        },
    }


def build_system_prompt(tool_calls_per_run: int) -> str:
    if tool_calls_per_run <= 1:
        return SYSTEM_PROMPT
    return (
        "You are a research assistant. "
        "Before finalizing any answer, you MUST call the search_database tool "
        f"exactly {tool_calls_per_run} times with distinct focused queries. "
        "After completing all required tool calls, synthesize the findings."
    )


def build_run_question(question: str, tool_calls_per_run: int) -> str:
    if tool_calls_per_run <= 1:
        return question
    return (
        f"{question}\n\n"
        "Important instructions:\n"
        f"- Call search_database exactly {tool_calls_per_run} times.\n"
        "- Each call must use a distinct query angle.\n"
        "- Only produce final answer after all tool calls complete."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study for token usage across truncation strategies."
    )
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--tool-output-size", type=int, default=DEFAULT_TOOL_OUTPUT_SIZE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--post-threshold", type=int, default=DEFAULT_POST_THRESHOLD)
    parser.add_argument("--post-start-run", type=int, default=DEFAULT_POST_START_RUN)
    parser.add_argument("--lazy-threshold", type=int, default=DEFAULT_LAZY_THRESHOLD)
    parser.add_argument(
        "--lazy-summ-threshold", type=int, default=DEFAULT_LAZY_SUMM_THRESHOLD
    )
    parser.add_argument("--lazy-start-iter", type=int, default=DEFAULT_LAZY_START_ITER)
    parser.add_argument("--lazy-start-run", type=int, default=DEFAULT_LAZY_START_RUN)
    parser.add_argument(
        "--lazy-max-concurrency", type=int, default=DEFAULT_LAZY_MAX_CONCURRENCY
    )
    parser.add_argument(
        "--tool-calls-per-run", type=int, default=DEFAULT_TOOL_CALLS_PER_RUN
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS
    )
    parser.add_argument(
        "--context-safety-margin", type=float, default=DEFAULT_CONTEXT_SAFETY_MARGIN
    )
    parser.add_argument(
        "--disable-context-guard",
        action="store_true",
        help="Disable automatic tool_output_size capping for context safety.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
async def run_experiment(
    config_name: str,
    config_dict: dict,
    tracker: UsageTracker,
    num_runs: int,
    questions: list[str],
    tool_calls_per_run: int,
    fake_db_results: dict[str, str],
    tool_output_size: int,
) -> "ChainLite":
    """Run NUM_RUNS turns with a given config and collect token usage."""
    print(f"\n{'='*60}")
    print(f"  Strategy: {config_name}")
    print(f"{'='*60}")

    config = ChainLiteConfig(**config_dict)
    chain = ChainLite(config)

    # Register tool
    @chain.agent.tool_plain
    def search_database(query: str) -> str:
        """Search the research database for information on a given topic."""
        return get_search_result(query, fake_db_results, tool_output_size)

    prev_msg_count = 0

    for run_idx in range(1, num_runs + 1):
        base_question = questions[(run_idx - 1) % len(questions)]
        question = build_run_question(base_question, tool_calls_per_run)
        print(f"\n  Run {run_idx}: {question[:50]}...")

        try:
            result = await chain.arun({"input": question})
            print(f"    Answer: {str(result)[:80]}...")
        except Exception as e:
            msg = str(e)
            if "maximum context length" in msg.lower():
                print(
                    "    ERROR: context limit exceeded. "
                    "Try lowering --tool-output-size or enabling stronger truncation."
                )
            else:
                print(f"    ERROR: {e}")
            continue

        # Extract NEW messages from this run (delta from previous)
        all_raw = chain.history_manager._raw_messages
        new_messages = all_raw[prev_msg_count:]
        prev_msg_count = len(all_raw)

        # Track usage from new messages
        tracker.track_run_messages(new_messages, run_index=run_idx)

        # Print per-run stats
        run_records = [r for r in tracker.records if r.run_index == run_idx]
        total_input = sum(r.input_tokens for r in run_records)
        total_output = sum(r.output_tokens for r in run_records)
        print(
            f"    Tokens: input={total_input:,}, output={total_output:,}, "
            f"calls={len(run_records)}"
        )

    return chain


def _get_summarizer_stats(chain: ChainLite) -> dict:
    """Extract summarizer token usage from truncator(s) if they are AutoSummarizor."""
    from chainlite.truncators import AutoSummarizor

    stats = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    # Check post-run truncator
    if isinstance(chain._truncator, AutoSummarizor):
        stats["input_tokens"] += chain._truncator.summarizer_input_tokens
        stats["output_tokens"] += chain._truncator.summarizer_output_tokens
        stats["calls"] += chain._truncator.summarizer_calls

    # Check in-run compactor
    if isinstance(chain._in_run_compactor, AutoSummarizor):
        stats["input_tokens"] += chain._in_run_compactor.summarizer_input_tokens
        stats["output_tokens"] += chain._in_run_compactor.summarizer_output_tokens
        stats["calls"] += chain._in_run_compactor.summarizer_calls

    return stats


async def main(args: argparse.Namespace):
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found. Check .env file.")
        sys.exit(1)

    effective_tool_output_size = args.tool_output_size
    if not args.disable_context_guard:
        capped = _safe_tool_output_size(
            requested_size=args.tool_output_size,
            num_runs=args.num_runs,
            tool_calls_per_run=args.tool_calls_per_run,
            max_context_tokens=args.max_context_tokens,
            context_safety_margin=args.context_safety_margin,
        )
        if capped < args.tool_output_size:
            print(
                "WARNING: tool_output_size capped for context safety "
                f"({args.tool_output_size} -> {capped}). "
                "Use --disable-context-guard to bypass."
            )
            effective_tool_output_size = capped

    results_root = args.results_root
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"run_{run_stamp}_nr{args.num_runs}_tool{effective_tool_output_size}_"
        f"{sanitize_for_path(args.model)}"
    )
    results_dir = results_root / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    fake_db_results = make_fake_db_results(effective_tool_output_size)
    configs = build_configs(args)

    trackers: dict[str, UsageTracker] = {}
    chains: dict[str, ChainLite] = {}

    for config_name, config_dict in configs.items():
        tracker = UsageTracker(name=config_name)
        chain = await run_experiment(
            config_name=config_name,
            config_dict=config_dict,
            tracker=tracker,
            num_runs=args.num_runs,
            questions=DEFAULT_QUESTIONS,
            tool_calls_per_run=args.tool_calls_per_run,
            fake_db_results=fake_db_results,
            tool_output_size=effective_tool_output_size,
        )
        trackers[config_name] = tracker
        chains[config_name] = chain

    params_path = results_dir / "params.json"
    params = {
        "num_runs": args.num_runs,
        "tool_output_size": args.tool_output_size,
        "effective_tool_output_size": effective_tool_output_size,
        "model": args.model,
        "results_root": str(results_root),
        "output_dir": str(results_dir),
        "post_threshold": args.post_threshold,
        "post_start_run": args.post_start_run,
        "lazy_threshold": args.lazy_threshold,
        "lazy_summ_threshold": args.lazy_summ_threshold,
        "lazy_start_iter": args.lazy_start_iter,
        "lazy_start_run": args.lazy_start_run,
        "lazy_max_concurrency": args.lazy_max_concurrency,
        "tool_calls_per_run": args.tool_calls_per_run,
        "max_context_tokens": args.max_context_tokens,
        "context_safety_margin": args.context_safety_margin,
        "disable_context_guard": args.disable_context_guard,
        "questions": DEFAULT_QUESTIONS,
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved params: {params_path}")

    # Generate charts
    print(f"\n{'='*60}")
    print("  Generating charts...")
    print(f"{'='*60}")

    # Chart 1: Per-call dot plots for each strategy
    for name, tracker in trackers.items():
        path = results_dir / f"per_call_{name}.png"
        tracker.plot(
            title=f"Token Usage Per LLM Call - {name}",
            save_path=str(path),
        )
        print(f"  Saved: {_resolve_chart_path(path)}")

    # Chart 2: Cumulative comparison
    path_cumulative = results_dir / "ablation_cumulative.png"
    UsageTracker.plot_comparison(
        trackers,
        title="Ablation: Cumulative Input Tokens Per Run",
        save_path=str(path_cumulative),
    )
    print(f"  Saved: {_resolve_chart_path(path_cumulative)}")

    # Chart 3: Per-run bar comparison
    path_per_run = results_dir / "ablation_per_run.png"
    UsageTracker.plot_per_run_comparison(
        trackers,
        title="Ablation: Input Tokens Per Run",
        save_path=str(path_per_run),
    )
    print(f"  Saved: {_resolve_chart_path(path_per_run)}")

    # Summary table
    print(f"\n{'='*60}")
    print("  Summary — Primary Agent")
    print(f"{'='*60}")
    print(f"  {'Strategy':<25} {'Total Input':>15} {'Total Output':>15} {'Calls':>8}")
    print(f"  {'-'*63}")
    for name, tracker in trackers.items():
        total_in = sum(r.input_tokens for r in tracker.records)
        total_out = sum(r.output_tokens for r in tracker.records)
        calls = len(tracker.records)
        print(f"  {name:<25} {total_in:>15,} {total_out:>15,} {calls:>8}")

    # Summarizer stats
    print(f"\n{'='*60}")
    print("  Summary — Summarizer Overhead (auto mode)")
    print(f"{'='*60}")
    print(f"  {'Strategy':<25} {'Summ Input':>15} {'Summ Output':>15} {'Summ Calls':>12}")
    print(f"  {'-'*67}")
    for name, chain in chains.items():
        stats = _get_summarizer_stats(chain)
        print(
            f"  {name:<25} {stats['input_tokens']:>15,} "
            f"{stats['output_tokens']:>15,} {stats['calls']:>12}"
        )

    # Grand total
    print(f"\n{'='*60}")
    print("  Grand Total (Primary + Summarizer)")
    print(f"{'='*60}")
    print(f"  {'Strategy':<25} {'Grand Input':>15} {'Grand Output':>15}")
    print(f"  {'-'*55}")
    for name in trackers:
        primary_in = sum(r.input_tokens for r in trackers[name].records)
        primary_out = sum(r.output_tokens for r in trackers[name].records)
        summ = _get_summarizer_stats(chains[name])
        grand_in = primary_in + summ["input_tokens"]
        grand_out = primary_out + summ["output_tokens"]
        print(f"  {name:<25} {grand_in:>15,} {grand_out:>15,}")

    print(f"\nResults saved to: {results_dir.resolve()}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
