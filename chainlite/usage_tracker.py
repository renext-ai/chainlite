"""Token usage tracking and visualization for ChainLite runs."""
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
import warnings
from pydantic_ai.messages import ModelMessage, ModelResponse


@dataclass
class CallRecord:
    """A single LLM API call record."""

    run_index: int  # which agent.run() call (1-based)
    step_index: int  # which LLM request within the run (1-based)
    call_type: str  # "final_answer" | "tool_use" | "summarizer"
    input_tokens: int
    output_tokens: int
    model_name: Optional[str] = None


class UsageTracker:
    """Collects per-request token usage from message history and generates charts."""

    _STRATEGY_SHORT_NAMES: Dict[str, str] = {
        "no_truncation": "No Trunc",
        "post_run_compaction_auto": "Post-Run",
        "post_and_in_run_compaction_auto": "Post+In-Run",
    }

    @staticmethod
    def _short_name(name: str) -> str:
        return UsageTracker._STRATEGY_SHORT_NAMES.get(name, name)

    def __init__(self, name: str = "default"):
        self.name = name
        self.records: List[CallRecord] = []

    @staticmethod
    def _save_figure(fig, save_path: str) -> None:
        """Save figure based on file extension.

        Supported:
            - .html (interactive)
            - Plotly static formats: .png/.jpg/.jpeg/.webp/.svg/.pdf/.eps
        """
        path = Path(save_path)
        suffix = path.suffix.lower()

        if suffix == ".html":
            fig.write_html(str(path))
            return

        static_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf", ".eps"}
        if suffix in static_suffixes:
            try:
                fig.write_image(str(path))
            except Exception as exc:  # pragma: no cover - dependency/runtime specific
                fallback_path = path.with_suffix(".html")
                fig.write_html(str(fallback_path))
                warnings.warn(
                    (
                        "Static image export failed; saved interactive HTML instead: "
                        f"{fallback_path}. Install plotly image backend "
                        "(e.g. `pip install -U kaleido`) to enable image export."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            return

        raise ValueError(
            f"Unsupported chart output format: '{suffix}'. "
            "Use .html or an image extension like .png."
        )

    def track_run_messages(
        self, messages: List[ModelMessage], run_index: int
    ) -> None:
        """Extract per-request usage from ModelResponse objects in a run's messages.

        Args:
            messages: The NEW messages from this run (not cumulative history).
            run_index: 1-based index of the agent.run() call.
        """
        step = 0
        for msg in messages:
            if isinstance(msg, ModelResponse) and msg.usage:
                step += 1
                call_type = self._classify_call(msg)
                self.records.append(
                    CallRecord(
                        run_index=run_index,
                        step_index=step,
                        call_type=call_type,
                        input_tokens=msg.usage.input_tokens,
                        output_tokens=msg.usage.output_tokens,
                        model_name=msg.model_name,
                    )
                )

    @staticmethod
    def _classify_call(response: ModelResponse) -> str:
        """Classify a ModelResponse as tool_use or final_answer."""
        for part in response.parts:
            if hasattr(part, "tool_name"):
                return "tool_use"
        return "final_answer"

    def get_cumulative_input_tokens_per_run(self) -> Dict[int, int]:
        """Get cumulative input tokens up to each run."""
        per_run: Dict[int, int] = {}
        cumulative = 0
        for rec in sorted(self.records, key=lambda r: (r.run_index, r.step_index)):
            cumulative += rec.input_tokens
            per_run[rec.run_index] = cumulative
        return per_run

    def get_per_run_input_tokens(self) -> Dict[int, int]:
        """Get total input tokens for each individual run."""
        per_run: Dict[int, int] = {}
        for rec in self.records:
            per_run[rec.run_index] = per_run.get(rec.run_index, 0) + rec.input_tokens
        return per_run

    def get_per_run_input_output_tokens(self) -> Dict[int, Dict[str, int]]:
        """Get total input/output tokens for each individual run."""
        per_run: Dict[int, Dict[str, int]] = {}
        for rec in self.records:
            if rec.run_index not in per_run:
                per_run[rec.run_index] = {"input": 0, "output": 0}
            per_run[rec.run_index]["input"] += rec.input_tokens
            per_run[rec.run_index]["output"] += rec.output_tokens
        return per_run

    def get_per_run_tokens_by_call_type(self) -> Dict[int, Dict[str, Dict[str, int]]]:
        """Get per-run token/call breakdown by call_type.

        Returns:
            {
              run_index: {
                call_type: {"input": int, "output": int, "calls": int}
              }
            }
        """
        per_run: Dict[int, Dict[str, Dict[str, int]]] = {}
        for rec in self.records:
            if rec.run_index not in per_run:
                per_run[rec.run_index] = {}
            if rec.call_type not in per_run[rec.run_index]:
                per_run[rec.run_index][rec.call_type] = {
                    "input": 0,
                    "output": 0,
                    "calls": 0,
                }
            bucket = per_run[rec.run_index][rec.call_type]
            bucket["input"] += rec.input_tokens
            bucket["output"] += rec.output_tokens
            bucket["calls"] += 1
        return per_run

    def plot(self, title: Optional[str] = None, save_path: Optional[str] = None):
        """Generate an interactive dot plot of per-call token usage.

        X-axis: sequential call index (grouped by run)
        Y-axis: token count
        Colors by call_type, markers by token type (input vs output)
        """
        import plotly.graph_objects as go

        title = title or f"Token Usage - {self.name}"

        color_map = {
            "final_answer": "#3b82f6",
            "tool_use": "#f97316",
            "summarizer": "#ef4444",
        }

        fig = go.Figure()

        # Sort records by run then step
        sorted_records = sorted(
            self.records, key=lambda r: (r.run_index, r.step_index)
        )

        # Build x-axis labels and positions
        call_idx = 0
        run_boundaries = []
        prev_run = None

        x_positions = []
        for rec in sorted_records:
            if prev_run is not None and rec.run_index != prev_run:
                run_boundaries.append(call_idx - 0.5)
            prev_run = rec.run_index
            x_positions.append(call_idx)
            call_idx += 1

        # Group by (call_type, token_type) for legend
        traces = {}
        for i, rec in enumerate(sorted_records):
            for token_type, value, symbol in [
                ("input", rec.input_tokens, "circle"),
                ("output", rec.output_tokens, "diamond"),
            ]:
                key = (rec.call_type, token_type)
                if key not in traces:
                    traces[key] = {"x": [], "y": [], "text": []}
                traces[key]["x"].append(x_positions[i])
                traces[key]["y"].append(value)
                traces[key]["text"].append(
                    f"Run {rec.run_index}, Step {rec.step_index}<br>"
                    f"Type: {rec.call_type}<br>"
                    f"{token_type}_tokens: {value}"
                )

        for (call_type, token_type), data in traces.items():
            symbol = "circle" if token_type == "input" else "diamond"
            fig.add_trace(
                go.Scatter(
                    x=data["x"],
                    y=data["y"],
                    mode="markers",
                    name=f"{call_type} ({token_type})",
                    marker=dict(
                        color=color_map.get(call_type, "#888"),
                        symbol=symbol,
                        size=10,
                    ),
                    text=data["text"],
                    hoverinfo="text",
                )
            )

        # Add run boundary lines
        for boundary in run_boundaries:
            fig.add_vline(
                x=boundary, line_dash="dash", line_color="gray", opacity=0.5
            )

        # Add run labels
        run_starts = {}
        run_ends = {}
        for i, rec in enumerate(sorted_records):
            if rec.run_index not in run_starts:
                run_starts[rec.run_index] = x_positions[i]
            run_ends[rec.run_index] = x_positions[i]

        for run_idx in run_starts:
            mid = (run_starts[run_idx] + run_ends[run_idx]) / 2
            fig.add_annotation(
                x=mid,
                y=-0.08,
                yref="paper",
                text=f"Run {run_idx}",
                showarrow=False,
                font=dict(size=11, color="gray"),
            )

        fig.update_layout(
            title=title,
            xaxis_title="LLM Call Index",
            yaxis_title="Token Count",
            hovermode="closest",
            template="plotly_white",
            xaxis=dict(showticklabels=False),
        )

        if save_path:
            self._save_figure(fig, save_path)
        return fig

    @staticmethod
    def plot_comparison(
        trackers: Dict[str, "UsageTracker"],
        title: str = "Ablation: Cumulative Input Tokens Per Run",
        save_path: Optional[str] = None,
    ):
        """Generate comparison chart of cumulative input tokens across strategies.

        Args:
            trackers: dict mapping strategy name to UsageTracker.
            title: chart title.
            save_path: optional path to save chart (.html or image extension).
        """
        import plotly.graph_objects as go

        color_palette = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316"]
        fig = go.Figure()

        for i, (strategy_name, tracker) in enumerate(trackers.items()):
            cumulative = tracker.get_cumulative_input_tokens_per_run()
            runs = sorted(cumulative.keys())
            tokens = [cumulative[r] for r in runs]

            display_name = UsageTracker._short_name(strategy_name)
            color = color_palette[i % len(color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=runs,
                    y=tokens,
                    mode="lines+markers",
                    name=display_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        "Run %{x}<br>"
                        "Cumulative input tokens: %{y:,}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Run Number",
            yaxis_title="Cumulative Input Tokens",
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(dtick=1),
        )

        if save_path:
            UsageTracker._save_figure(fig, save_path)
        return fig

    @staticmethod
    def plot_per_run_comparison(
        trackers: Dict[str, "UsageTracker"],
        title: str = "Ablation: Input Tokens Per Run",
        save_path: Optional[str] = None,
    ):
        """Generate bar chart comparing per-run input tokens across strategies.

        Args:
            trackers: dict mapping strategy name to UsageTracker.
            title: chart title.
            save_path: optional path to save chart (.html or image extension).
        """
        import plotly.graph_objects as go

        color_palette = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316"]
        fig = go.Figure()

        for i, (strategy_name, tracker) in enumerate(trackers.items()):
            per_run = tracker.get_per_run_input_tokens()
            runs = sorted(per_run.keys())
            tokens = [per_run[r] for r in runs]

            display_name = UsageTracker._short_name(strategy_name)
            color = color_palette[i % len(color_palette)]
            fig.add_trace(
                go.Bar(
                    x=runs,
                    y=tokens,
                    name=display_name,
                    marker_color=color,
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        "Run %{x}<br>"
                        "Input tokens: %{y:,}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Run Number",
            yaxis_title="Input Tokens",
            barmode="group",
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(dtick=1),
        )

        if save_path:
            UsageTracker._save_figure(fig, save_path)
        return fig

    @staticmethod
    def plot_per_run_stacked_strategy_io(
        trackers: Dict[str, "UsageTracker"],
        title: str = "Ablation: Per-Run Tokens (Stacked by Strategy, Input/Output)",
        save_path: Optional[str] = None,
    ):
        """Generate a stacked per-run chart.

        For each run, the total bar is composed of strategy segments.
        Each strategy is split into dark(input)/light(output) shades.
        """
        import plotly.graph_objects as go

        strategy_colors = {
            "no_truncation": ("#b91c1c", "#fca5a5"),
            "post_run_compaction_auto": ("#1d4ed8", "#93c5fd"),
            "post_and_in_run_compaction_auto": ("#15803d", "#86efac"),
        }
        fallback_palette = [
            ("#7c3aed", "#c4b5fd"),
            ("#c2410c", "#fdba74"),
            ("#0f766e", "#99f6e4"),
        ]

        all_runs = sorted(
            {
                run_idx
                for tracker in trackers.values()
                for run_idx in tracker.get_per_run_input_output_tokens().keys()
            }
        )
        fig = go.Figure()

        for i, (strategy_name, tracker) in enumerate(trackers.items()):
            per_run = tracker.get_per_run_input_output_tokens()
            input_tokens = [per_run.get(r, {}).get("input", 0) for r in all_runs]
            output_tokens = [per_run.get(r, {}).get("output", 0) for r in all_runs]

            if strategy_name in strategy_colors:
                input_color, output_color = strategy_colors[strategy_name]
            else:
                input_color, output_color = fallback_palette[i % len(fallback_palette)]

            display_name = UsageTracker._short_name(strategy_name)
            fig.add_trace(
                go.Bar(
                    x=all_runs,
                    y=input_tokens,
                    name=f"{display_name} (input)",
                    marker_color=input_color,
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        "Run %{x}<br>"
                        "Input tokens: %{y:,}<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Bar(
                    x=all_runs,
                    y=output_tokens,
                    name=f"{display_name} (output)",
                    marker_color=output_color,
                    hovertemplate=(
                        f"<b>{display_name}</b><br>"
                        "Run %{x}<br>"
                        "Output tokens: %{y:,}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Run Number",
            yaxis_title="Tokens",
            barmode="stack",
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(dtick=1),
        )

        if save_path:
            UsageTracker._save_figure(fig, save_path)
        return fig

    @staticmethod
    def plot_per_run_strategy_calltype_segments(
        trackers: Dict[str, "UsageTracker"],
        summarizer_per_run: Dict[str, Dict[int, Dict[str, int]]],
        title: str = "Ablation: Per-Run Token Mix by Strategy",
        save_path: Optional[str] = None,
    ):
        """Generate stacked bars faceted by run, with strategies as x-axis categories.

        Each subplot represents one run.  Within each subplot, bars are grouped
        by strategy (using short display names) and stacked by segment:
        - tool_use input/output
        - final_answer input/output
        - summarizer input/output
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        strategy_order = list(trackers.keys())
        all_runs = sorted(
            {
                run_idx
                for tracker in trackers.values()
                for run_idx in tracker.get_per_run_tokens_by_call_type().keys()
            }
        )

        segment_specs = [
            ("tool_use", "input", "#b45309"),
            ("tool_use", "output", "#fdba74"),
            ("final_answer", "input", "#1d4ed8"),
            ("final_answer", "output", "#93c5fd"),
            ("summarizer", "input", "#15803d"),
            ("summarizer", "output", "#86efac"),
        ]

        # Pre-compute per-strategy data to avoid redundant calls.
        primary_data = {
            name: trackers[name].get_per_run_tokens_by_call_type()
            for name in strategy_order
        }

        num_runs = len(all_runs)
        fig = make_subplots(
            rows=1,
            cols=num_runs,
            subplot_titles=[f"Run {r}" for r in all_runs],
            shared_yaxes=True,
            horizontal_spacing=0.05,
        )

        x_labels = [UsageTracker._short_name(s) for s in strategy_order]

        for col_idx, run_idx in enumerate(all_runs, 1):
            for call_type, token_kind, color in segment_specs:
                y_vals = []
                customdata = []
                for strategy_name in strategy_order:
                    summarizer = summarizer_per_run.get(strategy_name, {})
                    if call_type == "summarizer":
                        run_data = summarizer.get(
                            run_idx,
                            {"input_tokens": 0, "output_tokens": 0, "calls": 0},
                        )
                        val = (
                            run_data["input_tokens"]
                            if token_kind == "input"
                            else run_data["output_tokens"]
                        )
                        calls = run_data["calls"]
                    else:
                        run_data = primary_data[strategy_name].get(
                            run_idx, {}
                        ).get(call_type, {"input": 0, "output": 0, "calls": 0})
                        val = (
                            run_data["input"]
                            if token_kind == "input"
                            else run_data["output"]
                        )
                        calls = run_data["calls"]
                    y_vals.append(val)
                    customdata.append(
                        [calls, strategy_name, call_type, token_kind]
                    )

                segment_name = f"{call_type} ({token_kind})"
                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=y_vals,
                        marker_color=color,
                        name=segment_name,
                        legendgroup=segment_name,
                        showlegend=(col_idx == 1),
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{customdata[1]}</b><br>"
                            "Segment: %{customdata[2]} (%{customdata[3]})<br>"
                            "Tokens: %{y:,}<br>"
                            "Calls: %{customdata[0]:,}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            barmode="stack",
            hovermode="closest",
            template="plotly_white",
            height=500,
            width=max(800, 300 * num_runs + 180),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            margin=dict(b=120),
        )

        fig.update_yaxes(title_text="Tokens", row=1, col=1)

        if save_path:
            UsageTracker._save_figure(fig, save_path)
        return fig
