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
    call_type: str  # "user_request" | "tool_use" | "summarizer"
    input_tokens: int
    output_tokens: int
    model_name: Optional[str] = None


class UsageTracker:
    """Collects per-request token usage from message history and generates charts."""

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
        """Classify a ModelResponse as tool_use or user_request."""
        for part in response.parts:
            if hasattr(part, "tool_name"):
                return "tool_use"
        return "user_request"

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

    def plot(self, title: Optional[str] = None, save_path: Optional[str] = None):
        """Generate an interactive dot plot of per-call token usage.

        X-axis: sequential call index (grouped by run)
        Y-axis: token count
        Colors by call_type, markers by token type (input vs output)
        """
        import plotly.graph_objects as go

        title = title or f"Token Usage - {self.name}"

        color_map = {
            "user_request": "#3b82f6",
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

            color = color_palette[i % len(color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=runs,
                    y=tokens,
                    mode="lines+markers",
                    name=strategy_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    hovertemplate=(
                        f"<b>{strategy_name}</b><br>"
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

            color = color_palette[i % len(color_palette)]
            fig.add_trace(
                go.Bar(
                    x=runs,
                    y=tokens,
                    name=strategy_name,
                    marker_color=color,
                    hovertemplate=(
                        f"<b>{strategy_name}</b><br>"
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
