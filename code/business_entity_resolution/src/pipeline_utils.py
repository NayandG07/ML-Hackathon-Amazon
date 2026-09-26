"""
pipeline_utils.py
=================
Shared console utilities for the Business Entity Resolution pipeline.

Provides:
  - Coloured Rich logging handler (replaces plain logging)
  - StageTimer  — context manager that prints a stage banner + elapsed time
  - track()     — tqdm/Rich progress bar wrapper
  - sysinfo()   — snapshot of CPU, RAM, GPU usage
  - PipelineProgress — top-level pipeline step tracker

Usage in any pipeline file:
    from pipeline_utils import console, log, StageTimer, track, sysinfo
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Iterable, Iterator, Optional, TypeVar

import psutil
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Shared Rich console (all modules import this same object)
# ---------------------------------------------------------------------------

THEME = Theme({
    "info":      "bold cyan",
    "success":   "bold green",
    "warning":   "bold yellow",
    "error":     "bold red",
    "stage":     "bold white on dark_blue",
    "metric":    "bold magenta",
    "dim_white": "dim white",
    "highlight": "bold bright_white",
})

console = Console(theme=THEME, highlight=False)

# ---------------------------------------------------------------------------
# Logging setup — redirect all `logging` calls to Rich
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a logger whose output goes through the Rich console."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
            log_time_format="[%H:%M:%S]",
        )
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

# Convenience — default pipeline logger
log = get_logger("pipeline")


# ---------------------------------------------------------------------------
# System info snapshot
# ---------------------------------------------------------------------------

def sysinfo(show: bool = True) -> dict:
    """
    Collect CPU / RAM / GPU stats and optionally print a Rich table.
    Returns a dict with the stats.
    """
    vm = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.3)
    stats = {
        "cpu_pct": cpu_pct,
        "ram_used_gb": vm.used / 1e9,
        "ram_total_gb": vm.total / 1e9,
        "ram_pct": vm.percent,
        "gpu_vram_used_gb": None,
        "gpu_vram_total_gb": None,
        "gpu_util_pct": None,
        "gpu_name": None,
    }

    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                stats["gpu_name"] = parts[0].strip()
                stats["gpu_util_pct"] = float(parts[1].strip())
                stats["gpu_vram_used_gb"] = float(parts[2].strip()) / 1024
                stats["gpu_vram_total_gb"] = float(parts[3].strip()) / 1024
    except Exception:
        pass

    if show:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=False,
            padding=(0, 1),
            title="[dim_white]System Resources[/dim_white]",
        )
        table.add_column("Resource", style="dim_white")
        table.add_column("Value", style="highlight")

        # CPU
        cpu_bar = _mini_bar(cpu_pct)
        table.add_row("CPU Usage", f"{cpu_bar}  {cpu_pct:.1f}%")

        # RAM
        ram_bar = _mini_bar(vm.percent)
        table.add_row(
            "RAM",
            f"{ram_bar}  {vm.used/1e9:.1f} / {vm.total/1e9:.1f} GB  ({vm.percent:.0f}%)",
        )

        # GPU
        if stats["gpu_name"]:
            vram_pct = stats["gpu_vram_used_gb"] / stats["gpu_vram_total_gb"] * 100
            vram_bar = _mini_bar(vram_pct)
            table.add_row("GPU", stats["gpu_name"])
            table.add_row(
                "GPU Util",
                f"{_mini_bar(stats['gpu_util_pct'])}  {stats['gpu_util_pct']:.0f}%",
            )
            table.add_row(
                "VRAM",
                f"{vram_bar}  {stats['gpu_vram_used_gb']:.1f} / {stats['gpu_vram_total_gb']:.1f} GB",
            )
        else:
            table.add_row("GPU", "[dim_white]Not detected[/dim_white]")

        console.print(table)

    return stats


def _mini_bar(pct: float, width: int = 10) -> str:
    """Return a simple ASCII progress bar."""
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    if pct >= 85:
        return f"[red]{bar}[/red]"
    if pct >= 60:
        return f"[yellow]{bar}[/yellow]"
    return f"[green]{bar}[/green]"


# ---------------------------------------------------------------------------
# Stage timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def StageTimer(name: str, show_sysinfo: bool = True) -> Iterator[None]:
    """
    Context manager that prints a stage banner before and a timing summary after.

    Usage:
        with StageTimer("Blocking — train set"):
            run_blocking(...)
    """
    width = console.width or 100

    # ── Banner ─────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(f"[stage]  {name}  [/stage]", style="blue", align="center"))
    console.print(
        f"[dim_white]  Started at [/dim_white][highlight]{datetime.now().strftime('%H:%M:%S')}[/highlight]"
    )

    if show_sysinfo:
        sysinfo(show=True)

    console.print()
    t0 = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        td = timedelta(seconds=int(elapsed))

        console.print()
        console.print(
            Panel(
                f"[success]OK  {name}[/success]\n"
                f"[dim_white]Elapsed: [/dim_white][metric]{td}[/metric]"
                f"[dim_white]  ({elapsed:.1f}s)[/dim_white]",

                box=box.ROUNDED,
                border_style="green",
                padding=(0, 2),
            )
        )
        console.print()


# ---------------------------------------------------------------------------
# Progress bar factory
# ---------------------------------------------------------------------------

def make_progress(**kwargs) -> Progress:
    """
    Return a Rich Progress bar with sensible defaults for long-running loops.
    Use as a context manager.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
        **kwargs,
    )


def track(
    iterable: Iterable[T],
    description: str = "Processing",
    total: Optional[int] = None,
) -> Iterator[T]:
    """
    Wrap an iterable with a Rich progress bar.

    Usage:
        for row in track(df.iter_rows(), "Featurising", total=len(df)):
            ...
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    with make_progress() as progress:
        task = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task)


# ---------------------------------------------------------------------------
# Pipeline-level tracker (printed at the very start of run_pipeline.py)
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    "Preprocessing",
    "Blocking (train)",
    "Feature Engineering",
    "GPU Encoding (train)",
    "Embedding Scoring (train)",
    "Training LightGBM",
    "Blocking (test)",
    "GPU Encoding (test)",
    "Embedding Scoring (test)",
    "Inference",
    "Format Validation",
]


def print_pipeline_overview(current_stage: Optional[str] = None) -> None:
    """Print a numbered checklist of all pipeline stages."""
    table = Table(
        title="[bold blue]Business Entity Resolution Pipeline[/bold blue]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("#", style="dim_white", width=3)
    table.add_column("Stage", width=35)
    table.add_column("Status", width=12)

    for i, stage in enumerate(PIPELINE_STAGES, 1):
        if current_stage and stage == current_stage:
            status = Text("▶ RUNNING", style="bold yellow")
            row_style = "yellow"
        elif current_stage and PIPELINE_STAGES.index(stage) < PIPELINE_STAGES.index(current_stage):
            status = Text("✓ Done", style="green")
            row_style = "dim"
        else:
            status = Text("○ Pending", style="dim_white")
            row_style = ""
        table.add_row(str(i), stage, status, style=row_style)

    console.print(table)


def print_metrics(label: str, metrics: dict) -> None:
    """Print a key→value metrics table."""
    table = Table(
        title=f"[bold]{label}[/bold]",
        box=box.SIMPLE_HEAVY,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Metric", style="dim_white")
    table.add_column("Value", style="metric")
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)


def print_done(message: str = "Pipeline Complete! Ready to submit.") -> None:
    """Print a final success banner."""
    console.print()
    console.print(
        Panel(
            f"[success]{message}[/success]",
            box=box.DOUBLE,
            border_style="bright_green",
            padding=(1, 4),
        )
    )
