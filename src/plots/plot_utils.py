from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt

from src.core.paths import FIGURES_DIR
from src.core.utils import (
    CIStats,
    Q4Stratum,
)


DEFAULT_FIGSIZE = (5, 3)
DEFAULT_LEGEND_POS = (1.05, 1)

class ErrorBars(NamedTuple):
    ci_lowers: list[float]
    ci_uppers: list[float]


def format_name(
    name: str,
) -> str:
    return name.replace('_', ' ').title()


def get_error_bars(
    ci_stats_list: list[CIStats],
) -> ErrorBars:
    ci_lowers = [ci_stats['estimate'] - ci_stats['ci_lower'] for ci_stats in ci_stats_list]
    ci_uppers = [ci_stats['ci_upper'] - ci_stats['estimate'] for ci_stats in ci_stats_list]
    error_bars = ErrorBars(ci_lowers, ci_uppers)
    return error_bars


def save_figure(
    fig: plt.Figure,
    filename: str,
    out_dir: Path = FIGURES_DIR,
    dpi: int = 200,
) -> None:
    fig.savefig(out_dir / f'{filename}.png', bbox_inches='tight', dpi=dpi)
    fig.savefig(out_dir / f'{filename}.pdf', bbox_inches='tight')