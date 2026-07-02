from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common.io_utils import ensure_parent
from common.models import Cell, MazeGame, Position


CELL_TO_VALUE = {
    Cell.WALL.value: 0,
    Cell.ROAD.value: 1,
    Cell.START.value: 2,
    Cell.EXIT.value: 3,
    Cell.BOSS.value: 4,
    Cell.GOLD.value: 5,
    Cell.TRAP.value: 6,
}

COLORS = [
    "#1f2937",
    "#f8fafc",
    "#10b981",
    "#ef4444",
    "#7c3aed",
    "#f59e0b",
    "#0ea5e9",
]


def save_ascii_grid(grid: List[List[str]], path: str | Path) -> None:
    path = Path(path)
    ensure_parent(path)
    text = "\n".join("".join(row) for row in grid)
    path.write_text(text, encoding="utf-8")


def _grid_to_numeric(grid: List[List[str]]) -> List[List[int]]:
    return [[CELL_TO_VALUE[cell] for cell in row] for row in grid]


def save_grid_image(grid: List[List[str]], path: str | Path, title: str = "") -> None:
    path = Path(path)
    ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_grid_to_numeric(grid), cmap=ListedColormap(COLORS), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value != Cell.ROAD.value and value != Cell.WALL.value:
                ax.text(c, r, value, ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_snapshot_sequence(snapshots: Sequence[List[List[str]]], out_dir: str | Path, prefix: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, grid in enumerate(snapshots, start=1):
        save_grid_image(grid, out_dir / f"{prefix}_{i:03d}.png", title=f"{prefix} step {i}")


def save_path_overlay(
    maze: MazeGame,
    path_positions: Sequence[Position],
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_grid_to_numeric(maze.grid), cmap=ListedColormap(COLORS), interpolation="nearest")
    xs = [p[1] for p in path_positions]
    ys = [p[0] for p in path_positions]
    ax.plot(xs, ys, color="#111827", linewidth=2.0, marker="o", markersize=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


