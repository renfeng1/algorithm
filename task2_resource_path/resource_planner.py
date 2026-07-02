from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Sequence, Set, Tuple

from common.models import Cell, MazeGame, Position, StepFrame
from common.pathing import shortest_path
from common.rules import GOLD_VALUE, TRAP_VALUE


@dataclass
class ResourcePlan:
    max_resource: int
    walk_path: List[Position]
    resource_cells_in_order: List[Position]
    main_path: List[Position]
    branch_gains: Dict[str, int]
    frames: List[StepFrame]


def _cell_first_gain(maze: MazeGame, pos: Position) -> int:
    cell = maze.grid[pos[0]][pos[1]]
    if cell == Cell.GOLD.value:
        return GOLD_VALUE
    if cell == Cell.TRAP.value:
        return TRAP_VALUE
    return 0


def _cell_repeat_gain(maze: MazeGame, pos: Position) -> int:
    return 0


def _children_off_path(maze: MazeGame, pos: Position, path_prev: Position | None, path_next: Position | None) -> List[Position]:
    out: List[Position] = []
    for nxt in maze.neighbors(pos):
        if nxt != path_prev and nxt != path_next:
            out.append(nxt)
    return out


def _branch_key(parent: Position, child: Position) -> str:
    return f"{parent}->{child}"


def plan_optimal_resource_path(maze: MazeGame) -> ResourcePlan:
    start = maze.find_unique(Cell.START.value)
    exit_pos = maze.find_unique(Cell.EXIT.value)
    main_path = shortest_path(maze, start, exit_pos)

    resource_cells = [
        (r, c)
        for r, row in enumerate(maze.grid)
        for c, value in enumerate(row)
        if value in {Cell.GOLD.value, Cell.TRAP.value}
    ]
    resource_index = {pos: i for i, pos in enumerate(resource_cells)}
    resource_values = [
        GOLD_VALUE if maze.grid[r][c] == Cell.GOLD.value else TRAP_VALUE
        for r, c in resource_cells
    ]

    def mask_value(mask: int) -> int:
        total = 0
        current = mask
        while current:
            bit = current & -current
            total += resource_values[bit.bit_length() - 1]
            current -= bit
        return total

    start_mask = 0
    if start in resource_index:
        start_mask |= 1 << resource_index[start]

    state = (start[0], start[1], start_mask)
    queue = deque([state])
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int] | None] = {state: None}
    depth: Dict[Tuple[int, int, int], int] = {state: 0}
    best_exit_state = state if start == exit_pos else None

    while queue:
        r, c, mask = queue.popleft()
        current_state = (r, c, mask)
        if (r, c) == exit_pos:
            if best_exit_state is None:
                best_exit_state = current_state
            else:
                current_key = (mask_value(mask), -depth[current_state])
                best_key = (mask_value(best_exit_state[2]), -depth[best_exit_state])
                if current_key > best_key:
                    best_exit_state = current_state
            continue

        for nxt in maze.neighbors((r, c)):
            next_mask = mask
            if nxt in resource_index:
                next_mask |= 1 << resource_index[nxt]
            next_state = (nxt[0], nxt[1], next_mask)
            if next_state in parent:
                continue
            parent[next_state] = current_state
            depth[next_state] = depth[current_state] + 1
            queue.append(next_state)

    if best_exit_state is None:
        raise ValueError(f"No path from {start} to {exit_pos}")

    walk_path: List[Position] = []
    cursor: Tuple[int, int, int] | None = best_exit_state
    while cursor is not None:
        walk_path.append((cursor[0], cursor[1]))
        cursor = parent[cursor]
    walk_path.reverse()

    max_resource = mask_value(best_exit_state[2])
    branch_gains: Dict[str, int] = {
        "resource_cells": len(resource_cells),
        "state_count": len(parent),
        "objective": max_resource,
    }
    frames: List[StepFrame] = []
    running_resource = 0
    triggered: Set[Position] = set()
    resource_cells_in_order: List[Position] = []

    def append_frame(current_path: List[Position], current: Position, description: str) -> None:
        frames.append(
            StepFrame(
                grid=maze.clone_grid(),
                title=f"Resource step {len(frames) + 1}",
                description=description,
                path=current_path[:],
                highlights=[current],
                meta={"resource": running_resource},
            )
        )

    append_frame([start], start, "从起点开始分析最优资源路径")
    for index, pos in enumerate(walk_path[1:], start=1):
        gain_here = 0
        if pos in resource_index and pos not in triggered:
            triggered.add(pos)
            resource_cells_in_order.append(pos)
            gain_here = _cell_first_gain(maze, pos)
            running_resource += gain_here
        append_frame(
            walk_path[: index + 1],
            pos,
            f"移动到 {pos}，本格首次触发收益 {gain_here}，累计资源 {running_resource}",
        )

    return ResourcePlan(
        max_resource=max_resource,
        walk_path=walk_path,
        resource_cells_in_order=resource_cells_in_order,
        main_path=main_path,
        branch_gains=branch_gains,
        frames=frames,
    )


