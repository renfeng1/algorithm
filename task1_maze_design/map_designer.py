from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from common.models import Cell, GeneratedMaze, MazeConfig, MazeGame, Position
from common.pathing import bfs_distances, shortest_path


@dataclass
class DesignMetrics:
    main_path_length: int
    branch_count: int
    average_branch_depth: float
    gold_on_branches_ratio: float
    trap_near_gold_count: int
    score: float


def _walkable_positions(grid: List[List[str]]) -> List[Position]:
    out: List[Position] = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value != Cell.WALL.value:
                out.append((r, c))
    return out


def _neighbors(grid: List[List[str]], pos: Position) -> List[Position]:
    rows, cols = len(grid), len(grid[0])
    r, c = pos
    out: List[Position] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != Cell.WALL.value:
            out.append((nr, nc))
    return out


def _degree_map(grid: List[List[str]]) -> Dict[Position, int]:
    return {pos: len(_neighbors(grid, pos)) for pos in _walkable_positions(grid)}


def _pick_far_pair(grid: List[List[str]]) -> Tuple[Position, Position]:
    walkable = _walkable_positions(grid)
    start = walkable[0]
    d0 = bfs_distances(MazeGame(grid, [], [], 0, 0), start)
    a = max(d0, key=d0.get)
    da = bfs_distances(MazeGame(grid, [], [], 0, 0), a)
    b = max(da, key=da.get)
    return a, b


def _branch_cells(grid: List[List[str]], main_path: Sequence[Position]) -> List[Position]:
    main_set = set(main_path)
    degrees = _degree_map(grid)
    branches: List[Position] = []
    for pos in main_path:
        if degrees[pos] >= 3:
            branches.append(pos)
    extras = [pos for pos in _walkable_positions(grid) if pos not in main_set]
    extras.sort(key=lambda p: min(abs(p[0] - b[0]) + abs(p[1] - b[1]) for b in branches) if branches else 999)
    return branches + extras


def _distance_to_path(grid: List[List[str]], start: Position, path_set: set[Position]) -> int:
    maze = MazeGame(grid, [], [], 0, 0)
    d = bfs_distances(maze, start)
    return min(d[p] for p in path_set if p in d)


def _select_resource_positions(
    grid: List[List[str]],
    main_path: Sequence[Position],
    num_gold: int,
    num_traps: int,
    rng: random.Random,
) -> Tuple[List[Position], List[Position]]:
    main_set = set(main_path)
    candidates = [p for p in _walkable_positions(grid) if p not in main_set]
    candidates.sort(key=lambda p: (_distance_to_path(grid, p, main_set), -p[0], p[1]), reverse=True)

    gold_positions: List[Position] = []
    trap_positions: List[Position] = []

    for pos in candidates:
        if len(gold_positions) < num_gold and all(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) >= 2 for g in gold_positions):
            gold_positions.append(pos)
            continue
        if len(trap_positions) < num_traps:
            trap_positions.append(pos)
        if len(gold_positions) >= num_gold and len(trap_positions) >= num_traps:
            break

    pool = [p for p in candidates if p not in gold_positions and p not in trap_positions]
    rng.shuffle(pool)
    while len(gold_positions) < num_gold and pool:
        gold_positions.append(pool.pop())
    while len(trap_positions) < num_traps and pool:
        trap_positions.append(pool.pop())

    for pos in list(gold_positions):
        near = [p for p in _neighbors(grid, pos) if p not in main_set and p not in gold_positions and p not in trap_positions]
        if near and len(trap_positions) < num_traps:
            trap_positions.append(near[0])

    return gold_positions[:num_gold], trap_positions[:num_traps]


def _corridor_segments_off_main(
    grid: List[List[str]],
    main_path: Sequence[Position],
) -> List[Tuple[Position, List[Position]]]:
    main_set = set(main_path)
    degree = _degree_map(grid)
    seen: set[Position] = set()
    segments: List[Tuple[Position, List[Position]]] = []
    for anchor in main_path:
        for child in _neighbors(grid, anchor):
            if child in main_set or child in seen:
                continue
            segment: List[Position] = []
            prev = anchor
            current = child
            while current not in main_set and current not in seen:
                segment.append(current)
                seen.add(current)
                choices = [n for n in _neighbors(grid, current) if n != prev and n not in main_set]
                if len(choices) != 1:
                    break
                prev, current = current, choices[0]
            if len(segment) >= 2:
                segments.append((anchor, segment))
    segments.sort(
        key=lambda item: (
            abs(item[1][1][0] - item[0][0]) <= 1 and abs(item[1][1][1] - item[0][1]) <= 1,
            len(item[1]),
            item[1][0][0],
            item[1][0][1],
        ),
        reverse=True,
    )
    return segments


def _apply_visible_trap_gold_patterns(
    grid: List[List[str]],
    main_path: Sequence[Position],
    gold_positions: List[Position],
    trap_positions: List[Position],
    max_patterns: int = 3,
) -> Tuple[List[Position], List[Position]]:
    gold_set = set(gold_positions)
    trap_set = set(trap_positions)
    protected = set(main_path)
    placed = 0

    for anchor, segment in _corridor_segments_off_main(grid, main_path):
        if placed >= max_patterns:
            break
        if any(pos in gold_set or pos in trap_set or pos in protected for pos in segment[:2]):
            continue
        if abs(segment[1][0] - anchor[0]) > 1 or abs(segment[1][1] - anchor[1]) > 1:
            continue
        trap, gold = segment[0], segment[1]
        trap_set.add(trap)
        gold_set.add(gold)
        placed += 1

    pattern_golds = sorted(gold_set - set(gold_positions))
    pattern_traps = sorted(trap_set - set(trap_positions))
    ordered_gold = pattern_golds + [p for p in gold_positions if p not in trap_set and p not in pattern_golds]
    ordered_trap = pattern_traps + [p for p in trap_positions if p not in gold_set and p not in pattern_traps]
    return ordered_gold, ordered_trap


def design_maze(generated: GeneratedMaze, config: MazeConfig) -> Tuple[MazeGame, DesignMetrics]:
    rng = random.Random(config.random_seed + generated.random_seed)
    grid = [row[:] for row in generated.grid]
    start, exit_pos = _pick_far_pair(grid)
    main_path = shortest_path(MazeGame(grid, [], [], 0, 0), start, exit_pos)

    boss_anchor_index = max(1, len(main_path) - 3)
    boss_pos = main_path[boss_anchor_index]
    if boss_pos == exit_pos:
        boss_pos = main_path[-2]

    gold_positions, trap_positions = _select_resource_positions(
        grid,
        main_path=main_path,
        num_gold=config.num_gold,
        num_traps=config.num_traps,
        rng=rng,
    )
    gold_positions, trap_positions = _apply_visible_trap_gold_patterns(
        grid,
        main_path=main_path,
        gold_positions=gold_positions,
        trap_positions=trap_positions,
    )
    gold_positions = gold_positions[: config.num_gold]
    trap_positions = trap_positions[: config.num_traps]

    for pos in gold_positions:
        r, c = pos
        grid[r][c] = Cell.GOLD.value

    for pos in trap_positions:
        r, c = pos
        if grid[r][c] == Cell.ROAD.value:
            grid[r][c] = Cell.TRAP.value

    sr, sc = start
    er, ec = exit_pos
    br, bc = boss_pos
    grid[sr][sc] = Cell.START.value
    grid[er][ec] = Cell.EXIT.value
    grid[br][bc] = Cell.BOSS.value

    branch_positions = [p for p in _walkable_positions(grid) if p not in set(main_path)]
    branch_count = sum(1 for p in main_path if len(_neighbors(grid, p)) >= 3)
    average_branch_depth = (
        sum(_distance_to_path(grid, p, set(main_path)) for p in branch_positions) / len(branch_positions)
        if branch_positions
        else 0.0
    )
    gold_on_branches = sum(1 for p in gold_positions if p in branch_positions)
    trap_near_gold = sum(
        1
        for g in gold_positions
        if any(t in _neighbors(grid, g) for t in trap_positions)
    )

    score = (
        len(main_path) * 1.5
        + branch_count * 8.0
        + average_branch_depth * 6.0
        + gold_on_branches * 10.0
        + trap_near_gold * 4.0
    )
    metrics = DesignMetrics(
        main_path_length=len(main_path),
        branch_count=branch_count,
        average_branch_depth=average_branch_depth,
        gold_on_branches_ratio=gold_on_branches / max(1, len(gold_positions)),
        trap_near_gold_count=trap_near_gold,
        score=score,
    )

    game = MazeGame(
        grid=grid,
        boss_health=config.boss_health[:],
        player_skills=config.player_skills[:],
        min_rounds=0,
        coin_consumption=config.coin_consumption,
        generator_name=generated.generator_name,
        design_score=score,
        notes={"design_metrics": metrics.__dict__},
    )
    return game, metrics


