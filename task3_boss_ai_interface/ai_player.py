from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from task3_boss_ai_interface.boss_solver import solve_boss_battle
from common.models import Cell, MazeGame, Position
from common.pathing import bfs_distances, shortest_path
from common.rules import GOLD_VALUE, TRAP_VALUE


@dataclass
class AIStep:
    position: Position
    action: str
    coins: int
    visible: List[List[str]]


@dataclass
class AIRunResult:
    success: bool
    steps: List[AIStep]
    final_coins: int
    path: List[Position]
    boss_rounds: int
    boss_skill_sequence: List[int]
    score_ratio: float


@dataclass(frozen=True)
class AIProfile:
    name: str
    trap_penalty: float
    unexplored_bonus: float
    boss_focus: float
    min_safe_coins: int = 0
    lookahead_depth: int = 1
    path_risk_aware: bool = False


def _visible_window(maze: MazeGame, pos: Position) -> List[List[str]]:
    out: List[List[str]] = []
    for dr in (-1, 0, 1):
        row: List[str] = []
        for dc in (-1, 0, 1):
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < maze.rows and 0 <= nc < maze.cols:
                row.append(maze.grid[nr][nc])
            else:
                row.append("#")
        out.append(row)
    return out


def _cell_gain(cell: str, first_time: bool) -> int:
    if cell == Cell.GOLD.value and first_time:
        return GOLD_VALUE
    if cell == Cell.TRAP.value and first_time:
        return TRAP_VALUE
    return 0


def _best_local_target(
    maze: MazeGame,
    pos: Position,
    visited_gold: Set[Position],
    visited: Set[Position],
    boss_pos: Position,
    profile: AIProfile,
) -> Position | None:
    candidates: List[Tuple[float, Position]] = []
    for nr in range(pos[0] - 1, pos[0] + 2):
        for nc in range(pos[1] - 1, pos[1] + 2):
            if not (0 <= nr < maze.rows and 0 <= nc < maze.cols):
                continue
            cell = maze.grid[nr][nc]
            if cell == Cell.WALL.value or (nr, nc) == pos:
                continue
            dist = abs(nr - pos[0]) + abs(nc - pos[1]) + 1
            score = 0.0
            if cell == Cell.GOLD.value and (nr, nc) not in visited_gold:
                score += GOLD_VALUE / dist
            elif cell == Cell.TRAP.value:
                score += (TRAP_VALUE * profile.trap_penalty) / dist
                if profile.lookahead_depth >= 2:
                    deeper_gold = [
                        nxt
                        for nxt in maze.neighbors((nr, nc))
                        if nxt != pos and maze.grid[nxt[0]][nxt[1]] == Cell.GOLD.value and nxt not in visited_gold
                    ]
                    if deeper_gold:
                        score += max(GOLD_VALUE + TRAP_VALUE, 0) / dist
            if profile.path_risk_aware:
                try:
                    local_path = shortest_path(maze, pos, (nr, nc))[1:]
                except ValueError:
                    local_path = []
                path_adjustment = 0.0
                for step in local_path:
                    step_cell = maze.grid[step[0]][step[1]]
                    first_gold = step not in visited_gold
                    first_visit = step not in visited
                    if step_cell == Cell.GOLD.value and first_gold:
                        path_adjustment += GOLD_VALUE
                    elif step_cell == Cell.TRAP.value and first_visit:
                        path_adjustment += TRAP_VALUE * profile.trap_penalty
                if local_path:
                    score = score * 0.35 + path_adjustment / max(1, len(local_path)) * 0.65
            if (nr, nc) not in visited:
                score += profile.unexplored_bonus
            boss_bias = abs(pos[0] - boss_pos[0]) + abs(pos[1] - boss_pos[1]) - abs(nr - boss_pos[0]) - abs(nc - boss_pos[1])
            score += boss_bias * profile.boss_focus
            candidates.append((score, (nr, nc)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    target = candidates[0][1]
    if candidates[0][0] <= 0:
        return None
    return target


def _explore_target(maze: MazeGame, pos: Position, visited: Set[Position]) -> Position | None:
    frontier = [n for n in maze.neighbors(pos) if n not in visited]
    if frontier:
        frontier.sort()
        return frontier[0]
    return None


def _nearest_unvisited(maze: MazeGame, pos: Position, visited: Set[Position]) -> Position | None:
    dist = bfs_distances(maze, pos)
    candidates = [p for p in dist if p not in visited]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (dist[p], p[0], p[1]))
    return candidates[0]


def run_greedy_ai(maze: MazeGame, initial_coins: int = 0) -> AIRunResult:
    return run_profiled_ai(maze, profile=AIProfile("greedy_ratio", trap_penalty=1.0, unexplored_bonus=4.0, boss_focus=0.0), initial_coins=initial_coins)


def run_profiled_ai(maze: MazeGame, profile: AIProfile, initial_coins: int = 0) -> AIRunResult:
    start = maze.find_unique(Cell.START.value)
    exit_pos = maze.find_unique(Cell.EXIT.value)
    boss_pos = maze.find_unique(Cell.BOSS.value)
    current = start
    visited: Set[Position] = {start}
    visited_gold: Set[Position] = set()
    path: List[Position] = [start]
    steps: List[AIStep] = [
        AIStep(position=start, action="start", coins=initial_coins, visible=_visible_window(maze, start))
    ]
    coins = initial_coins
    max_iterations = maze.rows * maze.cols * 10
    iterations = 0

    while current != boss_pos:
        iterations += 1
        if iterations > max_iterations:
            break
        target = _best_local_target(maze, current, visited_gold, visited, boss_pos, profile)
        if coins >= profile.min_safe_coins and target is None and profile.boss_focus > 0:
            target = boss_pos
        if target is None:
            target = _explore_target(maze, current, visited)
        if target is None:
            target = _nearest_unvisited(maze, current, visited)
        if target is None:
            target = boss_pos
        segment = shortest_path(maze, current, target)
        for nxt in segment[1:]:
            first_time = nxt not in path
            gain = _cell_gain(maze.grid[nxt[0]][nxt[1]], first_time)
            if maze.grid[nxt[0]][nxt[1]] == Cell.GOLD.value:
                visited_gold.add(nxt)
            coins += gain
            current = nxt
            path.append(current)
            visited.add(current)
            steps.append(
                AIStep(
                    position=current,
                    action=f"move:{maze.grid[current[0]][current[1]]}",
                    coins=coins,
                    visible=_visible_window(maze, current),
                )
            )
            if coins < 0:
                return AIRunResult(
                    success=False,
                    steps=steps,
                    final_coins=coins,
                    path=path,
                    boss_rounds=0,
                    boss_skill_sequence=[],
                    score_ratio=coins / max(1, len(path) - 1),
                )
            if current == boss_pos:
                break

    boss_plan = solve_boss_battle(maze.boss_health, maze.player_skills, maze.coin_consumption)
    if current != boss_pos:
        to_boss = shortest_path(maze, current, boss_pos)
        for nxt in to_boss[1:]:
            first_time = nxt not in path
            gain = _cell_gain(maze.grid[nxt[0]][nxt[1]], first_time)
            if maze.grid[nxt[0]][nxt[1]] == Cell.GOLD.value:
                visited_gold.add(nxt)
            coins += gain
            current = nxt
            path.append(current)
            steps.append(
                AIStep(
                    position=current,
                    action=f"move:{maze.grid[current[0]][current[1]]}",
                    coins=coins,
                    visible=_visible_window(maze, current),
                )
            )
            if coins < 0:
                return AIRunResult(
                    success=False,
                    steps=steps,
                    final_coins=coins,
                    path=path,
                    boss_rounds=0,
                    boss_skill_sequence=[],
                    score_ratio=coins / max(1, len(path) - 1),
                )

    if coins < maze.coin_consumption and boss_plan.round_limit < boss_plan.min_rounds:
        return AIRunResult(
            success=False,
            steps=steps,
            final_coins=coins,
            path=path,
            boss_rounds=boss_plan.min_rounds,
            boss_skill_sequence=boss_plan.skill_sequence,
            score_ratio=coins / max(1, len(path) - 1),
        )

    to_exit = shortest_path(maze, boss_pos, exit_pos)
    for nxt in to_exit[1:]:
        first_time = nxt not in path
        gain = _cell_gain(maze.grid[nxt[0]][nxt[1]], first_time)
        if maze.grid[nxt[0]][nxt[1]] == Cell.GOLD.value:
            visited_gold.add(nxt)
        coins += gain
        current = nxt
        path.append(current)
        steps.append(
            AIStep(
                position=current,
                action=f"move:{maze.grid[current[0]][current[1]]}",
                coins=coins,
                visible=_visible_window(maze, current),
            )
        )
        if coins < 0:
            return AIRunResult(
                success=False,
                steps=steps,
                final_coins=coins,
                path=path,
                boss_rounds=boss_plan.min_rounds,
                boss_skill_sequence=boss_plan.skill_sequence,
                score_ratio=coins / max(1, len(path) - 1),
            )

    return AIRunResult(
        success=current == exit_pos and coins >= 0,
        steps=steps,
        final_coins=coins,
        path=path,
        boss_rounds=boss_plan.min_rounds,
        boss_skill_sequence=boss_plan.skill_sequence,
        score_ratio=coins / max(1, len(path) - 1),
    )


def default_ai_profiles() -> List[AIProfile]:
    return [
        AIProfile("greedy_ratio", trap_penalty=1.0, unexplored_bonus=4.0, boss_focus=0.0, lookahead_depth=1),
        AIProfile("nearest_gold", trap_penalty=1.2, unexplored_bonus=2.0, boss_focus=0.0, lookahead_depth=1),
        AIProfile("safe_explorer", trap_penalty=2.2, unexplored_bonus=5.0, boss_focus=0.1, lookahead_depth=1, path_risk_aware=True),
        AIProfile("boss_conservative", trap_penalty=1.5, unexplored_bonus=2.5, boss_focus=1.0, min_safe_coins=20, lookahead_depth=1, path_risk_aware=True),
        AIProfile("risk_seeking_collector", trap_penalty=0.45, unexplored_bonus=6.0, boss_focus=0.15, lookahead_depth=2, path_risk_aware=True),
        AIProfile("lookahead_collector", trap_penalty=0.7, unexplored_bonus=4.5, boss_focus=0.2, lookahead_depth=2, path_risk_aware=True),
        AIProfile("exit_rusher", trap_penalty=1.0, unexplored_bonus=0.5, boss_focus=2.0, min_safe_coins=0, lookahead_depth=1),
    ]


