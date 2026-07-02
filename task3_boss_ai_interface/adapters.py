from __future__ import annotations

import heapq
from collections import deque
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from task3_boss_ai_interface.ai_player import AIProfile, run_profiled_ai
from task3_boss_ai_interface.boss_solver import solve_boss_battle
from task2_resource_path.resource_planner import plan_optimal_resource_path
from common.io_utils import load_json
from common.models import MazeGame, Skill
from common.rules import GOLD_VALUE, TRAP_VALUE


def _skills_from_pairs(items: List[List[int]]) -> List[Skill]:
    return [Skill(damage=item[0], cooldown=item[1]) for item in items]


def _maze_from_task2_payload(payload: Dict[str, Any]) -> MazeGame:
    return MazeGame(
        grid=payload["maze"],
        boss_health=[],
        player_skills=[],
        min_rounds=0,
        coin_consumption=0,
    )


def _maze_from_task5_payload(payload: Dict[str, Any]) -> MazeGame:
    return MazeGame(
        grid=payload["maze"],
        boss_health=payload["B"],
        player_skills=_skills_from_pairs(payload["PlayerSkills"]),
        min_rounds=payload["minRouds"],
        coin_consumption=payload["CoinConsumption"],
    )


def task2_dp_result(input_path: str) -> Dict[str, Any]:
    payload = load_json(input_path)
    plan = _general_resource_plan(payload["maze"])
    return {
        "path_length": max(0, len(plan["path"]) - 1),
        "path": [[r, c] for r, c in plan["path"]],
        "max_resource": plan["max_resource"],
    }


def task3_boss_result(input_path: str) -> Dict[str, Any]:
    payload = load_json(input_path)
    skills = _skills_from_pairs(payload["PlayerSkills"])
    plan = solve_boss_battle(payload["B"], skills, coin_consumption=payload.get("CoinConsumption", 0))
    return {
        "B": payload["B"],
        "PlayerSkills": payload["PlayerSkills"],
        "min_turns": plan.min_rounds,
        "SkillSequence": plan.skill_sequence,
    }


def task4_resource_pickup_result(input_path: str) -> Dict[str, Any]:
    payload = load_json(input_path)
    grid = payload["grid"]
    start = _find_cell(grid, "P")
    best = _best_3x3_resource_path(grid, start)
    path = best["path"]
    move_steps = max(0, len(path) - 1)
    resource = best["resource"]
    return {
        "case_id": payload.get("case_id"),
        "path": [[r, c] for r, c in path],
        "path_length": len(path),
        "move_steps": move_steps,
        "final_resource": resource,
        "resource_step_ratio": resource / max(1, move_steps),
        "gold_count": best["gold_count"],
        "trap_count": best["trap_count"],
    }


def task5_explore_result(input_path: str, profile: AIProfile | None = None) -> Dict[str, Any]:
    payload = load_json(input_path)
    maze = _maze_from_task5_payload(payload)
    profile = profile or AIProfile("interface_balanced", trap_penalty=1.5, unexplored_bonus=4.0, boss_focus=0.6, min_safe_coins=10)
    result = run_profiled_ai(maze, profile)
    move_steps = max(0, len(result.path) - 1)
    boss_report = _simulate_task5_boss_attempts(
        payload["B"],
        _skills_from_pairs(payload["PlayerSkills"]),
        payload["minRouds"],
        payload["CoinConsumption"],
        result.final_coins,
    )
    final_coin = result.final_coins - boss_report["boss_coin_cost"]
    success = result.success and boss_report["boss_success"] and final_coin >= 0
    return {
        "success": success,
        "path": [[r, c] for r, c in result.path],
        "path_length": len(result.path),
        "move_steps": move_steps,
        "final_coin": final_coin if success else result.final_coins,
        "coin_step_ratio": final_coin / max(1, move_steps) if success else 0,
        **boss_report,
    }


def _simulate_task5_boss_attempts(
    boss_health: List[int],
    skills: List[Skill],
    min_rounds: int,
    coin_consumption: int,
    available_coins: int,
) -> Dict[str, Any]:
    optimal = solve_boss_battle(boss_health, skills, coin_consumption=coin_consumption)

    # Each boss attempt is capped by min_rounds. A smaller min_rounds therefore
    # makes the battle harder: the first myopic attempt is more likely to fail,
    # and the revival only helps if the optimal known-HP plan can fit the cap.
    greedy_attempt = _limited_greedy_boss_sequence(boss_health, skills, min_rounds)
    if _boss_sequence_defeats(boss_health, skills, greedy_attempt):
        return {
            "boss_success": True,
            "boss_total_turns": len(greedy_attempt),
            "boss_revive_count": 0,
            "boss_coin_cost": 0,
            "boss_skill_sequence_lengths": [len(greedy_attempt)],
            "boss_skill_sequences": [greedy_attempt],
        }

    if available_coins >= coin_consumption > 0 and optimal.min_rounds <= min_rounds:
        return {
            "boss_success": True,
            "boss_total_turns": len(greedy_attempt) + optimal.min_rounds,
            "boss_revive_count": 1,
            "boss_coin_cost": coin_consumption,
            "boss_skill_sequence_lengths": [len(greedy_attempt), len(optimal.skill_sequence)],
            "boss_skill_sequences": [greedy_attempt, optimal.skill_sequence],
        }

    if available_coins >= coin_consumption > 0:
        capped_optimal = optimal.skill_sequence[:min_rounds]
        return {
            "boss_success": False,
            "boss_total_turns": len(greedy_attempt) + len(capped_optimal),
            "boss_revive_count": 1,
            "boss_coin_cost": coin_consumption,
            "boss_skill_sequence_lengths": [len(greedy_attempt), len(capped_optimal)],
            "boss_skill_sequences": [greedy_attempt, capped_optimal],
        }

    return {
        "boss_success": False,
        "boss_total_turns": len(greedy_attempt),
        "boss_revive_count": 0,
        "boss_coin_cost": 0,
        "boss_skill_sequence_lengths": [len(greedy_attempt)],
        "boss_skill_sequences": [greedy_attempt],
    }


def _boss_sequence_defeats(boss_health: List[int], skills: List[Skill], sequence: List[int]) -> bool:
    hp = list(boss_health)
    boss_index = 0
    for skill_index in sequence:
        if boss_index >= len(hp):
            return True
        hp[boss_index] -= skills[skill_index].damage
        while boss_index < len(hp) and hp[boss_index] <= 0:
            boss_index += 1
    return boss_index >= len(hp)


def _limited_greedy_boss_sequence(boss_health: List[int], skills: List[Skill], rounds: int) -> List[int]:
    hp = list(boss_health)
    boss_index = 0
    cooldowns = [0 for _ in skills]
    sequence: List[int] = []
    for _ in range(rounds):
        if boss_index >= len(hp):
            break
        available = [i for i, cd in enumerate(cooldowns) if cd == 0]
        if not available:
            cooldowns = [max(0, cd - 1) for cd in cooldowns]
            continue
        # First attempt is intentionally myopic: highest immediate damage, then lower index.
        skill_index = min(available, key=lambda i: (-skills[i].damage, i))
        sequence.append(skill_index)
        if boss_index < len(hp):
            hp[boss_index] -= skills[skill_index].damage
            while boss_index < len(hp) and hp[boss_index] <= 0:
                boss_index += 1
        cooldowns = [max(0, cd - 1) for cd in cooldowns]
        cooldowns[skill_index] = skills[skill_index].cooldown
    return sequence


def _find_cell(grid: List[List[str]], target: str) -> Tuple[int, int]:
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == target:
                return (r, c)
    raise ValueError(f"Cell {target!r} not found")


def _best_3x3_resource_path(grid: List[List[str]], start: Tuple[int, int]) -> Dict[str, Any]:
    rows, cols = len(grid), len(grid[0])
    resource_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] in {"G", "T"}
    ]
    resource_index = {pos: i for i, pos in enumerate(resource_cells)}

    def gain(cell: str) -> int:
        if cell == "G":
            return GOLD_VALUE
        if cell == "T":
            return TRAP_VALUE
        return 0

    start_mask = 0
    start_value = 0
    if start in resource_index:
        start_mask = 1 << resource_index[start]
        start_value = gain(grid[start[0]][start[1]])

    best_score, best_state, parent = _maximize_walk_with_masks(
        grid,
        starts=[(start, start_mask, start_value)],
        resource_index=resource_index,
        resource_once=True,
    )
    path = _reconstruct_state_path(best_state, parent)
    triggered = {
        resource_cells[i]
        for i in range(len(resource_cells))
        if best_state[2] & (1 << i)
    }
    return {
        "resource": best_score,
        "path": path,
        "gold_count": sum(1 for p in triggered if grid[p[0]][p[1]] == "G"),
        "trap_count": sum(1 for p in triggered if grid[p[0]][p[1]] == "T"),
    }


def _general_resource_plan(grid: List[List[str]]) -> Dict[str, Any]:
    rows, cols = len(grid), len(grid[0])
    resource_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] in {"G", "T"}
    ]
    resource_index = {pos: i for i, pos in enumerate(resource_cells)}

    def value_of(cell: str) -> int:
        if cell == "G":
            return GOLD_VALUE
        if cell == "T":
            return TRAP_VALUE
        return 0

    starts = []
    for start in [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != "#"]:
        mask = 0
        start_value = 0
        if start in resource_index:
            mask |= 1 << resource_index[start]
            start_value = value_of(grid[start[0]][start[1]])
        starts.append((start, mask, start_value))

    best_value, best_state, parent = _maximize_walk_with_masks(
        grid,
        starts=starts,
        resource_index=resource_index,
        resource_once=True,
    )
    shortest_path = _shortest_path_for_resource_value(grid, starts, resource_index, resource_cells, best_value)
    if shortest_path is None:
        shortest_path = _reconstruct_state_path(best_state, parent)
    return {"path": shortest_path, "max_resource": best_value}


def _shortest_path_for_resource_value(
    grid: List[List[str]],
    starts: List[Tuple[Tuple[int, int], int, int]],
    resource_index: Dict[Tuple[int, int], int],
    resource_cells: List[Tuple[int, int]],
    target_value: int,
) -> List[Tuple[int, int]] | None:
    rows, cols = len(grid), len(grid[0])
    resource_values = [GOLD_VALUE if grid[r][c] == "G" else TRAP_VALUE for r, c in resource_cells]

    @lru_cache(maxsize=None)
    def mask_value(mask: int) -> int:
        total = 0
        current = mask
        while current:
            bit = current & -current
            total += resource_values[bit.bit_length() - 1]
            current -= bit
        return total

    queue = deque()
    parent: Dict[State, State | None] = {}
    seen: set[State] = set()
    for pos, mask, _ in starts:
        state = (pos[0], pos[1], mask)
        if state in seen:
            continue
        seen.add(state)
        parent[state] = None
        if mask_value(mask) == target_value:
            return _reconstruct_state_path(state, parent)
        queue.append(state)

    while queue:
        r, c, mask = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] == "#":
                continue
            next_mask = mask
            nxt = (nr, nc)
            if nxt in resource_index:
                next_mask |= 1 << resource_index[nxt]
            next_state = (nr, nc, next_mask)
            if next_state in seen:
                continue
            seen.add(next_state)
            parent[next_state] = (r, c, mask)
            if mask_value(next_mask) == target_value:
                return _reconstruct_state_path(next_state, parent)
            queue.append(next_state)
    return None


State = Tuple[int, int, int]


def _maximize_walk_with_masks(
    grid: List[List[str]],
    starts: List[Tuple[Tuple[int, int], int, int]],
    resource_index: Dict[Tuple[int, int], int],
    resource_once: bool,
) -> Tuple[int, State, Dict[State, State | None]]:
    rows, cols = len(grid), len(grid[0])
    best_score_by_state: Dict[State, int] = {}
    parent: Dict[State, State | None] = {}
    heap: List[Tuple[int, int, State]] = []
    counter = 0

    for pos, mask, score in starts:
        state = (pos[0], pos[1], mask)
        if score > best_score_by_state.get(state, -10**9):
            best_score_by_state[state] = score
            parent[state] = None
            heapq.heappush(heap, (-score, counter, state))
            counter += 1

    best_state = next(iter(best_score_by_state))
    best_score = best_score_by_state[best_state]

    while heap:
        neg_score, _, state = heapq.heappop(heap)
        score = -neg_score
        if score != best_score_by_state.get(state):
            continue
        if score > best_score:
            best_score = score
            best_state = state
        r, c, mask = state
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] == "#":
                continue
            next_mask = mask
            delta = 0
            nxt = (nr, nc)
            if nxt in resource_index:
                bit = 1 << resource_index[nxt]
                if not (mask & bit):
                    next_mask |= bit
                    delta = GOLD_VALUE if grid[nr][nc] == "G" else TRAP_VALUE
            elif grid[nr][nc] == "T" and not resource_once:
                delta = TRAP_VALUE
            next_state = (nr, nc, next_mask)
            next_score = score + delta
            if next_score > best_score_by_state.get(next_state, -10**9):
                best_score_by_state[next_state] = next_score
                parent[next_state] = state
                heapq.heappush(heap, (-next_score, counter, next_state))
                counter += 1

    return best_score, best_state, parent


def _reconstruct_state_path(state: State, parent: Dict[State, State | None]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    cur: State | None = state
    while cur is not None:
        out.append((cur[0], cur[1]))
        cur = parent[cur]
    out.reverse()
    return out


