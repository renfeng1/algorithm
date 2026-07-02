from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from common.models import Skill


@dataclass
class BossPlan:
    min_rounds: int
    skill_sequence: List[int]
    damage_sequence: List[int]
    round_limit: int
    coin_consumption: int


def _apply_skill(
    hp: Tuple[int, ...],
    boss_index: int,
    cooldowns: Tuple[int, ...],
    skills: List[Skill],
    skill_index: int,
) -> Tuple[Tuple[int, ...], int, Tuple[int, ...]]:
    hp_list = list(hp)
    hp_list[boss_index] -= skills[skill_index].damage
    next_boss = boss_index
    while next_boss < len(hp_list) and hp_list[next_boss] <= 0:
        next_boss += 1

    new_cooldowns = [max(0, cd - 1) for cd in cooldowns]
    new_cooldowns[skill_index] = skills[skill_index].cooldown
    return tuple(hp_list), next_boss, tuple(new_cooldowns)


def solve_boss_battle(boss_health: List[int], skills: List[Skill], coin_consumption: int) -> BossPlan:
    start_hp = tuple(boss_health)
    max_damage = max(skill.damage for skill in skills)

    def lower_bound(hp: Tuple[int, ...], boss_index: int) -> int:
        remaining = sum(max(0, hp[i]) for i in range(boss_index, len(hp)))
        return math.ceil(remaining / max_damage)

    def sequence_rank(sequence: List[int]) -> Tuple[Tuple[int, int, int], ...]:
        # Stable tie-break: among equal-length optimal plans, prefer stronger skills earlier.
        return tuple((-skills[i].damage, -skills[i].cooldown, -i) for i in sequence)

    heap: List[
        Tuple[
            int,
            int,
            Tuple[Tuple[int, int, int], ...],
            Tuple[int, ...],
            int,
            Tuple[int, ...],
            List[int],
        ]
    ] = []
    heapq.heappush(heap, (lower_bound(start_hp, 0), 0, (), start_hp, 0, tuple(0 for _ in skills), []))
    best_seen: Dict[Tuple[Tuple[int, ...], int, Tuple[int, ...]], int] = {}

    while heap:
        _, rounds, _, hp, boss_index, cooldowns, sequence = heapq.heappop(heap)
        state_key = (hp, boss_index, cooldowns)
        if best_seen.get(state_key, 10**9) < rounds:
            continue
        if boss_index >= len(boss_health):
            return BossPlan(
                min_rounds=rounds,
                skill_sequence=sequence,
                damage_sequence=[skills[i].damage for i in sequence],
                round_limit=max(rounds + 2, math.ceil(rounds * 1.15)),
                coin_consumption=coin_consumption,
            )

        for i, skill in enumerate(skills):
            if cooldowns[i] != 0:
                continue
            next_hp, next_boss, next_cd = _apply_skill(hp, boss_index, cooldowns, skills, i)
            next_rounds = rounds + 1
            next_key = (next_hp, next_boss, next_cd)
            if next_rounds >= best_seen.get(next_key, 10**9):
                continue
            best_seen[next_key] = next_rounds
            bound = next_rounds + lower_bound(next_hp, next_boss)
            next_sequence = sequence + [i]
            heapq.heappush(
                heap,
                (bound, next_rounds, sequence_rank(next_sequence), next_hp, next_boss, next_cd, next_sequence),
            )

    raise RuntimeError("Boss battle search failed to find a solution")


