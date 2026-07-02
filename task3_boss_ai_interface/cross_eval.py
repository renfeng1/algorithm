from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Sequence

from task3_boss_ai_interface.ai_player import AIRunResult, AIProfile, default_ai_profiles, run_profiled_ai
from common.models import MazeGame


@dataclass
class CrossEvalResult:
    maze_score: float
    distinguishability: float
    stability: float
    balance: float
    ai_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    ai_results: Dict[str, dict]
    ai_count: int


def _percentile_rank(value: float, values: Sequence[float]) -> float:
    ordered = sorted(values)
    rank = sum(1 for item in ordered if item <= value)
    return rank / max(1, len(ordered))


def evaluate_maze_with_profiles(maze: MazeGame, profiles: List[AIProfile] | None = None) -> CrossEvalResult:
    profiles = profiles or default_ai_profiles()
    ai_scores: Dict[str, float] = {}
    ai_results: Dict[str, dict] = {}
    raw_scores: List[float] = []

    for profile in profiles:
        result = run_profiled_ai(maze, profile)
        score = result.final_coins / max(1, len(result.path) - 1)
        ai_scores[profile.name] = score
        raw_scores.append(score)
        ai_results[profile.name] = {
            "profile": {
                "trap_penalty": profile.trap_penalty,
                "unexplored_bonus": profile.unexplored_bonus,
                "boss_focus": profile.boss_focus,
                "min_safe_coins": profile.min_safe_coins,
            },
            "success": result.success,
            "final_coins": result.final_coins,
            "steps": len(result.path) - 1,
            "score_ratio": result.score_ratio,
            "path": [[r, c] for r, c in result.path],
        }

    if len(raw_scores) == 1:
        normalized = [1.0]
    else:
        lo, hi = min(raw_scores), max(raw_scores)
        gap = max(1e-9, hi - lo)
        normalized = [(x - lo) / gap for x in raw_scores]

    normalized_scores = {profile.name: normalized[i] for i, profile in enumerate(profiles)}
    raw_std = pstdev(raw_scores) if len(raw_scores) > 1 else 0.0
    normalized_std = pstdev(normalized) if len(normalized) > 1 else 0.0
    distinguishability = normalized_std
    avg = mean(raw_scores)
    q = _percentile_rank(avg, raw_scores)
    balance = 4 * q * (1 - q)
    # Full stability needs the class-wide AI x maze matrix. For a single submitted
    # maze we use a neutral local proxy and expose it explicitly in the report.
    stability = 1.0
    maze_score = 100 * ((max(distinguishability, 1e-9) * max(stability, 1e-9) * max(balance, 1e-9)) ** (1 / 3))

    return CrossEvalResult(
        maze_score=maze_score,
        distinguishability=distinguishability,
        stability=stability,
        balance=balance,
        ai_scores=ai_scores,
        normalized_scores=normalized_scores,
        ai_results=ai_results,
        ai_count=len(profiles),
    )


