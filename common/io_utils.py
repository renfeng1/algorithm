from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from common.models import MazeGame, Skill, skills_to_pairs


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maze_to_dict(maze: MazeGame, include_meta: bool = True) -> Dict[str, Any]:
    payload = {
        "maze": maze.grid,
        "B": maze.boss_health,
        "PlayerSkills": skills_to_pairs(maze.player_skills),
        "minRouds": maze.min_rounds,
        "CoinConsumption": maze.coin_consumption,
    }
    if include_meta and maze.generator_name:
        payload["_generator"] = maze.generator_name
    if include_meta and maze.notes:
        payload["_notes"] = maze.notes
    if include_meta and maze.design_score:
        payload["_design_score"] = maze.design_score
    return payload


def save_maze_json(path: str | Path, maze: MazeGame, include_meta: bool = True) -> None:
    save_json(path, maze_to_dict(maze, include_meta=include_meta))


def load_maze_json(path: str | Path) -> MazeGame:
    data = load_json(path)
    skills = [Skill(damage=item[0], cooldown=item[1]) for item in data["PlayerSkills"]]
    return MazeGame(
        grid=data["maze"],
        boss_health=data["B"],
        player_skills=skills,
        min_rounds=data["minRouds"],
        coin_consumption=data["CoinConsumption"],
        generator_name=data.get("_generator", ""),
        design_score=float(data.get("_design_score", 0.0)),
        notes=data.get("_notes", {}),
    )

