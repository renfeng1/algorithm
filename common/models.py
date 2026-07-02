from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, List, Sequence, Tuple


Position = Tuple[int, int]


class Cell(str, Enum):
    WALL = "#"
    ROAD = " "
    START = "S"
    EXIT = "E"
    BOSS = "B"
    GOLD = "G"
    TRAP = "T"


@dataclass(frozen=True)
class Skill:
    damage: int
    cooldown: int


@dataclass
class MazeConfig:
    size: int = 15
    num_gold: int = 8
    num_traps: int = 6
    boss_health: List[int] = field(default_factory=lambda: [16, 21, 18, 24])
    player_skills: List[Skill] = field(
        default_factory=lambda: [
            Skill(2, 0),
            Skill(4, 1),
            Skill(6, 3),
            Skill(9, 5),
        ]
    )
    coin_consumption: int = 5
    random_seed: int = 42


@dataclass
class MazeGame:
    grid: List[List[str]]
    boss_health: List[int]
    player_skills: List[Skill]
    min_rounds: int
    coin_consumption: int
    generator_name: str = ""
    design_score: float = 0.0
    notes: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def clone_grid(self) -> List[List[str]]:
        return [row[:] for row in self.grid]

    def positions_of(self, cell: str) -> List[Position]:
        out: List[Position] = []
        for r, row in enumerate(self.grid):
            for c, value in enumerate(row):
                if value == cell:
                    out.append((r, c))
        return out

    def find_unique(self, cell: str) -> Position:
        positions = self.positions_of(cell)
        if len(positions) != 1:
            raise ValueError(f"Expected exactly one {cell}, found {len(positions)}")
        return positions[0]

    def walkable(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != Cell.WALL.value

    def neighbors(self, pos: Position) -> Iterable[Position]:
        r, c = pos
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if self.walkable(nxt):
                yield nxt


def skills_to_pairs(skills: Sequence[Skill]) -> List[List[int]]:
    return [[skill.damage, skill.cooldown] for skill in skills]


@dataclass
class GeneratedMaze:
    grid: List[List[str]]
    room_rows: int
    room_cols: int
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    snapshots: List[List[List[str]]]
    generator_name: str
    random_seed: int
    events: List[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepFrame:
    grid: List[List[str]]
    title: str
    description: str
    path: List[Position] = field(default_factory=list)
    highlights: List[Position] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

