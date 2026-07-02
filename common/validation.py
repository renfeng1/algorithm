from __future__ import annotations

from collections import deque
from typing import Iterable, List, Set, Tuple

from common.models import Cell, MazeGame, Position


def walkable_positions(maze: MazeGame) -> List[Position]:
    out: List[Position] = []
    for r, row in enumerate(maze.grid):
        for c, value in enumerate(row):
            if value != Cell.WALL.value:
                out.append((r, c))
    return out


def connected_components(maze: MazeGame) -> List[Set[Position]]:
    walkable = set(walkable_positions(maze))
    components: List[Set[Position]] = []
    while walkable:
        start = next(iter(walkable))
        queue = deque([start])
        comp: Set[Position] = {start}
        walkable.remove(start)
        while queue:
            node = queue.popleft()
            for nxt in maze.neighbors(node):
                if nxt in walkable:
                    walkable.remove(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components


def count_edges(maze: MazeGame) -> int:
    edges = 0
    for node in walkable_positions(maze):
        for nxt in maze.neighbors(node):
            if nxt > node:
                edges += 1
    return edges


def is_connected(maze: MazeGame) -> bool:
    comps = connected_components(maze)
    return len(comps) == 1


def is_perfect_maze(maze: MazeGame) -> bool:
    nodes = walkable_positions(maze)
    if not nodes:
        return False
    return is_connected(maze) and count_edges(maze) == len(nodes) - 1


def validate_course_format(maze: MazeGame) -> List[str]:
    errors: List[str] = []
    for cell in (Cell.START.value, Cell.EXIT.value, Cell.BOSS.value):
        count = len(maze.positions_of(cell))
        if count != 1:
            errors.append(f"{cell} count must be 1, got {count}")
    if not maze.player_skills:
        errors.append("At least one skill is required")
    if not any(skill.cooldown == 0 for skill in maze.player_skills):
        errors.append("At least one zero-cooldown skill is required")
    if not is_connected(maze):
        errors.append("Maze is not connected")
    if not is_perfect_maze(maze):
        errors.append("Maze is not a perfect maze")
    return errors

