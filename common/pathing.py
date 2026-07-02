from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from common.models import MazeGame, Position


def bfs_distances(maze: MazeGame, start: Position) -> Dict[Position, int]:
    queue = deque([start])
    dist: Dict[Position, int] = {start: 0}
    while queue:
        node = queue.popleft()
        for nxt in maze.neighbors(node):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def shortest_path(maze: MazeGame, start: Position, goal: Position) -> List[Position]:
    queue = deque([start])
    parent: Dict[Position, Optional[Position]] = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for nxt in maze.neighbors(node):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    if goal not in parent:
        raise ValueError(f"No path from {start} to {goal}")
    path: List[Position] = []
    cur: Optional[Position] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

