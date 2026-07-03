from typing import Dict, List
from common.models import Cell, MazeGame, Position
from common.rules import GOLD_VALUE, TRAP_VALUE

class MazeResourceRegistry:
    """
    业务意图：统一管理和查询迷宫内金币和陷阱等资源的注册表。
    通过查询函数替代原有的临时变量，以更好地体现各个数据结构的业务含义。
    """
    def __init__(self, maze: MazeGame):
        self.maze = maze
        self.all_resource_positions = [
            (r, c)
            for r, row in enumerate(maze.grid)
            for c, value in enumerate(row)
            if value in {Cell.GOLD.value, Cell.TRAP.value}
        ]
        self._position_to_bitmap_index = {pos: i for i, pos in enumerate(self.all_resource_positions)}
        self._bitmap_index_to_value = [
            GOLD_VALUE if maze.grid[r][c] == Cell.GOLD.value else TRAP_VALUE
            for r, c in self.all_resource_positions
        ]

    def bitmap_index_for(self, pos: Position) -> int | None:
        """根据网格坐标，查询其在收集状态位图（collected_bitmap）中对应的二进制位索引"""
        return self._position_to_bitmap_index.get(pos)

    def value_for_index(self, index: int) -> int:
        """根据位图中的二进制位索引，查询该资源的净收益值（金币为正，陷阱为负）"""
        return self._bitmap_index_to_value[index]

    def position_for_index(self, index: int) -> Position:
        """根据二进制位索引，获取该资源的物理网格坐标"""
        return self.all_resource_positions[index]

    def total_count(self) -> int:
        """获取地图上所有可触发资源（金币和陷阱）的总数"""
        return len(self.all_resource_positions)

    def has_resource_at(self, pos: Position) -> bool:
        """判断某个物理网格坐标是否是注册的资源格"""
        return pos in self._position_to_bitmap_index
