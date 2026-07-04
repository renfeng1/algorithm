import json
from collections import deque

# 从JSON文件中读取迷宫
def read_maze(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['maze']

# 广度优先搜索函数
def bfs(maze, start, end=None):
    rows, cols = len(maze), len(maze[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = [[False] * cols for _ in range(rows)]
    visited[start[0]][start[1]] = True
    queue = deque([(start, [start])])
    paths = []

    while queue:
        (x, y), path = queue.popleft()
        if end and (x, y) == end:
            paths.append(path)
            continue
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < rows and 0 <= new_y < cols and maze[new_x][new_y] != '#' and not visited[new_x][new_y]:
                visited[new_x][new_y] = True
                new_path = path + [(new_x, new_y)]
                queue.append(((new_x, new_y), new_path))

    return paths, visited

# 判断两点间是否有唯一通路
def has_unique_path(maze, start, end):
    paths, _ = bfs(maze, start, end)
    if not paths:  # 如果没有路径，则不存在唯一通路
        return False
    return len(paths) == 1

# 找到起点和终点的坐标以及所有可通行的格子
def find_start_end_and_passages(maze):
    start = None
    end = None
    passages = []
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            cell = maze[i][j]
            if cell == 'S':
                start = (i, j)
                passages.append((i, j))
            elif cell == 'E':
                end = (i, j)
                passages.append((i, j))
            elif cell != '#':  # 非墙壁即为通道
                passages.append((i, j))
                
    return start, end, passages

# 检测迷宫中的孤立区域
def find_isolated_areas(maze, start):
    rows, cols = len(maze), len(maze[0])
    
    # 从起点开始BFS，标记所有可达的格子
    _, visited_from_start = bfs(maze, start)
    
    # 查找所有非墙壁但不可达的格子
    isolated_areas = []
    
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] != '#' and not visited_from_start[i][j]:
                isolated_areas.append((i, j))
    
    return isolated_areas

# 检查终点是否可达
def is_end_reachable(maze, start, end):
    paths, _ = bfs(maze, start, end)
    return len(paths) > 0

# 主函数
def main():
    file_path = 'maze_7_7.json'  # 迷宫文件路径
    maze = read_maze(file_path)
    start, end, passages = find_start_end_and_passages(maze)
    
    if start and end:
        # 检查终点是否可达
        end_reachable = is_end_reachable(maze, start, end)
        if not end_reachable:
            print("终点不可达！")
        else:
            # 检查是否有唯一通路
            unique_path = has_unique_path(maze, start, end)
            print(f"迷宫中从起点到终点是否有唯一通路: {unique_path}")
        
        # 检查是否有孤立区域
        isolated_areas = find_isolated_areas(maze, start)
        
        if isolated_areas:
            # 检查终点是否在孤立区域中
            end_isolated = end in isolated_areas
            if end_isolated:
                print(f"终点 {end} 不可从起点到达！")
                # 移除终点，只显示其他孤立区域
                isolated_areas.remove(end)
            
            if isolated_areas:  # 如果还有其他孤立区域
                print(f"迷宫中存在孤立区域，共有 {len(isolated_areas)} 个格子不可达:")
                for i, area in enumerate(isolated_areas[:5]):  # 只显示前5个孤立格子
                    print(f"  - 孤立格子 {i+1}: 坐标 {area}, 内容: {maze[area[0]][area[1]]}")
                if len(isolated_areas) > 5:
                    print(f"  ... 以及其他 {len(isolated_areas) - 5} 个孤立格子")
            else:
                print("除了终点外，迷宫中不存在其他孤立区域。")
        else:
            print("迷宫中不存在孤立区域，所有非墙壁格子都可以从起点到达。")
    else:
        print("未找到起点或终点。")

if __name__ == "__main__":
    main()