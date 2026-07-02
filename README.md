# 算法课设：迷宫设计任务核心代码

本仓库只保留核心代码与 README，按任务书中的三个阶段任务拆分，便于老师查看和复现实验。

## 目录划分

```text
.
├── main.py                         # 命令行入口
├── cli.py                          # 参数解析
├── pipeline.py                     # 统一生成、验证、提交包构建流程
├── requirements.txt                # 最小依赖
├── common/                         # 公共模型、规则、路径、校验、JSON IO
├── task1_maze_design/              # 任务一：四种迷宫生成与地图设计
├── task2_resource_path/            # 任务二：最优资源收集路径规划
├── task3_boss_ai_interface/        # 任务三：BOSS 战、AI 玩家与接口适配
└── visualization/                  # 可视化查看器与图片渲染核心代码
```

## 三个任务对应关系

### 任务一：迷宫生成与地图设计

核心文件：

- `task1_maze_design/maze_generators.py`
- `task1_maze_design/map_designer.py`
- `common/validation.py`

实现内容：

- 分治生成迷宫
- 最小生成树/Prim 贪心生成迷宫
- DFS 回溯生成迷宫
- BFS/分支限界风格生成迷宫
- 自动布置 `S` 起点、`E` 终点、`B` BOSS、`G` 金币、`T` 陷阱
- 校验连通性、无孤立区域、唯一通路，即完美迷宫

### 任务二：最优资源收集路径

核心文件：

- `task2_resource_path/resource_planner.py`
- `common/pathing.py`
- `common/rules.py`

实现内容：

- 金币 `G=+50`，陷阱 `T=-30`
- 金币和陷阱均只在首次经过时触发一次
- 使用 `(位置, 已触发资源集合 mask)` 状态搜索求最优资源路径
- 输出最大资源值、路径、资源触发顺序和逐步可视化帧

### 任务三：BOSS 战、AI 玩家与接口适配

核心文件：

- `task3_boss_ai_interface/boss_solver.py`
- `task3_boss_ai_interface/ai_player.py`
- `task3_boss_ai_interface/cross_eval.py`
- `task3_boss_ai_interface/adapters.py`
- `task3_boss_ai_interface/runner.py`

实现内容：

- 分支限界/动态规划搜索 BOSS 最少回合数和最优技能序列
- 至少 5 个 AI 玩家，本项目默认 7 个 AI 玩家画像
- AI 使用 3x3 视野，支持保守型、冒险型、前瞻型、冲出口型等策略
- 支持课程接口文件 Task2、Task3、Task4、Task5 的 JSON 输入输出

## 快速运行

```powershell
python main.py prepare-submission --size 15 --seed 42
python main.py live-viewer --port 8765
```

接口样例运行：

```powershell
python main.py interface --task task2 --input "接口文件\task2_DP\接口文件\maze_15_15.json"
python main.py interface --task task3 --input "接口文件\task3_boss\接口文件\boss_case.json"
python main.py interface --task task4 --input "接口文件\task4_resourcePickup\接口文件\case_0002.json"
python main.py interface --task task5 --input "接口文件\task5_explore_maze\接口文件\maze_7_7_0.json"
```

说明：本仓库按要求只上传核心代码，因此不包含本地 `接口文件/`、`output/`、`docs/`、PDF、图片和缓存文件。若要运行接口样例，请将课程提供的 `接口文件/` 目录放到仓库根目录。

## 最终地图 JSON 格式

最终地图严格匹配 Task5 输入格式：

```json
{
  "maze": [],
  "B": [],
  "PlayerSkills": [],
  "minRouds": 0,
  "CoinConsumption": 0
}
```

其中 `maze` 中使用：

- `#`：墙
- 空格：通路
- `S`：起点
- `E`：终点
- `B`：BOSS
- `G`：金币
- `T`：陷阱
