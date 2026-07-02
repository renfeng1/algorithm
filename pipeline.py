from __future__ import annotations

import json
import shutil
from pathlib import Path

from task3_boss_ai_interface.ai_player import run_greedy_ai
from task3_boss_ai_interface.boss_solver import solve_boss_battle
from task3_boss_ai_interface.cross_eval import evaluate_maze_with_profiles
from task1_maze_design.map_designer import design_maze
from task1_maze_design.maze_generators import generate_all
from task2_resource_path.resource_planner import plan_optimal_resource_path
from common.io_utils import load_maze_json, maze_to_dict, save_json, save_maze_json
from common.models import MazeConfig
from common.validation import validate_course_format
from task3_boss_ai_interface.runner import run_interface_task
from visualization.interactive_bundle import create_visualization_bundle
from visualization.renderers import (
    save_ascii_grid,
    save_grid_image,
    save_path_overlay,
    save_snapshot_sequence,
)


def _project_dirs() -> dict[str, Path]:
    base = Path.cwd()
    return {
        "output": base / "output",
        "final": base / "output" / "final",
        "demo": base / "output" / "demo",
        "docs": base / "docs",
    }


def _serialize_path(path):
    return [[r, c] for r, c in path]


def _serialize_cross_eval(cross_eval):
    return {
        "ai_count": cross_eval.ai_count,
        "maze_score": cross_eval.maze_score,
        "distinguishability": cross_eval.distinguishability,
        "stability": cross_eval.stability,
        "balance": cross_eval.balance,
        "ai_scores": cross_eval.ai_scores,
        "normalized_scores": cross_eval.normalized_scores,
        "ai_results": cross_eval.ai_results,
        "formula_note": (
            "raw_score = final_coin / move_steps; normalized_score is min-max normalized "
            "inside this maze. Single-maze stability is reported as a neutral proxy; "
            "the full Spearman stability needs the class-wide AI x maze matrix."
        ),
    }


def _build_single_candidate(generated, config: MazeConfig):
    maze, metrics = design_maze(generated, config)
    boss = solve_boss_battle(maze.boss_health, maze.player_skills, maze.coin_consumption)
    maze.min_rounds = boss.round_limit
    maze.notes["boss_plan"] = {
        "min_rounds": boss.min_rounds,
        "skill_sequence": boss.skill_sequence,
        "damage_sequence": boss.damage_sequence,
        "round_limit": boss.round_limit,
    }
    resource = plan_optimal_resource_path(maze)
    maze.notes["resource_plan"] = {
        "max_resource": resource.max_resource,
        "walk_path": _serialize_path(resource.walk_path),
        "main_path": _serialize_path(resource.main_path),
        "branch_gains": resource.branch_gains,
    }
    ai_result = run_greedy_ai(maze)
    cross_eval = evaluate_maze_with_profiles(maze)
    maze.notes["ai_probe"] = {
        "success": ai_result.success,
        "final_coins": ai_result.final_coins,
        "steps": len(ai_result.path) - 1,
        "score_ratio": ai_result.score_ratio,
    }
    maze.notes["cross_eval"] = {
        **_serialize_cross_eval(cross_eval),
    }

    branch_gap = resource.max_resource - ai_result.final_coins
    score = (
        maze.design_score
        + resource.max_resource * 0.9
        + max(0, branch_gap) * 1.3
        + (12 if ai_result.success else 0)
        + (15 - abs(ai_result.score_ratio - 3.0) * 4)
        + cross_eval.maze_score
    )
    maze.design_score = score
    maze.notes["selection_score"] = score
    return maze, metrics, resource, boss, ai_result, cross_eval


def _candidate_analysis_payload(generated, metrics, resource, boss, ai_result, cross_eval, selection_score):
    return {
        "generator": generated.generator_name,
        "metrics": metrics.__dict__,
        "resource": {
            "max_resource": resource.max_resource,
            "walk_path": _serialize_path(resource.walk_path),
            "main_path": _serialize_path(resource.main_path),
            "branch_gains": resource.branch_gains,
        },
        "boss": {
            "min_rounds": boss.min_rounds,
            "skill_sequence": boss.skill_sequence,
            "damage_sequence": boss.damage_sequence,
            "round_limit": boss.round_limit,
            "coin_consumption": boss.coin_consumption,
        },
        "ai_probe": {
            "success": ai_result.success,
            "final_coins": ai_result.final_coins,
            "path": _serialize_path(ai_result.path),
            "steps": len(ai_result.path) - 1,
            "score_ratio": ai_result.score_ratio,
        },
        "cross_eval": _serialize_cross_eval(cross_eval),
        "selection_score": selection_score,
    }


def build_project(size: int, seed: int, open_viewer: bool = False) -> None:
    dirs = _project_dirs()
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    config = MazeConfig(size=size, random_seed=seed)
    generated_list = generate_all(size=size, seed=seed)
    candidates = []
    decorated_grids = {}

    for generated in generated_list:
        maze, metrics, resource, boss, ai_result, cross_eval = _build_single_candidate(generated, config)
        errors = validate_course_format(maze)
        if errors:
            raise ValueError(f"{generated.generator_name} invalid: {errors}")
        decorated_grids[generated.generator_name] = maze.grid

        gen_dir = dirs["demo"] / generated.generator_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        save_snapshot_sequence(generated.snapshots, gen_dir, prefix=generated.generator_name)
        save_ascii_grid(maze.grid, gen_dir / "maze.txt")
        save_grid_image(maze.grid, gen_dir / "maze.png", title=generated.generator_name)
        save_path_overlay(
            maze,
            resource.walk_path,
            gen_dir / "resource_path.png",
            title=f"{generated.generator_name} resource path",
        )
        save_json(
            gen_dir / "analysis.json",
            _candidate_analysis_payload(generated, metrics, resource, boss, ai_result, cross_eval, maze.design_score),
        )
        candidates.append((maze.design_score, maze, metrics, resource, boss, ai_result, cross_eval))

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_maze, best_metrics, best_resource, best_boss, best_ai, best_cross_eval = candidates[0]

    save_maze_json(dirs["final"] / "best_maze.json", best_maze, include_meta=False)
    save_maze_json(dirs["final"] / "best_maze_with_meta.json", best_maze, include_meta=True)
    save_ascii_grid(best_maze.grid, dirs["final"] / "best_maze.txt")
    save_grid_image(best_maze.grid, dirs["final"] / "best_maze.png", title="best maze")
    save_path_overlay(best_maze, best_resource.walk_path, dirs["final"] / "best_resource_path.png", "best resource path")

    summary = {
        "best_generator": best_maze.generator_name,
        "selection_score": best_maze.design_score,
        "design_metrics": best_metrics.__dict__,
        "resource_plan": {
            "max_resource": best_resource.max_resource,
            "walk_path": _serialize_path(best_resource.walk_path),
            "main_path": _serialize_path(best_resource.main_path),
        },
        "boss_plan": {
            "min_rounds": best_boss.min_rounds,
            "round_limit": best_boss.round_limit,
            "skill_sequence": best_boss.skill_sequence,
            "damage_sequence": best_boss.damage_sequence,
            "coin_consumption": best_boss.coin_consumption,
        },
        "ai_probe": {
            "success": best_ai.success,
            "final_coins": best_ai.final_coins,
            "steps": len(best_ai.path) - 1,
            "score_ratio": best_ai.score_ratio,
        },
        "cross_eval": {
            **_serialize_cross_eval(best_cross_eval),
        },
        "all_candidates": [
            {"generator": maze.generator_name, "selection_score": score}
            for score, maze, *_ in candidates
        ],
    }
    save_json(dirs["final"] / "summary.json", summary)
    create_visualization_bundle(
        dirs["output"] / "viewer",
        generated_list=generated_list,
        best_maze=best_maze,
        resource_frames=best_resource.frames,
        auto_open=open_viewer,
        decorated_grids=decorated_grids,
    )
    _write_docs(best_maze, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_submission(size: int, seed: int) -> str:
    dirs = _project_dirs()
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    submission_dir = dirs["output"] / "submission"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    config = MazeConfig(size=size, random_seed=seed)
    generated_list = generate_all(size=size, seed=seed)
    candidates = []
    decorated_grids = {}

    for generated in generated_list:
        maze, metrics, resource, boss, ai_result, cross_eval = _build_single_candidate(generated, config)
        errors = validate_course_format(maze)
        if errors:
            raise ValueError(f"{generated.generator_name} invalid: {errors}")
        decorated_grids[generated.generator_name] = maze.grid
        candidates.append((maze.design_score, generated, maze, metrics, resource, boss, ai_result, cross_eval))

        gen_dir = submission_dir / "task1_four_algorithms" / generated.generator_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        save_maze_json(gen_dir / "maze.json", maze, include_meta=False)
        save_maze_json(gen_dir / "maze_with_meta.json", maze, include_meta=True)
        save_json(
            gen_dir / "generation_process.json",
            {
                "algorithm": generated.generator_name,
                "size": size,
                "frames": [
                    {
                        "index": i + 1,
                        "grid": frame,
                        "event": generated.events[i] if i < len(generated.events) else {},
                    }
                    for i, frame in enumerate(generated.snapshots)
                ],
            },
        )
        save_json(
            gen_dir / "analysis.json",
            _candidate_analysis_payload(generated, metrics, resource, boss, ai_result, cross_eval, maze.design_score),
        )
        save_ascii_grid(maze.grid, gen_dir / "maze.txt")
        save_grid_image(maze.grid, gen_dir / "maze.png", title=f"{generated.generator_name} maze")
        save_path_overlay(maze, resource.walk_path, gen_dir / "resource_path.png", f"{generated.generator_name} resource path")

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_generated, best_maze, best_metrics, best_resource, best_boss, best_ai, best_cross_eval = candidates[0]

    final_dir = submission_dir / "best_maze"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_maze_json(final_dir / "best_maze.json", best_maze, include_meta=False)
    save_maze_json(final_dir / "best_maze_with_meta.json", best_maze, include_meta=True)
    save_ascii_grid(best_maze.grid, final_dir / "best_maze.txt")
    save_grid_image(best_maze.grid, final_dir / "best_maze.png", title="best maze")
    save_path_overlay(best_maze, best_resource.walk_path, final_dir / "best_resource_path.png", "best resource path")

    save_maze_json(dirs["final"] / "best_maze.json", best_maze, include_meta=False)
    save_maze_json(dirs["final"] / "best_maze_with_meta.json", best_maze, include_meta=True)
    save_ascii_grid(best_maze.grid, dirs["final"] / "best_maze.txt")
    save_grid_image(best_maze.grid, dirs["final"] / "best_maze.png", title="best maze")
    save_path_overlay(best_maze, best_resource.walk_path, dirs["final"] / "best_resource_path.png", "best resource path")

    ai_experiments = {
        "purpose": "Use multiple distinct 3x3-limited AI profiles to test whether the submitted map separates player strategies.",
        "best_generator": best_generated.generator_name,
        "score_formula": {
            "raw_score": "x_ij = final_coin / move_steps",
            "normalization": "n_ij = (x_ij - min_i x_ij) / (max_i x_ij - min_i x_ij + eps)",
            "maze_proxy_score": "100 * (distinguishability * stability * balance) ** (1/3)",
        },
        "notes": [
            "There are at least five distinct AI players; the exact count is recorded in ai_count.",
            "Full Spearman stability requires the class-wide AI x maze matrix; this package reports a single-maze neutral stability proxy and all raw data needed for the final matrix.",
        ],
        **_serialize_cross_eval(best_cross_eval),
    }
    save_json(submission_dir / "ai_experiments.json", ai_experiments)

    interface_dir = submission_dir / "interface_results"
    interface_dir.mkdir(parents=True, exist_ok=True)
    interface_samples = {
        "task2": Path("接口文件") / "task2_DP" / "接口文件" / "maze_15_15.json",
        "task3": Path("接口文件") / "task3_boss" / "接口文件" / "boss_case.json",
        "task4": Path("接口文件") / "task4_resourcePickup" / "接口文件" / "case_0002.json",
        "task5": Path("接口文件") / "task5_explore_maze" / "接口文件" / "maze_7_7_0.json",
    }
    interface_results = {}
    for task, input_path in interface_samples.items():
        output_path = interface_dir / f"{task}_result.json"
        interface_results[task] = run_interface_task(task, str(input_path), str(output_path))

    own_task2_input = interface_dir / "best_maze_task2_input.json"
    own_task3_input = interface_dir / "best_maze_task3_input.json"
    own_task5_input = interface_dir / "best_maze_task5_input.json"
    save_json(own_task2_input, {"maze": best_maze.grid})
    save_json(own_task3_input, {"B": best_maze.boss_health, "PlayerSkills": [[s.damage, s.cooldown] for s in best_maze.player_skills]})
    save_json(own_task5_input, maze_to_dict(best_maze, include_meta=False))
    run_interface_task("task2", str(own_task2_input), str(interface_dir / "best_maze_task2_result.json"))
    run_interface_task("task3", str(own_task3_input), str(interface_dir / "best_maze_task3_result.json"))
    run_interface_task("task5", str(own_task5_input), str(interface_dir / "best_maze_task5_result.json"))

    viewer_path = create_visualization_bundle(
        submission_dir / "viewer",
        generated_list=generated_list,
        best_maze=best_maze,
        resource_frames=best_resource.frames,
        auto_open=False,
        decorated_grids=decorated_grids,
    )
    create_visualization_bundle(
        dirs["output"] / "viewer",
        generated_list=generated_list,
        best_maze=best_maze,
        resource_frames=best_resource.frames,
        auto_open=False,
        decorated_grids=decorated_grids,
    )

    summary = {
        "submission_dir": str(submission_dir.resolve()),
        "size": size,
        "seed": seed,
        "best_generator": best_generated.generator_name,
        "best_maze_file": "best_maze/best_maze.json",
        "selection_score": best_maze.design_score,
        "design_metrics": best_metrics.__dict__,
        "resource_plan": {
            "max_resource": best_resource.max_resource,
            "path_length": max(0, len(best_resource.walk_path) - 1),
            "walk_path": _serialize_path(best_resource.walk_path),
            "main_path": _serialize_path(best_resource.main_path),
        },
        "boss_plan": {
            "min_rounds": best_boss.min_rounds,
            "round_limit": best_boss.round_limit,
            "skill_sequence": best_boss.skill_sequence,
            "damage_sequence": best_boss.damage_sequence,
            "coin_consumption": best_boss.coin_consumption,
        },
        "ai_experiments": ai_experiments,
        "interface_results": interface_results,
        "viewer": str(viewer_path.resolve()),
        "all_candidates": [
            {"generator": generated.generator_name, "selection_score": score}
            for score, generated, *_ in candidates
        ],
    }
    save_json(submission_dir / "summary.json", summary)
    save_json(dirs["final"] / "summary.json", summary)
    _write_submission_readme(submission_dir, summary)
    _write_docs(best_maze, summary)
    return str(submission_dir.resolve())


def build_viewer_only(size: int, seed: int, auto_open: bool = True) -> str:
    dirs = _project_dirs()
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    config = MazeConfig(size=size, random_seed=seed)
    generated_list = generate_all(size=size, seed=seed)
    candidates = []
    decorated_grids = {}
    for generated in generated_list:
        maze, metrics, resource, boss, ai_result, cross_eval = _build_single_candidate(generated, config)
        errors = validate_course_format(maze)
        if errors:
            raise ValueError(f"{generated.generator_name} invalid: {errors}")
        decorated_grids[generated.generator_name] = maze.grid
        candidates.append((maze.design_score, generated, maze, resource))

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, _, best_maze, best_resource = candidates[0]
    viewer_path = create_visualization_bundle(
        dirs["output"] / "viewer",
        generated_list=generated_list,
        best_maze=best_maze,
        resource_frames=best_resource.frames,
        auto_open=auto_open,
        decorated_grids=decorated_grids,
    )
    return str(viewer_path)


def analyze_maze_file(path: str) -> None:
    maze = load_maze_json(path)
    resource = plan_optimal_resource_path(maze)
    boss = solve_boss_battle(maze.boss_health, maze.player_skills, maze.coin_consumption)
    ai_result = run_greedy_ai(maze)
    report = {
        "generator": maze.generator_name,
        "resource": {
            "max_resource": resource.max_resource,
            "walk_path": _serialize_path(resource.walk_path),
        },
        "boss": {
            "min_rounds": boss.min_rounds,
            "skill_sequence": boss.skill_sequence,
            "round_limit": boss.round_limit,
        },
        "ai": {
            "success": ai_result.success,
            "final_coins": ai_result.final_coins,
            "steps": len(ai_result.path) - 1,
            "score_ratio": ai_result.score_ratio,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def demo_ai_on_maze(path: str) -> None:
    dirs = _project_dirs()
    maze = load_maze_json(path)
    result = run_greedy_ai(maze)
    save_path_overlay(maze, result.path, dirs["demo"] / "ai_demo_path.png", "AI demo path")
    save_json(
        dirs["demo"] / "ai_demo_steps.json",
        {
            "success": result.success,
            "final_coins": result.final_coins,
            "boss_rounds": result.boss_rounds,
            "boss_skill_sequence": result.boss_skill_sequence,
            "score_ratio": result.score_ratio,
            "path": _serialize_path(result.path),
            "steps": [
                {
                    "position": [step.position[0], step.position[1]],
                    "action": step.action,
                    "coins": step.coins,
                    "visible": step.visible,
                }
                for step in result.steps
            ],
        },
    )
    print(json.dumps({"success": result.success, "score_ratio": result.score_ratio}, ensure_ascii=False, indent=2))


def export_report_data() -> None:
    dirs = _project_dirs()
    best_path = dirs["final"] / "best_maze.json"
    if not best_path.exists():
        build_project(size=15, seed=42)
    maze = load_maze_json(best_path)
    resource = plan_optimal_resource_path(maze)
    boss = solve_boss_battle(maze.boss_health, maze.player_skills, maze.coin_consumption)
    ai_result = run_greedy_ai(maze)

    save_json(
        dirs["docs"] / "report_data.json",
        {
            "task_mapping": {
                "maze_generation": ["分治", "最小生成树", "DFS 回溯", "BFS 扩展"],
                "resource_planning": "动态规划 + 树形分支收益搜索",
                "boss_planning": "分支限界",
                "ai_interaction": "3x3 视野贪心联调器",
            },
            "best_maze_summary": {
                "generator": maze.generator_name,
                "boss_health": maze.boss_health,
                "player_skills": [[s.damage, s.cooldown] for s in maze.player_skills],
                "minRouds": maze.min_rounds,
                "CoinConsumption": maze.coin_consumption,
            },
            "resource_plan": {
                "max_resource": resource.max_resource,
                "path_length": len(resource.walk_path) - 1,
            },
            "boss_plan": {
                "min_rounds": boss.min_rounds,
                "round_limit": boss.round_limit,
                "skill_sequence": boss.skill_sequence,
            },
            "ai_demo": {
                "success": ai_result.success,
                "final_coins": ai_result.final_coins,
                "path_length": len(ai_result.path) - 1,
                "score_ratio": ai_result.score_ratio,
            },
        },
    )
    summary_path = dirs["final"] / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cross_text = ["# AI 玩家交叉测试结果", ""]
        cross = summary.get("cross_eval") or summary.get("ai_experiments", {})
        cross_text.append(f"- AI 玩家迷宫分数: {cross.get('maze_score', 0):.4f}")
        cross_text.append(f"- 区分度: {cross.get('distinguishability', 0):.4f}")
        cross_text.append(f"- 稳定性: {cross.get('stability', 0):.4f}")
        cross_text.append(f"- 平衡性: {cross.get('balance', 0):.4f}")
        cross_text.append("")
        cross_text.append("| AI 玩家 | 得分比值 | 归一化分 | 是否成功 | 剩余金币 | 步数 |")
        cross_text.append("| --- | ---: | ---: | --- | ---: | ---: |")
        normalized = cross.get("normalized_scores", {})
        for name, score in cross.get("ai_scores", {}).items():
            detail = cross.get("ai_results", {}).get(name, {})
            cross_text.append(
                f"| {name} | {score:.4f} | {normalized.get(name, 0):.4f} | {detail.get('success', False)} | {detail.get('final_coins', 0)} | {detail.get('steps', 0)} |"
            )
        (dirs["docs"] / "AI玩家交叉测试结果.md").write_text("\n".join(cross_text), encoding="utf-8")
    print(str(dirs["docs"] / "report_data.json"))


def _write_submission_readme(submission_dir: Path, summary: dict) -> None:
    ai = summary.get("ai_experiments", {})
    resource = summary.get("resource_plan", {})
    boss = summary.get("boss_plan", {})
    candidates = summary.get("all_candidates", [])
    candidate_lines = "\n".join(
        f"- {item['generator']}: selection_score={item['selection_score']:.2f}"
        for item in candidates
    )
    content = f"""# 地图设计组提交说明

## 一、提交包总览

本目录由 `python main.py prepare-submission --size {summary.get('size', 15)} --seed {summary.get('seed', 42)}` 自动生成，重点面向地图设计组任务，同时保留 AI 玩家组交互实验数据。

## 二、核心文件

- `best_maze/best_maze.json`：最终提交用最佳迷宫，字段严格保持 `maze`、`B`、`PlayerSkills`、`minRouds`、`CoinConsumption`。
- `best_maze/best_maze_with_meta.json`：带生成来源、评分和路径信息的调试版，不作为正式接口输入。
- `task1_four_algorithms/*/generation_process.json`：四种算法的逐步构建帧。
- `task1_four_algorithms/*/analysis.json`：每个候选图的路径、BOSS、AI 区分度分析。
- `ai_experiments.json`：{ai.get('ai_count', 0)} 个不同 AI 玩家在最佳图上的实验结果。
- `interface_results/`：官方接口样例与本项目最佳迷宫的接口运行输出。
- `viewer/viewer.html`：离线可视化页面，可逐步播放四种生成过程和最佳路径过程。

## 三、最佳地图摘要

- 迷宫规模 n：{summary.get('size')}
- 随机种子：{summary.get('seed')}
- 最佳生成算法：{summary.get('best_generator')}
- 地图选择评分：{summary.get('selection_score', 0):.2f}
- 最佳资源路径收益：{resource.get('max_resource')}
- 最佳资源路径步数：{resource.get('path_length')}
- BOSS 最少回合：{boss.get('min_rounds')}
- 提供给 AI 的 `minRouds`：{boss.get('round_limit')}

## 四、四种迷宫算法

本项目实现并保存了四类生成过程：

- 分治生成：`divide_conquer`
- 最小生成树贪心：`mst_greedy`
- DFS 回溯：`dfs_backtracking`
- BFS/分支限界风格扩展：`bfs_expansion`

候选图排序如下：

{candidate_lines}

## 五、AI 玩家交互实验

AI 玩家数量：{ai.get('ai_count', 0)}，满足至少 5 个不同 AI 玩家的要求。实验使用 3x3 视野约束，并记录原始分、归一化分、路径和最终金币。

- 区分度：{ai.get('distinguishability', 0):.4f}
- 稳定性代理值：{ai.get('stability', 0):.4f}
- 平衡性：{ai.get('balance', 0):.4f}
- 综合迷宫代理分：{ai.get('maze_score', 0):.4f}

说明：完整 Spearman 稳定性需要全班 AI x 迷宫交叉矩阵。本提交包已输出单图可计算数据和中性稳定性代理值，便于后续并入班级总评矩阵。

## 六、实时可视化运行

离线查看：

```powershell
start output\\submission\\viewer\\viewer.html
```

实时输入 n 并重新生成：

```powershell
python main.py live-viewer --port 8765
```

打开页面后可输入 n 和 seed，点击“按 n 生成四种算法”或“只生成当前算法”，页面会实时调用后端生成迷宫构建过程与最佳路径过程，而不是预先生成 PNG 后播放。

## 七、接口验证命令

```powershell
python main.py interface --task task2 --input "接口文件\\task2_DP\\接口文件\\maze_15_15.json"
python main.py interface --task task3 --input "接口文件\\task3_boss\\接口文件\\boss_case.json"
python main.py interface --task task4 --input "接口文件\\task4_resourcePickup\\接口文件\\case_0002.json"
python main.py interface --task task5 --input "接口文件\\task5_explore_maze\\接口文件\\maze_7_7_0.json"
```

Task5 中第二次 BOSS 技能序列可能与样例展示序列不同，但同样达到最优 12 回合；评分关注最少回合、复活消耗、最终金币和路径合法性。
"""
    (submission_dir / "README_提交说明.md").write_text(content, encoding="utf-8")


def _write_docs(best_maze, summary) -> None:
    dirs = _project_dirs()
    ai = summary.get("ai_probe") or {}
    ai_experiments = summary.get("ai_experiments") or summary.get("cross_eval") or {}
    ai_score_ratio = ai.get("score_ratio")
    ai_line = (
        f"- AI 联调得分比值：{ai_score_ratio:.3f}"
        if isinstance(ai_score_ratio, (int, float))
        else f"- AI 玩家数量：{ai_experiments.get('ai_count', 0)}，区分度：{ai_experiments.get('distinguishability', 0):.4f}"
    )
    resource = summary.get("resource_plan", {})
    boss = summary.get("boss_plan", {})
    content = f"""# 任务书对照说明

## 已完成的迷宫设计组内容

1. 四种迷宫生成算法均已实现并输出可视化过程：
   - 分治
   - 最小生成树（贪心）
   - DFS 回溯
   - BFS 扩展
2. 最佳迷宫已输出为 `output/final/best_maze.json`。
3. 动态规划资源路径分析结果已输出，并附带过程路径图。
4. 分支限界 BOSS 战最优结果已输出，包括最少回合、技能序列、限定回合数和复活金币消耗。
5. 补充实现了 3x3 视野 AI 联调器，用于与地图进行交互测试和可视化展示。

## 当前最佳迷宫摘要

- 来源算法：{summary['best_generator']}
- 选择评分：{summary['selection_score']:.2f}
- 最佳资源路径收益：{resource.get('max_resource')}
- BOSS 回合下界：{boss.get('min_rounds')}
- 提供给 AI 的 `minRouds`：{boss.get('round_limit')}
{ai_line}

## 关键交付路径

- 最终迷宫 JSON：`output/final/best_maze.json`
- 最终迷宫图片：`output/final/best_maze.png`
- 资源路径图：`output/final/best_resource_path.png`
- 提交包：`output/submission`
- 候选分析：`output/demo/*/analysis.json`
- 报告数据：`docs/report_data.json`

## 可视化运行

- 离线逐步查看：打开 `output/submission/viewer/viewer.html`
- 实时输入 n 生成：执行 `python main.py live-viewer --port 8765`
"""
    (dirs["docs"] / "任务对照说明.md").write_text(content, encoding="utf-8")


