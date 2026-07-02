from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="算法课设迷宫设计组提交包")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="生成候选迷宫并输出最佳迷宫")
    build.add_argument("--size", type=int, default=15)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--open-viewer", action="store_true", help="生成后自动打开逐步可视化查看器")

    viewer = subparsers.add_parser("viewer", help="生成并打开逐步可视化查看器")
    viewer.add_argument("--size", type=int, default=15)
    viewer.add_argument("--seed", type=int, default=42)

    live_viewer = subparsers.add_parser("live-viewer", help="启动实时生成的可视化查看器")
    live_viewer.add_argument("--host", default="127.0.0.1")
    live_viewer.add_argument("--port", type=int, default=8765)

    interface = subparsers.add_parser("interface", help="按正式接口样例运行指定任务")
    interface.add_argument("--task", choices=["task2", "task3", "task4", "task5"], required=True)
    interface.add_argument("--input", required=True)
    interface.add_argument("--output")

    prepare = subparsers.add_parser("prepare-submission", help="生成提交和实验所需全部文件")
    prepare.add_argument("--size", type=int, default=15)
    prepare.add_argument("--seed", type=int, default=42)

    analyze = subparsers.add_parser("analyze", help="分析指定迷宫")
    analyze.add_argument("--maze", required=True)

    demo_ai = subparsers.add_parser("demo-ai", help="运行 AI 玩家演示")
    demo_ai.add_argument("--maze", required=True)

    subparsers.add_parser("report-data", help="导出报告素材")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        from pipeline import build_project

        build_project(size=args.size, seed=args.seed, open_viewer=args.open_viewer)
        return

    if args.command == "viewer":
        from pipeline import build_viewer_only

        print(build_viewer_only(size=args.size, seed=args.seed, auto_open=True))
        return

    if args.command == "live-viewer":
        from visualization.live_server import run_live_viewer

        run_live_viewer(host=args.host, port=args.port, open_browser=True)
        return

    if args.command == "interface":
        import json
        from task3_boss_ai_interface.runner import run_interface_task

        result = run_interface_task(args.task, args.input, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "prepare-submission":
        from pipeline import prepare_submission

        print(prepare_submission(size=args.size, seed=args.seed))
        return

    if args.command == "analyze":
        from pipeline import analyze_maze_file

        analyze_maze_file(args.maze)
        return

    if args.command == "demo-ai":
        from pipeline import demo_ai_on_maze

        demo_ai_on_maze(args.maze)
        return

    if args.command == "report-data":
        from pipeline import export_report_data

        export_report_data()
        return

    parser.error("Unknown command")


