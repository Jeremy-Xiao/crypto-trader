#!/usr/bin/env python
"""小时级实时监控调度器 CLI。

把「被动等运行」升级为「主动定时循环」：每个周期自己抓行情+情绪，喂给元策略，
出信号后交给执行器（默认模拟，不下真单）。

用法示例：
    # 单次测试（推荐先跑这个，确认能连通、能出信号）
    venv/bin/python scripts/live_monitor.py --once

    # 小时级常驻循环（Ctrl+C 退出）
    venv/bin/python scripts/live_monitor.py --interval 3600

    # 开波动率目标化（安全气囊②，更稳但少赚）
    venv/bin/python scripts/live_monitor.py --once --vol-target 0.45

    # 只跑 SOL，15 分钟一轮
    venv/bin/python scripts/live_monitor.py --symbols SOL-USDT --bar 15m --interval 900

安全说明：默认 DryRun 模拟执行，绝不会真实下单。真实下单需显式实现 OKXExecutor。
"""
import os
import sys
import json
import argparse
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))  # OKX 凭证从这里读

from src.live.orchestrator import LiveOrchestrator, DryRunExecutor, setup_logging  # noqa: E402
from src.live.paper import PaperBroker  # noqa: E402
from src.live.okx_paper import OKXPaperExecutor  # noqa: E402

DEFAULT_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]


def build_parser():
    p = argparse.ArgumentParser(description="小时级实时监控调度器（元策略 + 情绪过滤 + 波动率调仓）")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                   help=f"交易对，默认 {' '.join(DEFAULT_SYMBOLS)}")
    p.add_argument("--mode", default="adaptive", choices=["adaptive", "ensemble"],
                   help="元策略模式：adaptive=王者通吃(默认，回测最优) / ensemble=混合投票")
    p.add_argument("--bar", default="1H", help="K线周期，默认 1H")
    p.add_argument("--interval", type=int, default=3600, help="循环间隔秒数，默认 3600(1小时)")
    p.add_argument("--warmup", type=int, default=120, help="预热 K 线根数，默认 120")
    p.add_argument("--once", action="store_true", help="只跑一个周期就退出（测试用）")
    p.add_argument("--vol-target", type=float, default=None,
                   help="波动率目标化年化目标(如 0.45)；不传=关闭。开启更稳但绝对收益略低")
    p.add_argument("--min-adx", type=float, default=0.0, help="ADX 门槛，默认 0(不过滤)")
    p.add_argument("--no-bias", action="store_true", help="关闭情绪过滤器（气囊①）")
    p.add_argument("--balance", type=float, default=10000.0, help="模拟初始资金，默认 10000")
    p.add_argument("--state-dir", default="state", help="模拟盘状态目录（持仓/流水/权益曲线），默认 state/")
    p.add_argument("--no-persist", action="store_true",
                   help="用纯内存执行器（DryRunExecutor），不落盘；默认为持久化模拟盘 PaperBroker")
    p.add_argument("--exec", dest="exec_mode", default="local", choices=["local", "okx"],
                   help="执行模式：local=本地记账模拟盘(默认) / okx=OKX模拟盘真实下单"
                        "(用 .env 凭证，不碰真钱)")
    p.add_argument("--cadence", default="bar", choices=["bar", "daily"],
                   help="决策节奏：bar=每根K线决策(1H下即每小时，已证伪) / "
                        "daily=每天日K收盘后决策一次(与回测语义对齐，推荐)")
    p.add_argument("--out", default="backtest_reports/live_tick.json", help="每周期报告落盘路径")
    p.add_argument("--log", default="logs/live_monitor.log", help="日志文件路径")
    p.add_argument("--force", action="store_true",
                   help="跳过防双开检查（仅在你确认没有其它实例在跑时使用）")
    return p


def main():
    args = build_parser().parse_args()
    setup_logging(args.log)
    log = logging.getLogger("live.cli")

    # 防双开：已有实例在跑则拒绝（两个进程对同一根 K 线重复下单 = 仓位翻倍）。
    # 垂死进程（刚被 kill 还未退净）会导致误报，故等待重试一次再判定。
    import time as _time

    def _existing_pids():
        """两级判活：PID 文件(kill -0，最可靠) + pgrep 精确匹配启动命令。

        pgrep 宽泛匹配 "live_monitor.py" 会误捕瞬态进程（如本启动链自身），
        故 pgrep 只匹配完整启动命令，且仅作 PID 文件缺失时的兜底。
        """
        pids = set()
        pid_file = os.path.join(args.state_dir, "live_monitor.pid")
        if os.path.exists(pid_file):
            try:
                pid = int(open(pid_file).read().strip())
                if pid == os.getpid():
                    pass  # 读到的是自己（start_trader.sh 先写 pid 文件、本进程后检查的竞态），不算重复
                else:
                    os.kill(pid, 0)  # 存在性检查
                    pids.add(pid)
            except (ValueError, OSError):
                pass
        # 注意：不能用 pgrep -f 宽泛/精确匹配——zsh -c 执行的命令文本里
        # 含 "live_monitor.py" 字样，pgrep 会匹配到 shell 自身导致永远误报。
        # PID 文件 + kill -0 是唯一可靠的判活方式。
        return sorted(pids)

    others = _existing_pids()
    if others:
        log.warning(f"检测到疑似已有实例 (PID: {others})，等待 3s 后复查…")
        _time.sleep(3)
        others = _existing_pids()
    if others and not args.force:
        log.error(f"已有模拟盘进程在运行 (PID: {others})，拒绝启动以避免重复下单。"
                  f"确认无其它实例可用 --force 强制启动。")
        print(f"\u274c 已有模拟盘在运行 (PID: {others})。查看: tail -f logs/live_monitor.log")
        sys.exit(1)

    meta_params = {"min_adx": args.min_adx}
    if args.vol_target is not None:
        meta_params["vol_target_ann"] = args.vol_target

    if args.exec_mode == "okx":
        executor = OKXPaperExecutor(state_dir=args.state_dir,
                                    per_symbol_budget=args.balance)
    elif args.no_persist:
        executor = DryRunExecutor(initial_balance=args.balance)
    else:
        executor = PaperBroker(initial_balance=args.balance, state_dir=args.state_dir)
    # 写入自己的 PID（防双开判活依据；程序退出时由下次启动的 kill -0 失效自动通过）
    os.makedirs(args.state_dir, exist_ok=True)
    with open(os.path.join(args.state_dir, "live_monitor.pid"), "w") as f:
        f.write(str(os.getpid()))

    orch = LiveOrchestrator(
        symbols=args.symbols,
        mode=args.mode,
        meta_params=meta_params,
        bar=args.bar,
        interval_seconds=args.interval,
        warmup_bars=args.warmup,
        executor=executor,
        decision_cadence=args.cadence,
    )
    if args.no_bias:
        orch.use_bias = False

    log.info("=" * 84)
    log.info(f"实时调度器启动 | symbols={args.symbols} mode={args.mode} bar={args.bar} "
             f"决策节奏={args.cadence}")
    log.info(f"气囊① 情绪过滤={'关' if args.no_bias else '开'} | "
             f"气囊② 波动率目标={args.vol_target if args.vol_target else '关'}")
    log.info(f"执行器={'OKX模拟盘(真实下单,不碰真钱)' if args.exec_mode == 'okx' else ('DryRun(内存)' if args.no_persist else 'PaperBroker(本地记账模拟盘)')} "
             f"初始资金={args.balance}" + ("" if args.exec_mode == 'okx' else f" 状态目录={args.state_dir}"))
    log.info("=" * 84)

    if args.once:
        report = orch.run_once()
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(f"单周期报告已保存: {args.out}")
        print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    else:
        orch.report_path = args.out
        orch.run_forever()


if __name__ == "__main__":
    main()
