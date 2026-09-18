#!/bin/zsh
# 模拟盘一键启动脚本（防双开 + 后台常驻 + 落盘 PID）
# 用法: ./scripts/start_trader.sh
set -u
cd /Users/hello/Documents/workspace/crypto-trader

# 防双开：已在跑就退出（双开会在同一根K线重复下单）
if pgrep -f "live_monitor.py" > /dev/null; then
    echo "⚠️  模拟盘已在运行（PID: $(pgrep -f 'live_monitor.py' | tr '\n' ' ')），不重复启动。"
    echo "   查看日志: tail -f logs/live_monitor.log"
    exit 0
fi

mkdir -p logs state
# 代理：裸连 OKX 不稳定（间歇可达），统一走本机 Clash(mixed-port 7897)。
# CRYPTO_PROXY 供情绪数据源(src/monitor/base.py)使用；HTTP(S)_PROXY 供 okx_rest 使用。
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export CRYPTO_PROXY=http://127.0.0.1:7897
nohup venv/bin/python -u scripts/live_monitor.py --interval 3600 --exec okx \
    --cadence daily --warmup 200 \
    >> logs/live_monitor.log 2>&1 &
PID=$!
echo $PID > state/live_monitor.pid
sleep 2
if kill -0 $PID 2>/dev/null; then
    echo "✅ 模拟盘已启动 PID=$PID（OKX模拟盘 + 日线决策节奏，每小时风控巡检）"
    echo "   看日志:   tail -f logs/live_monitor.log"
    echo "   看成绩:   venv/bin/python scripts/paper_report.py"
    echo "   停止:     kill \$(cat state/live_monitor.pid)"
else
    echo "❌ 启动失败，请看日志: tail -20 logs/live_monitor.log"
    exit 1
fi
