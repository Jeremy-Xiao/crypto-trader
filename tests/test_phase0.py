"""
阶段0测试脚本
验证基础设施搭建完成情况
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.okx_rest import OKXPublicAPI
from src.api.okx_websocket import OKXWebSocketManager
from src.utils.logger import get_logger, TradeLogger
from src.config.config_manager import ConfigManager


def test_public_api():
    """测试公共API"""
    print("\n=== 测试公共API ===")
    
    api = OKXPublicAPI()
    
    # 1. 获取服务器时间
    result = api.get_server_time()
    if result.get("code") == "0":
        ts = result["data"][0]["ts"]
        print(f"✅ 服务器时间: {ts}")
    else:
        print(f"❌ 获取服务器时间失败")
        return False
    
    # 2. 获取BTC-USDT行情
    result = api.get_ticker("BTC-USDT")
    if result.get("code") == "0" and result.get("data"):
        ticker = result["data"][0]
        print(f"✅ BTC-USDT行情:")
        print(f"   最新价: {ticker.get('last')}")
        print(f"   买一价: {ticker.get('bidPx')}")
        print(f"   卖一价: {ticker.get('askPx')}")
        print(f"   24H涨跌: {ticker.get('open24h')} -> {ticker.get('last')}")
    else:
        print(f"❌ 获取行情失败")
        return False
    
    # 3. 获取深度数据
    result = api.get_books("BTC-USDT", 5)
    if result.get("code") == "0" and result.get("data"):
        books = result["data"][0]
        print(f"✅ BTC-USDT深度:")
        print(f"   买盘: {books.get('bids', [])[:2]}")
        print(f"   卖盘: {books.get('asks', [])[:2]}")
    else:
        print(f"❌ 获取深度失败")
        return False
    
    # 4. 获取K线数据
    result = api.get_candles("BTC-USDT", "1H", 5)
    if result.get("code") == "0" and result.get("data"):
        print(f"✅ BTC-USDT K线 (最近5条):")
        for candle in result["data"][:2]:
            print(f"   {candle}")
    else:
        print(f"❌ 获取K线失败")
        return False
    
    return True


async def test_websocket():
    """测试WebSocket"""
    print("\n=== 测试WebSocket ===")
    
    received_data = {"tickers": False, "books": False}
    
    def on_ticker(data):
        if data:
            received_data["tickers"] = True
            ticker = data[0]
            print(f"✅ WebSocket行情: BTC-USDT = {ticker.get('last')}")
    
    def on_books(data):
        if data:
            received_data["books"] = True
            print(f"✅ WebSocket深度: 收到数据")
    
    manager = OKXWebSocketManager()
    manager.on_ticker(on_ticker)
    manager.on_books(on_books)
    
    subscriptions = [
        {"channel": "tickers", "instId": "BTC-USDT"},
    ]
    
    print("启动WebSocket (运行10秒)...")
    
    try:
        # 启动并运行一段时间
        await asyncio.wait_for(
            manager.start_public(subscriptions),
            timeout=10
        )
    except asyncio.TimeoutError:
        await manager.stop()
    
    # 检查是否收到数据
    if received_data["tickers"]:
        print("✅ WebSocket行情订阅成功")
        return True
    else:
        print("❌ WebSocket未收到数据")
        return False


def test_logger():
    """测试日志"""
    print("\n=== 测试日志系统 ===")
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 测试系统日志
    logger = get_logger("test")
    logger.info("测试系统日志")
    print("✅ 系统日志写入成功")
    
    # 测试交易日志
    trade_logger = TradeLogger()
    trade_logger.log_order(
        instId="BTC-USDT",
        side="buy",
        amount=0.001,
        price=42000,
        order_id="test_order",
        result={"code": "0", "msg": "test"}
    )
    print("✅ 交易日志写入成功")
    
    return True


def test_config():
    """测试配置"""
    print("\n=== 测试配置管理 ===")
    
    manager = ConfigManager()
    
    # 创建默认配置模板
    manager.save_yaml_template("config/config.yaml")
    print("✅ 配置模板已创建")
    
    # 检查.env.example
    if os.path.exists(".env.example"):
        print("✅ .env.example 存在")
    else:
        print("⚠️ .env.example 不存在，请手动创建")
    
    return True


def main():
    """主测试函数"""
    print("=" * 50)
    print("  阶段0 基础设施测试")
    print("=" * 50)
    
    results = {
        "公共API": False,
        "WebSocket": False,
        "日志系统": False,
        "配置管理": False
    }
    
    # 测试公共API
    results["公共API"] = test_public_api()
    
    # 测试WebSocket
    try:
        results["WebSocket"] = asyncio.run(test_websocket())
    except Exception as e:
        print(f"❌ WebSocket测试异常: {e}")
    
    # 测试日志
    results["日志系统"] = test_logger()
    
    # 测试配置
    results["config"] = test_config()
    
    # 输出总结
    print("\n" + "=" * 50)
    print("  测试结果汇总")
    print("=" * 50)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 阶段0检查点全部通过！")
        print("可以进行阶段1：策略原型与回测")
    else:
        print("\n⚠️ 部分测试未通过，请检查")
    
    return all_passed


if __name__ == "__main__":
    main()