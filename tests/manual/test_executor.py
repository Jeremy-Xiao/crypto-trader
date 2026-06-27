"""
测试下单执行模块
需要配置API Key后运行
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from src.api.executor import TradingExecutor
from src.utils.logger import get_logger

def test_executor():
    """测试交易执行器"""
    
    # 检查API Key
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_SECRET_KEY")
    passphrase = os.getenv("OKX_PASSPHRASE")
    
    if not api_key:
        print("❌ 请先配置 .env 文件中的 API Key")
        return
    
    # 创建执行器（模拟盘模式）
    logger = get_logger("test_executor")
    executor = TradingExecutor(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        simulated=True,  # 使用模拟盘
        logger=logger
    )
    
    print("\n" + "=" * 50)
    print("  下单执行模块测试")
    print("=" * 50)
    
    # 1. 测试获取余额
    print("\n【测试1】获取账户余额...")
    balance = executor.get_balance("USDT")
    if balance:
        for bal_data in balance:
            for bal in bal_data.get("balData", []):
                print(f"  {bal.get('ccy')}: 总额={bal.get('bal')}, 可用={bal.get('availBal')}")
    
    # 2. 测试获取价格
    print("\n【测试2】获取当前价格...")
    price = executor.get_current_price("BTC-USDT")
    if price:
        print(f"  BTC-USDT 当前价格: {price}")
    
    # 3. 测试获取最优价格
    print("\n【测试3】获取最优买卖价...")
    best_price = executor.get_best_price("BTC-USDT")
    print(f"  买一价: {best_price['bid']}")
    print(f"  卖一价: {best_price['ask']}")
    print(f"  最新价: {best_price['last']}")
    
    # 4. 测试下单（极小金额）
    print("\n【测试4】测试下单...")
    
    # 计算极小测试金额（约1 USDT）
    test_amount = 1.0 / price  # 约0.0006 BTC
    
    print(f"  准备买入: {test_amount:.6f} BTC (约1 USDT)")
    
    result = executor.place_market_buy(
        instId="BTC-USDT",
        amount=test_amount
    )
    
    if result.get("code") == "0":
        order_data = result.get("data", [{}])[0]
        ord_id = order_data.get("ordId")
        print(f"  ✅ 下单成功! 订单ID: {ord_id}")
        
        # 查询订单状态
        print("\n【测试5】查询订单状态...")
        status = executor.get_order_status("BTC-USDT", ord_id)
        print(f"  订单状态: {status}")
        
        # 查询订单详情
        order_info = executor.get_order_info("BTC-USDT", ord_id)
        if order_info:
            print(f"  成交均价: {order_info.get('avgPx')}")
            print(f"  成交数量: {order_info.get('accFillSz')}")
    else:
        print(f"  ❌ 下单失败: {result.get('msg')}")
    
    # 6. 测试获取未成交订单
    print("\n【测试6】获取未成交订单...")
    pending = executor.get_orders_pending("SPOT")
    print(f"  未成交订单数: {len(pending)}")
    
    print("\n" + "=" * 50)
    print("  测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_executor()