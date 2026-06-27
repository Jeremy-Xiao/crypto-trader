#!/usr/bin/env python3
"""
Crypto Trader Main Entry Point
加密货币自动化交易系统主入口
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config_manager import ConfigManager
from src.utils.logger import get_logger, TradeLogger
from src.api.okx_rest import OKXPublicAPI
from src.api.executor import TradingExecutor


class CryptoTrader:
    """加密货币交易系统"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化交易系统
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = get_logger("CryptoTrader")
        self.trade_logger = TradeLogger()
        
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load()
        
        # 初始化API
        self.public_api = OKXPublicAPI()
        
        # 初始化执行器（如果有API配置）
        self.executor = None
        if self.config.okx.api_key:
            self.executor = TradingExecutor(
                api_key=self.config.okx.api_key,
                secret_key=self.config.okx.secret_key,
                passphrase=self.config.okx.passphrase,
                simulated=self.config.okx.simulated,
                logger=self.logger
            )
        
        self.logger.info("CryptoTrader 初始化完成")
    
    def get_market_status(self, instId: str = "BTC-USDT") -> dict:
        """
        获取市场状态
        
        Args:
            instId: 产品ID
        
        Returns:
            市场状态
        """
        ticker = self.public_api.get_ticker(instId)
        
        if ticker.get("code") == "0" and ticker.get("data"):
            data = ticker["data"][0]
            return {
                "instId": instId,
                "last": float(data.get("last", 0)),
                "bid": float(data.get("bidPx", 0)),
                "ask": float(data.get("askPx", 0)),
                "open24h": float(data.get("open24h", 0)),
                "high24h": float(data.get("high24h", 0)),
                "low24h": float(data.get("low24h", 0)),
                "vol24h": float(data.get("vol24h", 0)),
                "timestamp": datetime.now().isoformat()
            }
        
        return {}
    
    def display_market_info(self, instIds: list = ["BTC-USDT", "ETH-USDT"]):
        """
        显示市场信息
        
        Args:
            instIds: 产品ID列表
        """
        print("\n" + "=" * 60)
        print("  实时行情")
        print("=" * 60)
        
        for instId in instIds:
            status = self.get_market_status(instId)
            
            if status:
                change_pct = 0
                if status["open24h"] > 0:
                    change_pct = (status["last"] - status["open24h"]) / status["open24h"] * 100
                
                print(f"\n{instId}:")
                print(f"  最新价: {status['last']}")
                print(f"  买一价: {status['bid']}")
                print(f"  卖一价: {status['ask']}")
                print(f"  24H涨跌: {change_pct:.2f}%")
                print(f"  24H最高: {status['high24h']}")
                print(f"  24H最低: {status['low24h']}")
                print(f"  24H成交量: {status['vol24h']}")
    
    def check_system_status(self) -> bool:
        """检查系统状态"""
        # 检查API连接
        time_result = self.public_api.get_server_time()
        
        if time_result.get("code") == "0":
            self.logger.info("API连接正常")
            return True
        else:
            self.logger.error("API连接失败")
            return False
    
    def run(self):
        """运行交易系统"""
        print("\n" + "=" * 60)
        print("  Crypto Trader v0.1.0")
        print("  加密货币自动化交易系统")
        print("=" * 60)
        
        # 检查系统状态
        if not self.check_system_status():
            print("❌ 系统状态异常，请检查API配置")
            return
        
        # 显示市场信息
        self.display_market_info()
        
        # 显示账户信息（如果有执行器）
        if self.executor:
            print("\n" + "=" * 60)
            print("  账户信息")
            print("=" * 60)
            
            balance = self.executor.get_balance("USDT")
            if balance:
                for bal_data in balance:
                    for bal in bal_data.get("balData", []):
                        print(f"\n{bal.get('ccy')}:")
                        print(f"  总额: {bal.get('bal')}")
                        print(f"  可用: {bal.get('availBal')}")
        else:
            print("\n⚠️ 未配置API Key，无法获取账户信息")
            print("请配置 .env 文件或 config/config.yaml")


def main():
    """主函数"""
    trader = CryptoTrader()
    trader.run()


if __name__ == "__main__":
    main()