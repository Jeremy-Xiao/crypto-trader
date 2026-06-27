"""
交易执行模块
封装下单、撤单、查询等功能，带日志和风控检查
"""

import logging
from typing import Optional, Dict, Any
from decimal import Decimal
from .okx_rest import OKXPrivateAPI, OKXPublicAPI


class TradingExecutor:
    """交易执行器"""
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        simulated: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化交易执行器
        
        Args:
            api_key: API Key
            secret_key: Secret Key
            passphrase: Passphrase
            simulated: 是否使用模拟盘
            logger: 日志器
        """
        self.private_api = OKXPrivateAPI(api_key, secret_key, passphrase, simulated)
        self.public_api = OKXPublicAPI()
        
        self.logger = logger or logging.getLogger(__name__)
        self.simulated = simulated
        
        # 风控参数
        self.max_position_pct = 0.2  # 单币种最大持仓比例
        self.max_order_size = None   # 最大单笔订单量
        self.min_order_size = None   # 最小单笔订单量
        
    def set_risk_params(
        self,
        max_position_pct: float = 0.2,
        max_order_size: Optional[float] = None,
        min_order_size: Optional[float] = None
    ):
        """
        设置风控参数
        
        Args:
            max_position_pct: 单币种最大持仓比例
            max_order_size: 最大单笔订单量
            min_order_size: 最小单笔订单量
        """
        self.max_position_pct = max_position_pct
        self.max_order_size = max_order_size
        self.min_order_size = min_order_size
    
    # ==================== 账户查询 ====================
    
    def get_balance(self, ccy: Optional[str] = None) -> Dict:
        """
        获取账户余额
        
        Args:
            ccy: 币种
        
        Returns:
            余额信息
        """
        result = self.private_api.get_balance(ccy)
        
        if result.get("code") == "0":
            self.logger.info(f"查询余额成功: {result['data']}")
            return result["data"]
        else:
            self.logger.error(f"查询余额失败: {result.get('msg')}")
            return []
    
    def get_available_balance(self, ccy: str) -> float:
        """
        获取可用余额
        
        Args:
            ccy: 币种
        
        Returns:
            可用余额
        """
        balances = self.get_balance(ccy)
        
        for balance_data in balances:
            for bal in balance_data.get("balData", []):
                if bal.get("ccy") == ccy:
                    return float(bal.get("availBal", 0))
        
        return 0.0
    
    def get_positions(self, instId: Optional[str] = None) -> Dict:
        """
        获取持仓
        
        Args:
            instId: 产品ID
        
        Returns:
            持仓信息
        """
        result = self.private_api.get_positions(instId=instId)
        
        if result.get("code") == "0":
            self.logger.info(f"查询持仓成功: {result['data']}")
            return result["data"]
        else:
            self.logger.error(f"查询持仓失败: {result.get('msg')}")
            return []
    
    # ==================== 行情查询 ====================
    
    def get_current_price(self, instId: str) -> Optional[float]:
        """
        获取当前价格
        
        Args:
            instId: 产品ID
        
        Returns:
            当前价格
        """
        result = self.public_api.get_ticker(instId)
        
        if result.get("code") == "0" and result.get("data"):
            ticker = result["data"][0]
            return float(ticker.get("last", 0))
        
        self.logger.error(f"获取价格失败: {result.get('msg')}")
        return None
    
    def get_best_price(self, instId: str) -> Dict:
        """
        获取最优买卖价
        
        Args:
            instId: 产品ID
        
        Returns:
            {"bid": 买一价, "ask": 卖一价}
        """
        result = self.public_api.get_ticker(instId)
        
        if result.get("code") == "0" and result.get("data"):
            ticker = result["data"][0]
            return {
                "bid": float(ticker.get("bidPx", 0)),
                "ask": float(ticker.get("askPx", 0)),
                "last": float(ticker.get("last", 0))
            }
        
        return {"bid": 0, "ask": 0, "last": 0}
    
    # ==================== 下单 ====================
    
    def place_market_buy(
        self,
        instId: str,
        amount: float,
        amount_type: str = "quote"
    ) -> Dict:
        """
        市价买入
        
        Args:
            instId: 产品ID
            amount: 数量
            amount_type: 数量类型 (base=币种数量, quote=计价币金额)
        
        Returns:
            下单结果
        """
        # 风控检查
        if self.min_order_size and amount < self.min_order_size:
            self.logger.warning(f"订单数量太小: {amount} < {self.min_order_size}")
            return {"code": "-1", "msg": "订单数量太小"}
        
        if self.max_order_size and amount > self.max_order_size:
            self.logger.warning(f"订单数量太大: {amount} > {self.max_order_size}")
            return {"code": "-1", "msg": "订单数量太大"}
        
        # 下单参数
        sz = str(amount)
        
        # 市价单需要指定数量类型
        body = {
            "instId": instId,
            "tdMode": "cash",
            "side": "buy",
            "ordType": "market",
            "sz": sz,
            "tgtCcy": "quote_ccy" if amount_type == "quote" else "base_ccy"
        }
        
        result = self.private_api._request("POST", "/api/v5/trade/order", body=body)
        
        if result.get("code") == "0":
            self.logger.info(f"市价买入成功: {instId} {amount} ({amount_type})")
        else:
            self.logger.error(f"市价买入失败: {result.get('msg')}")
        
        return result
    
    def place_market_sell(self, instId: str, amount: float) -> Dict:
        """
        市价卖出
        
        Args:
            instId: 产品ID
            amount: 数量
        
        Returns:
            下单结果
        """
        result = self.private_api.place_market_order(
            instId=instId,
            tdMode="cash",
            side="sell",
            sz=str(amount)
        )
        
        if result.get("code") == "0":
            self.logger.info(f"市价卖出成功: {instId} {amount}")
        else:
            self.logger.error(f"市价卖出失败: {result.get('msg')}")
        
        return result
    
    def place_limit_buy(
        self,
        instId: str,
        amount: float,
        price: float
    ) -> Dict:
        """
        限价买入
        
        Args:
            instId: 产品ID
            amount: 数量
            price: 价格
        
        Returns:
            下单结果
        """
        result = self.private_api.place_limit_order(
            instId=instId,
            tdMode="cash",
            side="buy",
            sz=str(amount),
            px=str(price)
        )
        
        if result.get("code") == "0":
            self.logger.info(f"限价买入成功: {instId} {amount} @ {price}")
        else:
            self.logger.error(f"限价买入失败: {result.get('msg')}")
        
        return result
    
    def place_limit_sell(
        self,
        instId: str,
        amount: float,
        price: float
    ) -> Dict:
        """
        限价卖出
        
        Args:
            instId: 产品ID
            amount: 数量
            price: 价格
        
        Returns:
            下单结果
        """
        result = self.private_api.place_limit_order(
            instId=instId,
            tdMode="cash",
            side="sell",
            sz=str(amount),
            px=str(price)
        )
        
        if result.get("code") == "0":
            self.logger.info(f"限价卖出成功: {instId} {amount} @ {price}")
        else:
            self.logger.error(f"限价卖出失败: {result.get('msg')}")
        
        return result
    
    # ==================== 撤单 ====================
    
    def cancel_order(self, instId: str, ordId: str) -> Dict:
        """
        撤销订单
        
        Args:
            instId: 产品ID
            ordId: 订单ID
        
        Returns:
            撤单结果
        """
        result = self.private_api.cancel_order(instId, ordId)
        
        if result.get("code") == "0":
            self.logger.info(f"撤单成功: {ordId}")
        else:
            self.logger.error(f"撤单失败: {result.get('msg')}")
        
        return result
    
    def cancel_all_pending_orders(self, instId: Optional[str] = None) -> int:
        """
        撤销所有未成交订单
        
        Args:
            instId: 产品ID（可选）
        
        Returns:
            撤销数量
        """
        pending = self.private_api.get_orders_pending(instId=instId)
        
        if pending.get("code") != "0":
            self.logger.error("获取未成交订单失败")
            return 0
        
        orders = pending.get("data", [])
        cancelled = 0
        
        for order in orders:
            result = self.cancel_order(order["instId"], order["ordId"])
            if result.get("code") == "0":
                cancelled += 1
        
        self.logger.info(f"撤销了 {cancelled} 个订单")
        return cancelled
    
    # ==================== 订单查询 ====================
    
    def get_order_status(self, instId: str, ordId: str) -> Optional[str]:
        """
        查询订单状态
        
        Args:
            instId: 产品ID
            ordId: 订单ID
        
        Returns:
            订单状态 (live/canceled/filled/partially_filled)
        """
        result = self.private_api.get_order(instId, ordId)
        
        if result.get("code") == "0" and result.get("data"):
            return result["data"][0].get("state")
        
        return None
    
    def get_order_info(self, instId: str, ordId: str) -> Optional[Dict]:
        """
        查询订单详情
        
        Args:
            instId: 产品ID
            ordId: 订单ID
        
        Returns:
            订单详情
        """
        result = self.private_api.get_order(instId, ordId)
        
        if result.get("code") == "0" and result.get("data"):
            return result["data"][0]
        
        return None
    
    # ==================== 便捷方法 ====================
    
    def buy_with_amount_pct(self, instId: str, pct: float) -> Dict:
        """
        用账户权益比例买入
        
        Args:
            instId: 产品ID
            pct: 权益比例 (如 0.05 = 5%)
        
        Returns:
            下单结果
        """
        # 解析交易对
        base, quote = instId.split("-")
        
        # 获取可用余额
        available = self.get_available_balance(quote)
        
        if available <= 0:
            self.logger.warning(f"可用余额不足: {quote}")
            return {"code": "-1", "msg": "余额不足"}
        
        # 计算买入金额
        buy_amount = available * pct
        
        # 市价买入
        return self.place_market_buy(instId, buy_amount, amount_type="quote")


# 使用示例
if __name__ == "__main__":
    import os
    
    # 从环境变量读取配置
    api_key = os.getenv("OKX_API_KEY", "")
    secret_key = os.getenv("OKX_SECRET_KEY", "")
    passphrase = os.getenv("OKX_PASSPHRASE", "")
    
    if api_key:
        executor = TradingExecutor(api_key, secret_key, passphrase, simulated=True)
        
        # 获取余额
        balance = executor.get_balance("USDT")
        print("USDT余额:", balance)
        
        # 获取价格
        price = executor.get_current_price("BTC-USDT")
        print("BTC价格:", price)
    else:
        print("请设置环境变量: OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE")