"""
日志管理模块
支持文件日志、结构化日志、交易日志
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class TradeLogger:
    """交易日志器 - 记录所有交易事件"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        初始化交易日志器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建不同类型的日志文件
        self.trade_log_file = self.log_dir / "trades.log"
        self.error_log_file = self.log_dir / "errors.log"
        self.system_log_file = self.log_dir / "system.log"
        
        # 设置日志器
        self._setup_loggers()
    
    def _setup_loggers(self):
        """设置各种日志器"""
        # 交易日志
        self.trade_logger = logging.getLogger("trade")
        self.trade_logger.setLevel(logging.INFO)
        trade_handler = logging.FileHandler(self.trade_log_file)
        trade_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s'
        ))
        self.trade_logger.addHandler(trade_handler)
        
        # 错误日志
        self.error_logger = logging.getLogger("error")
        self.error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(self.error_log_file)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        self.error_logger.addHandler(error_handler)
        
        # 系统日志
        self.system_logger = logging.getLogger("system")
        self.system_logger.setLevel(logging.INFO)
        system_handler = logging.FileHandler(self.system_log_file)
        system_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        self.system_logger.addHandler(system_handler)
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        self.system_logger.addHandler(console_handler)
    
    def log_trade(
        self,
        action: str,
        instId: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        status: Optional[str] = None,
        extra: Optional[Dict] = None
    ):
        """
        记录交易
        
        Args:
            action: 动作 (order/cancel/fill)
            instId: 产品ID
            side: 方向 (buy/sell)
            amount: 数量
            price: 价格
            order_id: 订单ID
            status: 状态
            extra: 额外信息
        """
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "instId": instId,
            "side": side,
            "amount": amount,
            "price": price,
            "order_id": order_id,
            "status": status,
            "extra": extra or {}
        }
        
        self.trade_logger.info(json.dumps(trade_record))
    
    def log_order(self, instId: str, side: str, amount: float, price: Optional[float], order_id: str, result: Dict):
        """记录下单"""
        self.log_trade(
            action="order",
            instId=instId,
            side=side,
            amount=amount,
            price=price,
            order_id=order_id,
            status=result.get("code"),
            extra={"response": result}
        )
    
    def log_cancel(self, instId: str, order_id: str, result: Dict):
        """记录撤单"""
        self.log_trade(
            action="cancel",
            instId=instId,
            side="",
            amount=0,
            order_id=order_id,
            status=result.get("code"),
            extra={"response": result}
        )
    
    def log_fill(self, instId: str, side: str, amount: float, price: float, order_id: str):
        """记录成交"""
        self.log_trade(
            action="fill",
            instId=instId,
            side=side,
            amount=amount,
            price=price,
            order_id=order_id,
            status="filled"
        )
    
    def log_error(self, error_type: str, message: str, extra: Optional[Dict] = None):
        """
        记录错误
        
        Args:
            error_type: 错误类型
            message: 错误信息
            extra: 额外信息
        """
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": message,
            "extra": extra or {}
        }
        
        self.error_logger.error(json.dumps(error_record))
    
    def log_api_error(self, api_name: str, error_msg: str, params: Optional[Dict] = None):
        """记录API错误"""
        self.log_error(
            error_type="api_error",
            message=f"{api_name}: {error_msg}",
            extra={"params": params}
        )
    
    def log_system(self, level: str, message: str, extra: Optional[Dict] = None):
        """
        记录系统日志
        
        Args:
            level: 日志级别
            message: 消息
            extra: 额外信息
        """
        log_func = getattr(self.system_logger, level.lower(), self.system_logger.info)
        log_func(message)
        
        if extra:
            self.system_logger.info(json.dumps(extra))


class DailyReport:
    """日报生成器"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        初始化日报生成器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
    
    def generate_daily_summary(self, date: Optional[str] = None) -> Dict:
        """
        生成每日摘要
        
        Args:
            date: 日期 (YYYY-MM-DD)
        
        Returns:
            每日摘要
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        trade_log = self.log_dir / "trades.log"
        
        if not trade_log.exists():
            return {"date": date, "trades": 0, "summary": "无交易记录"}
        
        trades = []
        with open(trade_log, 'r') as f:
            for line in f:
                if date in line:
                    try:
                        trade = json.loads(line.split('|')[1].strip())
                        trades.append(trade)
                    except:
                        pass
        
        # 统计
        orders = len([t for t in trades if t.get("action") == "order"])
        fills = len([t for t in trades if t.get("action") == "fill"])
        cancels = len([t for t in trades if t.get("action") == "cancel"])
        
        return {
            "date": date,
            "total_events": len(trades),
            "orders": orders,
            "fills": fills,
            "cancels": cancels,
            "trades": trades
        }


# 获取默认日志器
def get_logger(name: str = "crypto_trader") -> logging.Logger:
    """
    获取日志器
    
    Args:
        name: 日志器名称
    
    Returns:
        日志器
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 文件handler
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "trader.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)
    
    return logger


# 使用示例
if __name__ == "__main__":
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 创建交易日志器
    trade_logger = TradeLogger()
    
    # 测试记录
    trade_logger.log_order(
        instId="BTC-USDT",
        side="buy",
        amount=0.001,
        price=42000,
        order_id="12345",
        result={"code": "0", "msg": "success"}
    )
    
    trade_logger.log_system("info", "系统启动")
    
    print("日志测试完成")