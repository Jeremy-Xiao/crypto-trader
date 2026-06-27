"""
OKX WebSocket 行情订阅模块
实时获取行情、深度、K线等数据
"""

import asyncio
import json
import hmac
import base64
import hashlib
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import websockets


class OKXWebSocket:
    """OKX WebSocket客户端"""
    
    # WebSocket URLs
    PUBLIC_URL = "wss://ws.okx.com:8443/ws/v5/public"
    PRIVATE_URL = "wss://ws.okx.com:8443/ws/v5/private"
    BUSINESS_URL = "wss://ws.okx.com:8443/ws/v5/business"
    
    # 模拟盘URLs
    PUBLIC_URL_SIM = "wss://wspap.okx.com:8443/ws/v5/public"
    PRIVATE_URL_SIM = "wss://wspap.okx.com:8443/ws/v5/private"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        simulated: bool = False,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        """
        初始化WebSocket客户端
        
        Args:
            api_key: API Key（私有频道需要）
            secret_key: Secret Key（私有频道需要）
            passphrase: Passphrase（私有频道需要）
            simulated: 是否使用模拟盘
            on_message: 消息回调函数
            on_error: 错误回调函数
            on_close: 关闭回调函数
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.simulated = simulated
        
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        
        self.ws = None
        self.connected = False
        self.subscriptions = []
        self.running = False
    
    def _generate_signature(self, timestamp: str) -> str:
        """
        生成WebSocket登录签名
        
        Args:
            timestamp: 时间戳
        
        Returns:
            签名
        """
        message = timestamp + 'GET' + '/users/self/verify'
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    async def connect(self, channel_type: str = "public"):
        """
        连接WebSocket
        
        Args:
            channel_type: 频道类型 (public/private/business)
        """
        if self.simulated:
            urls = {
                "public": self.PUBLIC_URL_SIM,
                "private": self.PRIVATE_URL_SIM,
                "business": self.BUSINESS_URL
            }
        else:
            urls = {
                "public": self.PUBLIC_URL,
                "private": self.PRIVATE_URL,
                "business": self.BUSINESS_URL
            }
        
        url = urls.get(channel_type, self.PUBLIC_URL)
        
        try:
            self.ws = await websockets.connect(url)
            self.connected = True
            print(f"[WebSocket] 已连接: {url}")
            
            # 私有频道需要登录
            if channel_type == "private" and self.api_key:
                await self.login()
            
        except Exception as e:
            self.connected = False
            print(f"[WebSocket] 连接失败: {e}")
            if self.on_error:
                self.on_error(e)
    
    async def login(self):
        """登录认证（私有频道）"""
        timestamp = str(int(datetime.now().timestamp()))
        sign = self._generate_signature(timestamp)
        
        login_msg = {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": sign
            }]
        }
        
        await self.ws.send(json.dumps(login_msg))
        response = await self.ws.recv()
        data = json.loads(response)
        
        if data.get("event") == "login" and data.get("code") == "0":
            print("[WebSocket] 登录成功")
        else:
            print(f"[WebSocket] 登录失败: {data}")
    
    async def subscribe(self, channel: str, instId: Optional[str] = None, instType: Optional[str] = None):
        """
        订阅频道
        
        Args:
            channel: 频道名称
            instId: 产品ID
            instType: 产品类型
        """
        args = {"channel": channel}
        if instId:
            args["instId"] = instId
        if instType:
            args["instType"] = instType
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [args]
        }
        
        await self.ws.send(json.dumps(subscribe_msg))
        self.subscriptions.append(args)
        print(f"[WebSocket] 已订阅: {channel} {instId or instType or ''}")
    
    async def unsubscribe(self, channel: str, instId: Optional[str] = None):
        """
        取消订阅
        
        Args:
            channel: 频道名称
            instId: 产品ID
        """
        args = {"channel": channel}
        if instId:
            args["instId"] = instId
        
        unsubscribe_msg = {
            "op": "unsubscribe",
            "args": [args]
        }
        
        await self.ws.send(json.dumps(unsubscribe_msg))
        print(f"[WebSocket] 已取消订阅: {channel}")
    
    async def listen(self):
        """监听消息"""
        self.running = True
        
        while self.running and self.connected:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=30)
                data = json.loads(message)
                
                # 处理pong响应
                if message == "pong":
                    continue
                
                # 回调处理
                if self.on_message:
                    self.on_message(data)
                else:
                    # 默认处理
                    if "data" in data:
                        channel = data.get("arg", {}).get("channel", "unknown")
                        print(f"[WebSocket] {channel}: {data['data'][:1]}...")
                    elif "event" in data:
                        print(f"[WebSocket] 事件: {data}")
                        
            except asyncio.TimeoutError:
                # 发送ping保活
                await self.ws.send("ping")
            except websockets.exceptions.ConnectionClosed:
                print("[WebSocket] 连接关闭")
                self.connected = False
                if self.on_close:
                    self.on_close()
                break
            except Exception as e:
                print(f"[WebSocket] 错误: {e}")
                if self.on_error:
                    self.on_error(e)
    
    async def close(self):
        """关闭连接"""
        self.running = False
        if self.ws:
            await self.ws.close()
        self.connected = False
        print("[WebSocket] 已关闭")
    
    # ==================== 快捷订阅方法 ====================
    
    async def subscribe_ticker(self, instId: str):
        """订阅行情频道"""
        await self.subscribe("tickers", instId=instId)
    
    async def subscribe_books(self, instId: str):
        """订阅深度频道"""
        await self.subscribe("books", instId=instId)
    
    async def subscribe_candles(self, instId: str, bar: str = "candle1H"):
        """订阅K线频道"""
        await self.subscribe(bar, instId=instId)
    
    async def subscribe_trades(self, instId: str):
        """订阅成交频道"""
        await self.subscribe("trades", instId=instId)
    
    async def subscribe_orders(self, instType: str = "ANY"):
        """订阅订单频道（私有）"""
        await self.subscribe("orders", instType=instType)
    
    async def subscribe_account(self):
        """订阅账户频道（私有）"""
        await self.subscribe("account")


class OKXWebSocketManager:
    """WebSocket管理器 - 简化使用"""
    
    def __init__(self, simulated: bool = False):
        """
        初始化管理器
        
        Args:
            simulated: 是否使用模拟盘
        """
        self.simulated = simulated
        self.ws = None
        self.callbacks = {}
        self.data_cache = {}
    
    def on_ticker(self, callback: Callable):
        """设置行情回调"""
        self.callbacks["tickers"] = callback
    
    def on_books(self, callback: Callable):
        """设置深度回调"""
        self.callbacks["books"] = callback
    
    def on_trades(self, callback: Callable):
        """设置成交回调"""
        self.callbacks["trades"] = callback
    
    def _handle_message(self, data: Dict):
        """处理消息"""
        if "arg" in data and "data" in data:
            channel = data["arg"].get("channel")
            
            # 缓存数据
            self.data_cache[channel] = data["data"]
            
            # 触发回调
            if channel in self.callbacks:
                self.callbacks[channel](data["data"])
    
    async def start_public(self, subscriptions: list):
        """
        启动公共频道
        
        Args:
            subscriptions: 订阅列表 [{"channel": "tickers", "instId": "BTC-USDT"}]
        """
        self.ws = OKXWebSocket(
            simulated=self.simulated,
            on_message=self._handle_message
        )
        
        await self.ws.connect("public")
        
        for sub in subscriptions:
            await self.ws.subscribe(
                sub["channel"],
                instId=sub.get("instId"),
                instType=sub.get("instType")
            )
        
        await self.ws.listen()
    
    async def stop(self):
        """停止"""
        if self.ws:
            await self.ws.close()


# 使用示例
async def example_usage():
    """WebSocket使用示例"""
    
    def on_ticker(data):
        """行情回调"""
        if data:
            ticker = data[0]
            print(f"BTC-USDT: {ticker.get('last')} | 买一: {ticker.get('bidPx')} | 卖一: {ticker.get('askPx')}")
    
    manager = OKXWebSocketManager()
    manager.on_ticker(on_ticker)
    
    subscriptions = [
        {"channel": "tickers", "instId": "BTC-USDT"},
        {"channel": "books", "instId": "BTC-USDT"}
    ]
    
    print("启动WebSocket行情订阅...")
    await manager.start_public(subscriptions)


if __name__ == "__main__":
    asyncio.run(example_usage())