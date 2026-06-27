"""
OKX REST API 封装模块
支持行情查询、账户查询、下单等功能
"""

import requests
import hmac
import base64
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from urllib.parse import urlencode


class OKXClient:
    """OKX API Client"""
    
    # API Base URL
    BASE_URL = "https://www.okx.com"
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        simulated: bool = False
    ):
        """
        初始化OKX客户端
        
        Args:
            api_key: API Key
            secret_key: Secret Key
            passphrase: Passphrase
            simulated: 是否使用模拟盘
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.simulated = simulated
        
    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = ""
    ) -> str:
        """
        生成API签名
        
        Args:
            timestamp: ISO格式时间戳
            method: HTTP方法 (GET/POST)
            request_path: 请求路径
            body: 请求体
        
        Returns:
            Base64编码的签名
        """
        if body is None or body == "":
            body = ""
        
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    def _get_headers(
        self,
        method: str,
        request_path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """
        获取请求头
        
        Args:
            method: HTTP方法
            request_path: 请求路径
            body: 请求体
        
        Returns:
            请求头字典
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        sign = self._generate_signature(timestamp, method, request_path, body)
        
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': sign,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        # 模拟盘需要额外header
        if self.simulated:
            headers['x-simulated-trading'] = '1'
        
        return headers
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法 (GET/POST)
            endpoint: API端点
            params: 查询参数
            body: 请求体
        
        Returns:
            API响应
        """
        url = self.BASE_URL + endpoint
        
        # 构建请求路径（用于签名）
        request_path = endpoint
        if params:
            query_string = urlencode(params)
            url += "?" + query_string
            request_path += "?" + query_string
        
        # 请求体
        body_str = json.dumps(body) if body else ""
        
        # 获取headers
        headers = self._get_headers(method, request_path, body_str)
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=headers, data=body_str, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"code": "-1", "msg": str(e), "data": []}


class OKXPublicAPI(OKXClient):
    """OKX 公共API（无需认证的行情接口）"""
    
    def __init__(self):
        """公共API无需认证"""
        super().__init__("", "", "")
    
    def _public_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """公共请求（无需签名）"""
        url = self.BASE_URL + endpoint
        if params:
            url += "?" + urlencode(params)
        
        try:
            response = requests.get(url, timeout=30)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"code": "-1", "msg": str(e), "data": []}
    
    # ==================== 公共数据接口 ====================
    
    def get_server_time(self) -> Dict:
        """获取服务器时间"""
        return self._public_request("/api/v5/public/time")
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return self._public_request("/api/v5/system/status")
    
    # ==================== 行情接口 ====================
    
    def get_tickers(self, instType: str) -> Dict:
        """
        获取所有产品行情信息
        
        Args:
            instType: 产品类型 (SPOT/MARGIN/SWAP/FUTURES/OPTION)
        """
        return self._public_request("/api/v5/market/tickers", {"instType": instType})
    
    def get_ticker(self, instId: str) -> Dict:
        """
        获取单个产品行情信息
        
        Args:
            instId: 产品ID，如 BTC-USDT
        """
        return self._public_request("/api/v5/market/ticker", {"instId": instId})
    
    def get_books(self, instId: str, sz: int = 5) -> Dict:
        """
        获取产品深度
        
        Args:
            instId: 产品ID
            sz: 深度档位数量
        """
        return self._public_request("/api/v5/market/books", {"instId": instId, "sz": str(sz)})
    
    def get_candles(
        self,
        instId: str,
        bar: str = "1H",
        limit: int = 100
    ) -> Dict:
        """
        获取K线数据
        
        Args:
            instId: 产品ID
            bar: K线周期 (1m/5m/15m/30m/1H/4H/1D/1W/1M)
            limit: 返回数量
        """
        return self._public_request("/api/v5/market/candles", {
            "instId": instId,
            "bar": bar,
            "limit": str(limit)
        })
    
    def get_trades(self, instId: str, limit: int = 100) -> Dict:
        """
        获取近期成交
        
        Args:
            instId: 产品ID
            limit: 返回数量
        """
        return self._public_request("/api/v5/market/trades", {
            "instId": instId,
            "limit": str(limit)
        })
    
    # ==================== 产品信息 ====================
    
    def get_instruments(self, instType: str, instId: Optional[str] = None) -> Dict:
        """
        获取交易产品基础信息
        
        Args:
            instType: 产品类型
            instId: 产品ID（可选）
        """
        params = {"instType": instType}
        if instId:
            params["instId"] = instId
        return self._public_request("/api/v5/public/instruments", params)


class OKXPrivateAPI(OKXClient):
    """OKX 私有API（需要认证的交易接口）"""
    
    # ==================== 账户接口 ====================
    
    def get_balance(self, ccy: Optional[str] = None) -> Dict:
        """
        获取账户余额
        
        Args:
            ccy: 币种（可选，不传返回所有）
        """
        params = {}
        if ccy:
            params["ccy"] = ccy
        return self._request("GET", "/api/v5/account/balance", params)
    
    def get_positions(self, instType: Optional[str] = None, instId: Optional[str] = None) -> Dict:
        """
        获取持仓信息
        
        Args:
            instType: 产品类型
            instId: 产品ID
        """
        params = {}
        if instType:
            params["instType"] = instType
        if instId:
            params["instId"] = instId
        return self._request("GET", "/api/v5/account/positions", params)
    
    def get_account_config(self) -> Dict:
        """获取账户配置"""
        return self._request("GET", "/api/v5/account/config")
    
    def set_leverage(self, instId: str, lever: str, mgnMode: str) -> Dict:
        """
        设置杠杆倍数
        
        Args:
            instId: 产品ID
            lever: 杠杆倍数
            mgnMode: 保证金模式 (cross/isolated)
        """
        body = {
            "instId": instId,
            "lever": lever,
            "mgnMode": mgnMode
        }
        return self._request("POST", "/api/v5/account/set-leverage", body=body)
    
    # ==================== 交易接口 ====================
    
    def place_order(
        self,
        instId: str,
        tdMode: str,
        side: str,
        ordType: str,
        sz: str,
        px: Optional[str] = None,
        reduceOnly: bool = False
    ) -> Dict:
        """
        下单
        
        Args:
            instId: 产品ID
            tdMode: 交易模式 (cash/cross/isolated)
            side: 订单方向 (buy/sell)
            ordType: 订单类型 (market/limit/post_only/fok/ioc)
            sz: 委托数量
            px: 委托价格（限价单必填）
            reduceOnly: 是否只减仓
        """
        body = {
            "instId": instId,
            "tdMode": tdMode,
            "side": side,
            "ordType": ordType,
            "sz": sz
        }
        if px:
            body["px"] = px
        if reduceOnly:
            body["reduceOnly"] = True
        
        return self._request("POST", "/api/v5/trade/order", body=body)
    
    def place_market_order(self, instId: str, tdMode: str, side: str, sz: str) -> Dict:
        """
        市价下单
        
        Args:
            instId: 产品ID
            tdMode: 交易模式
            side: 订单方向
            sz: 委托数量
        """
        return self.place_order(instId, tdMode, side, "market", sz)
    
    def place_limit_order(
        self,
        instId: str,
        tdMode: str,
        side: str,
        sz: str,
        px: str
    ) -> Dict:
        """
        限价下单
        
        Args:
            instId: 产品ID
            tdMode: 交易模式
            side: 订单方向
            sz: 委托数量
            px: 委托价格
        """
        return self.place_order(instId, tdMode, side, "limit", sz, px)
    
    def cancel_order(self, instId: str, ordId: Optional[str] = None, clOrdId: Optional[str] = None) -> Dict:
        """
        撤单
        
        Args:
            instId: 产品ID
            ordId: 订单ID
            clOrdId: 客户自定义订单ID
        """
        body = {"instId": instId}
        if ordId:
            body["ordId"] = ordId
        if clOrdId:
            body["clOrdId"] = clOrdId
        
        return self._request("POST", "/api/v5/trade/cancel-order", body=body)
    
    def get_order(self, instId: str, ordId: Optional[str] = None, clOrdId: Optional[str] = None) -> Dict:
        """
        查询订单
        
        Args:
            instId: 产品ID
            ordId: 订单ID
            clOrdId: 客户自定义订单ID
        """
        params = {"instId": instId}
        if ordId:
            params["ordId"] = ordId
        if clOrdId:
            params["clOrdId"] = clOrdId
        
        return self._request("GET", "/api/v5/trade/order", params)
    
    def get_orders_pending(self, instType: Optional[str] = None, limit: int = 100) -> Dict:
        """
        获取未成交订单
        
        Args:
            instType: 产品类型
            limit: 返回数量
        """
        params = {"limit": str(limit)}
        if instType:
            params["instType"] = instType
        
        return self._request("GET", "/api/v5/trade/orders-pending", params)
    
    def get_fills(self, instType: Optional[str] = None, limit: int = 100) -> Dict:
        """
        获取成交明细
        
        Args:
            instType: 产品类型
            limit: 返回数量
        """
        params = {"limit": str(limit)}
        if instType:
            params["instType"] = instType
        
        return self._request("GET", "/api/v5/trade/fills", params)


# 使用示例
if __name__ == "__main__":
    # 公共API示例（无需认证）
    public_api = OKXPublicAPI()
    
    # 获取服务器时间
    time_result = public_api.get_server_time()
    print("服务器时间:", time_result)
    
    # 获取BTC-USDT行情
    ticker = public_api.get_ticker("BTC-USDT")
    print("BTC-USDT行情:", ticker)
    
    # 获取K线数据
    candles = public_api.get_candles("BTC-USDT", "1H", 10)
    print("BTC-USDT K线:", candles)