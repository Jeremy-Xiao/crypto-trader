"""
行情监测框架（market monitor）

提供一套「可插拔」的行情数据源接入机制：
- 每个数据源继承 BaseSource，实现 fetch_raw() + parse()
- 框架负责 HTTP 请求、磁盘缓存、异常兜底、统一归一化
- 聚合器 MarketMonitor 把多源信号合成一个「市场状态」结论

当前已接入：
- fear_greed    : Alternative.me 恐惧贪婪指数（情绪面，免费免认证）
- funding_rate  : 永续合约资金费率（衍生品杠杆情绪，Binance 公共接口免 key）

所有数据源通过 register 装饰器自动注册，新增数据源只需在 sources/ 下
新建一个文件并加 @register 即可，无需改动聚合器。
"""
