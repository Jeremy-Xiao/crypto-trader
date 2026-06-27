# Crypto Trader

加密货币自动化交易系统 - 基于 OKX API v5

## 项目简介

本项目是一个加密货币自动化交易程序，支持：
- 实时行情监控
- 自动化交易策略
- 多交易对支持
- WebSocket 实时数据推送

## 技术栈

- Python 3.11+
- OKX API v5 (REST + WebSocket)
- 数据存储与分析

## 项目结构

```
crypto-trader/
├── src/              # 源代码
│   ├── api/          # API 接口封装
│   ├── strategies/   # 交易策略
│   ├── utils/        # 工具函数
│   └── config/       # 配置管理
├── tests/            # 测试代码
├── docs/             # 文档
├── config/           # 配置文件
├── scripts/          # 运行脚本
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置

1. 创建 OKX API Key
2. 配置环境变量或 config文件

## 快速开始

```bash
python src/main.py
```

## 文档

详细的 OKX API 对接文档位于 `docs/okx-api/` 目录。

## License

MIT