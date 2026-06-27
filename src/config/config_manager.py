"""
配置管理模块
支持环境变量和YAML配置文件
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OKXConfig:
    """OKX API配置"""
    api_key: str
    secret_key: str
    passphrase: str
    simulated: bool = False


@dataclass
class TradingConfig:
    """交易配置"""
    mode: str = "simulated"  # simulated / live
    default_instId: str = "BTC-USDT"
    max_position_pct: float = 0.2
    max_daily_loss_pct: float = 0.03
    max_single_loss_pct: float = 0.01


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    file: str = "logs/trader.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class AppConfig:
    """应用配置"""
    okx: OKXConfig
    trading: TradingConfig
    log: LogConfig


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or self._find_config_file()
        self.config: Optional[AppConfig] = None
        
    def _find_config_file(self) -> Optional[str]:
        """查找配置文件"""
        possible_paths = [
            "config/config.yaml",
            "config/config.yml",
            "config.yaml",
            ".env"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load(self) -> AppConfig:
        """
        加载配置
        
        Returns:
            配置对象
        """
        # 优先从环境变量读取
        okx_config = self._load_okx_config()
        trading_config = self._load_trading_config()
        log_config = self._load_log_config()
        
        # 如果有YAML配置文件，覆盖默认值
        if self.config_path and self.config_path.endswith(('.yaml', '.yml')):
            yaml_config = self._load_yaml_config()
            if yaml_config:
                # 合并配置
                if 'okx' in yaml_config:
                    okx_config = OKXConfig(**yaml_config['okx'])
                if 'trading' in yaml_config:
                    trading_config = TradingConfig(**yaml_config['trading'])
                if 'log' in yaml_config:
                    log_config = LogConfig(**yaml_config['log'])
        
        self.config = AppConfig(
            okx=okx_config,
            trading=trading_config,
            log=log_config
        )
        
        return self.config
    
    def _load_okx_config(self) -> OKXConfig:
        """从环境变量加载OKX配置"""
        return OKXConfig(
            api_key=os.getenv("OKX_API_KEY", ""),
            secret_key=os.getenv("OKX_SECRET_KEY", ""),
            passphrase=os.getenv("OKX_PASSPHRASE", ""),
            simulated=os.getenv("TRADING_MODE", "simulated") == "simulated"
        )
    
    def _load_trading_config(self) -> TradingConfig:
        """从环境变量加载交易配置"""
        return TradingConfig(
            mode=os.getenv("TRADING_MODE", "simulated"),
            default_instId=os.getenv("DEFAULT_INSTID", "BTC-USDT"),
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.2")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03")),
            max_single_loss_pct=float(os.getenv("MAX_SINGLE_LOSS_PCT", "0.01"))
        )
    
    def _load_log_config(self) -> LogConfig:
        """从环境变量加载日志配置"""
        return LogConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file=os.getenv("LOG_FILE", "logs/trader.log"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    
    def _load_yaml_config(self) -> Optional[Dict]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return None
    
    def save_yaml_template(self, path: str = "config/config.yaml"):
        """
        保存配置模板
        
        Args:
            path: 保存路径
        """
        template = {
            'okx': {
                'api_key': 'your_api_key_here',
                'secret_key': 'your_secret_key_here',
                'passphrase': 'your_passphrase_here',
                'simulated': True
            },
            'trading': {
                'mode': 'simulated',
                'default_instId': 'BTC-USDT',
                'max_position_pct': 0.2,
                'max_daily_loss_pct': 0.03,
                'max_single_loss_pct': 0.01
            },
            'log': {
                'level': 'INFO',
                'file': 'logs/trader.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        # 创建目录
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False)
        
        print(f"配置模板已保存到: {path}")
    
    def validate(self) -> bool:
        """
        验证配置
        
        Returns:
            是否有效
        """
        if not self.config:
            print("配置未加载")
            return False
        
        # 检查OKX配置
        if not self.config.okx.api_key:
            print("缺少 OKX_API_KEY")
            return False
        
        if not self.config.okx.secret_key:
            print("缺少 OKX_SECRET_KEY")
            return False
        
        if not self.config.okx.passphrase:
            print("缺少 OKX_PASSPHRASE")
            return False
        
        # 检查风控参数
        if self.config.trading.max_position_pct > 0.5:
            print("警告: 单币种最大持仓比例超过50%")
        
        if self.config.trading.max_daily_loss_pct > 0.1:
            print("警告: 最大日亏损比例超过10%")
        
        return True
    
    def get(self) -> AppConfig:
        """获取配置"""
        if not self.config:
            self.load()
        return self.config


# 创建默认配置模板
def create_default_config():
    """创建默认配置文件"""
    manager = ConfigManager()
    
    # 保存配置模板
    manager.save_yaml_template()
    
    # 保存.env示例
    env_template = """# OKX API Configuration
OKX_API_KEY=your_api_key_here
OKX_SECRET_KEY=your_secret_key_here
OKX_PASSPHRASE=your_passphrase_here

# Trading Configuration
TRADING_MODE=simulated
DEFAULT_INSTID=BTC-USDT
MAX_POSITION_PCT=0.2
MAX_DAILY_LOSS_PCT=0.03
MAX_SINGLE_LOSS_PCT=0.01

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trader.log
"""
    
    with open(".env.example", 'w') as f:
        f.write(env_template)
    
    print("默认配置文件已创建")


if __name__ == "__main__":
    create_default_config()