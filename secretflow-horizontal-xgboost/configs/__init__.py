"""水平联邦 XGBoost 项目的配置入口。"""

from configs.device_config import DEFAULT_DEVICE_CONFIG, DeviceConfig
from configs.training_config import DEFAULT_TRAINING_CONFIG, TrainingConfig

__all__ = [
    "DEFAULT_DEVICE_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DeviceConfig",
    "TrainingConfig",
]

