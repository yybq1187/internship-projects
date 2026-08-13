"""创建和关闭本地 SecureBoost 仿真所需的 SecretFlow 设备。"""

from dataclasses import dataclass
from typing import Any

import secretflow as sf

from configs.secureboost_config import DEFAULT_DEVICE_CONFIG, DeviceConfig


@dataclass(frozen=True)
class SecureBoostDevices:
    """保存一次本地仿真使用的 Alice、Bob 与 HEU 设备引用。"""

    alice: Any
    bob: Any
    heu: Any
    config: DeviceConfig


def create_devices(
    config: DeviceConfig = DEFAULT_DEVICE_CONFIG,
) -> SecureBoostDevices:
    """启动本地仿真，并创建 Alice、Bob 及双方共用的 HEU。

    Alice 同时是标签持有方和 HEU 私钥保管方；Bob 仅提供特征列，
    并作为 HEU 的密文计算参与方。本函数仅用于单机开发仿真，
    不构成跨机器的生产级多方部署。
    """
    sf.init(parties=list(config.parties), address="local")

    alice = sf.PYU(config.alice_party)
    bob = sf.PYU(config.bob_party)
    heu = sf.HEU(
        config.build_heu_config(),
        spu_field_type=config.spu_field_type,
    )
    return SecureBoostDevices(alice=alice, bob=bob, heu=heu, config=config)


def shutdown_devices() -> None:
    """在一次任务完成后关闭当前 SecretFlow 本地仿真。"""
    sf.shutdown()
