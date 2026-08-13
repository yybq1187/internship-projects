"""集中创建和关闭 SecretFlow 本地三方设备。"""

from __future__ import annotations

from dataclasses import dataclass

import secretflow as sf
from secretflow.device import PYU
from secretflow.security.aggregation import SecureAggregator

from configs.device_config import DeviceConfig


@dataclass(frozen=True)
class FederatedDevices:
    """保存 Alice、Bob、Charlie 及固定点安全聚合器。"""

    alice: PYU
    bob: PYU
    charlie: PYU
    aggregator: SecureAggregator


def create_devices(config: DeviceConfig | None = None) -> FederatedDevices:
    """以本地仿真模式启动三方，并让 Charlie 只承担聚合与决策职责。"""
    actual = config or DeviceConfig()
    sf.init(parties=list(actual.parties), address="local")
    alice = sf.PYU(actual.alice_party)
    bob = sf.PYU(actual.bob_party)
    charlie = sf.PYU(actual.charlie_party)
    aggregator = SecureAggregator(
        device=charlie,
        participants=[alice, bob],
        fxp_bits=actual.fxp_bits,
    )
    return FederatedDevices(alice, bob, charlie, aggregator)


def shutdown_devices() -> None:
    """关闭 SecretFlow/Ray 本地运行时，避免测试之间残留进程。"""
    sf.shutdown()
