"""??????? SecretFlow ???????"""

from __future__ import annotations

from dataclasses import dataclass

import secretflow as sf
from secretflow.device import PYU
from secretflow.security.aggregation import SecureAggregator

from configs.device_config import DeviceConfig


@dataclass(frozen=True)
class FederatedDevices:
    """?? Alice?Bob?Charlie ??????????"""

    alice: PYU
    bob: PYU
    charlie: PYU
    aggregator: SecureAggregator


def create_devices(config: DeviceConfig | None = None) -> FederatedDevices:
    """?????????????? Charlie ???????????"""
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
    """?? SecretFlow/Ray ?????????????????"""
    sf.shutdown()
