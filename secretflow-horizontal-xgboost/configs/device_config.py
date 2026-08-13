"""?? SecretFlow ??????????????"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeviceConfig:
    """?? Alice?Bob?Charlie ??????????"""

    alice_party: str = "alice"
    bob_party: str = "bob"
    charlie_party: str = "charlie"
    fxp_bits: int = 18

    def __post_init__(self) -> None:
        """??????????????????"""
        parties = self.parties
        if len(set(parties)) != len(parties):
            raise ValueError("Alice?Bob?Charlie ???????????")
        if any(not party.strip() for party in parties):
            raise ValueError("??????????")
        if not 8 <= self.fxp_bits <= 32:
            raise ValueError("fxp_bits ???? [8, 32]?")

    @property
    def parties(self) -> tuple[str, str, str]:
        """??????????????"""
        return self.alice_party, self.bob_party, self.charlie_party

    def to_dict(self) -> dict[str, object]:
        """??????????"""
        return asdict(self)


DEFAULT_DEVICE_CONFIG = DeviceConfig()

