"""定义 SecretFlow 本地三方仿真所需的角色配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeviceConfig:
    """保存 Alice、Bob、Charlie 及安全聚合精度配置。"""

    alice_party: str = "alice"
    bob_party: str = "bob"
    charlie_party: str = "charlie"
    fxp_bits: int = 18

    def __post_init__(self) -> None:
        """拒绝重复角色名和不合理的固定点精度。"""
        parties = self.parties
        if len(set(parties)) != len(parties):
            raise ValueError("Alice、Bob、Charlie 的角色名必须互不相同。")
        if any(not party.strip() for party in parties):
            raise ValueError("参与方名称不能为空。")
        if not 8 <= self.fxp_bits <= 32:
            raise ValueError("fxp_bits 必须位于 [8, 32]。")

    @property
    def parties(self) -> tuple[str, str, str]:
        """按初始化顺序返回全部参与方。"""
        return self.alice_party, self.bob_party, self.charlie_party

    def to_dict(self) -> dict[str, object]:
        """转换为可序列化字典。"""
        return asdict(self)


DEFAULT_DEVICE_CONFIG = DeviceConfig()

