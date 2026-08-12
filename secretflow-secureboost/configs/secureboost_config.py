"""本地 SecureBoost 仿真中的参与方与 HEU 静态配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceConfig:
    """定义一次两方本地仿真的参与方和密码学角色。"""

    alice_party: str = "alice"
    bob_party: str = "bob"
    label_holder: str = "alice"
    sk_keeper: str = "alice"
    evaluator: str = "bob"
    spu_field_type: str = "FM64"
    key_bit_size: int = 2048

    def __post_init__(self) -> None:
        if self.label_holder != self.alice_party:
            raise ValueError("本地演示中，标签必须由 Alice 持有。")
        if self.sk_keeper != self.alice_party:
            raise ValueError("Alice 必须是 HEU 私钥保管方。")
        if self.evaluator != self.bob_party:
            raise ValueError("Bob 必须是 HEU 密文计算参与方。")
        if self.alice_party == self.bob_party:
            raise ValueError("Alice 和 Bob 必须是两个不同的参与方。")
        if self.key_bit_size < 2048:
            raise ValueError("HEU 密钥长度不得低于 2048 位。")

    @property
    def parties(self) -> tuple[str, str]:
        return (self.alice_party, self.bob_party)

    def build_heu_config(self) -> dict:
        """构造与当前 SecretFlow 版本匹配的 HEU 配置字典。"""
        return {
            "sk_keeper": {"party": self.sk_keeper},
            "evaluators": [{"party": self.evaluator}],
            "mode": "PHEU",
            "encoding": {
                "cleartext_type": "DT_F32",
                "encoder": "FloatEncoder",
            },
            "he_parameters": {
                "schema": "paillier",
                "key_pair": {"generate": {"bit_size": self.key_bit_size}},
            },
        }


DEFAULT_DEVICE_CONFIG = DeviceConfig()
