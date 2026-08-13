"""验证本地 Alice、Bob 与 HEU 角色配置的集成测试。"""

import secretflow as sf

from secureboost_demo.devices import create_devices, shutdown_devices


def test_local_role_config_and_heu_ownership() -> None:
    """确认标签、私钥和密文计算角色均符合项目约束。"""
    devices = create_devices()
    try:
        assert devices.config.label_holder == "alice"
        assert devices.heu.sk_keeper_name() == "alice"
        assert list(devices.heu.evaluator_names()) == ["bob"]

        alice_value = devices.alice(lambda: "alice-ready")()
        bob_value = devices.bob(lambda: "bob-ready")()

        assert sf.reveal(alice_value) == "alice-ready"
        assert sf.reveal(bob_value) == "bob-ready"
    finally:
        shutdown_devices()
