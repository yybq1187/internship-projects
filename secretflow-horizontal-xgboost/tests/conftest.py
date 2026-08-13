"""提供跨联邦测试复用的 SecretFlow session 级设备夹具。"""

from __future__ import annotations

import pytest

from horizontal_xgb.devices import create_devices, shutdown_devices


@pytest.fixture(scope="session")
def federated_devices():
    """只启动一次本地 Ray，全部联邦测试结束后统一关闭。"""
    devices = create_devices()
    try:
        yield devices
    finally:
        shutdown_devices()
