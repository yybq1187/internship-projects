"""为可复现实验案例创建统一的 UTF-8 文件日志。"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_case_logger(
    case_name: str,
    output_directory: str | Path,
) -> tuple[logging.Logger, Path]:
    """创建案例专用日志记录器，并返回记录器及对应日志文件路径。"""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / f"{case_name}.log"
    logger = logging.getLogger(f"secureboost_demo.{case_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 同一解释器内重复运行案例时先关闭旧处理器，避免日志内容重复写入。
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger, log_path


def close_case_logger(logger: logging.Logger) -> None:
    """刷新并关闭案例日志处理器，确保内容在进程退出前写入磁盘。

    SecretFlow/Ray 初始化会重置部分 Python 日志资源。因此，案例应在
    设备初始化后创建记录器，并在关闭设备前调用本函数完成日志收尾。
    """
    for handler in logger.handlers[:]:
        handler.flush()
        logger.removeHandler(handler)
        handler.close()
