"""验证联邦设备初始化后的案例日志仍能可靠写入。"""

from secureboost_demo.experiment_logging import close_case_logger, configure_case_logger


def test_case_logger_writes_after_secretflow_initialization(
    trained_toy_context,
    tmp_path,
) -> None:
    """日志处理器应在 Ray 初始化后写入 UTF-8 内容并显式完成收尾。"""
    logger, log_path = configure_case_logger("logging_test", tmp_path)
    logger.info("联邦日志测试：设备已初始化。")
    close_case_logger(logger)

    content = log_path.read_text(encoding="utf-8")
    assert "INFO" in content
    assert "设备已初始化" in content
