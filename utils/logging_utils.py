# utils/logging_utils.py
"""로깅 유틸리티"""

import logging
from pathlib import Path
from typing import Union, Optional


def setup_logging(
    log_dir: Union[str, Path],
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """로깅 시스템 설정"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if format_string is None:
        format_string = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_dir / "main.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)