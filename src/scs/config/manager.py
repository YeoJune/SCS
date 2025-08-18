# src/scs/config/manager.py
"""
설정 파일 로딩 및 검증 관리자

YAML 파일을 로드하고 Pydantic 모델로 검증하는 중앙화된 기능 제공
"""

import yaml
from pathlib import Path
from typing import Union, Any, Dict
from pydantic import ValidationError

from .schemas import AppConfig


def _deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """딕셔너리 깊은 병합"""
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
            
    return result


def _load_yaml_with_defaults(config_path: Path) -> Dict[str, Any]:
    """defaults 처리를 포함한 YAML 로딩"""
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # defaults 처리 (다른 설정 파일 상속)
    if 'defaults' in config:
        base_configs = config.pop('defaults')
        if isinstance(base_configs, str):
            base_configs = [base_configs]
            
        merged_config = {}
        for base_config_name in base_configs:
            base_config_path = config_path.parent / f"{base_config_name}.yaml"
            if base_config_path.exists():
                base_config = _load_yaml_with_defaults(base_config_path)
                merged_config = _deep_merge_dict(merged_config, base_config)
        
        config = _deep_merge_dict(merged_config, config)
    
    return config


def load_and_validate_config(config_path: Union[str, Path]) -> AppConfig:
    """
    설정 파일을 로드하고 검증된 AppConfig 객체 반환
    
    Args:
        config_path: YAML 설정 파일 경로
        
    Returns:
        AppConfig: 검증된 설정 객체
        
    Raises:
        FileNotFoundError: 설정 파일이 존재하지 않는 경우
        ValidationError: 설정 구조가 올바르지 않은 경우
        yaml.YAMLError: YAML 파싱 오류
    """
    config_path = Path(config_path)
    
    try:
        # YAML 로딩 (defaults 처리 포함)
        raw_config = _load_yaml_with_defaults(config_path)
        
        # Pydantic 모델로 파싱 및 검증
        app_config = AppConfig(**raw_config)
        
        # 추가 검증 (노드 참조 무결성 등)
        app_config.validate_node_references()
        
        return app_config
        
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 파싱 오류: {e}") from e
    except ValidationError as e:
        # Pydantic 검증 오류를 사용자 친화적으로 변환
        error_details = []
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error['loc'])
            msg = error['msg']
            error_details.append(f"  {loc}: {msg}")
        
        raise ValueError(
            f"설정 파일 검증 실패:\n" + "\n".join(error_details)
        ) from e
    except Exception as e:
        raise ValueError(f"설정 로딩 중 오류 발생: {e}") from e


def validate_config_only(config_path: Union[str, Path]) -> bool:
    """
    설정 파일 검증만 수행 (객체 생성 없이)
    
    Args:
        config_path: YAML 설정 파일 경로
        
    Returns:
        bool: 검증 성공 여부
    """
    try:
        load_and_validate_config(config_path)
        return True
    except Exception:
        return False