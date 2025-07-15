# utils/file_utils.py
"""파일 입출력 유틸리티"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    config_path = Path(config_path)
    
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
                base_config = load_config(base_config_path)
                merged_config = _deep_merge_dict(merged_config, base_config)
        
        config = _deep_merge_dict(merged_config, config)
    
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """설정을 YAML 파일로 저장"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def save_json(data: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """JSON 파일 저장"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_data = _make_serializable(data)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


def load_json(load_path: Union[str, Path]) -> Dict[str, Any]:
    """JSON 파일 로드"""
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """딕셔너리 깊은 병합"""
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
            
    return result


def _make_serializable(obj: Any) -> Any:
    """객체를 JSON 직렬화 가능한 형태로 변환"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif hasattr(obj, 'item'):  # PyTorch tensor
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().numpy().tolist()
    else:
        return obj