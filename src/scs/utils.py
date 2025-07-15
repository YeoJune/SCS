"""
유틸리티 함수들

프로젝트 전반에서 사용되는 보조 기능들을 제공합니다.
"""

import os
import yaml
import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


def setup_logging(
    log_dir: Union[str, Path],
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    로깅 시스템 설정
    
    Args:
        log_dir: 로그 파일을 저장할 디렉토리
        level: 로깅 레벨
        format_string: 로그 포맷 문자열
    """
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


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
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
                merged_config = deep_merge_dict(merged_config, base_config)
        
        # 현재 설정으로 덮어쓰기
        config = deep_merge_dict(merged_config, config)
    
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    설정을 YAML 파일로 저장
    
    Args:
        config: 저장할 설정 딕셔너리
        save_path: 저장 경로
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    딕셔너리 깊은 병합
    
    Args:
        base: 기본 딕셔너리
        update: 업데이트할 딕셔너리
        
    Returns:
        병합된 딕셔너리
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
            
    return result


def set_random_seed(seed: int) -> None:
    """
    재현 가능한 결과를 위한 랜덤 시드 설정
    
    Args:
        seed: 랜덤 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 재현성을 위한 추가 설정
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch가 설치되지 않은 경우


def get_device(device: str = "auto") -> str:
    """
    사용할 연산 장치 결정
    
    Args:
        device: 원하는 장치 ("auto", "cuda", "cpu", "mps")
        
    Returns:
        실제 사용할 장치 문자열
    """
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    else:
        return device


def save_json(data: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    JSON 파일 저장
    
    Args:
        data: 저장할 데이터
        save_path: 저장 경로
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # NumPy 배열과 텐서를 JSON 직렬화 가능한 형태로 변환
    serializable_data = make_serializable(data)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


def load_json(load_path: Union[str, Path]) -> Dict[str, Any]:
    """
    JSON 파일 로드
    
    Args:
        load_path: 로드할 파일 경로
        
    Returns:
        로드된 데이터
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_serializable(obj: Any) -> Any:
    """
    객체를 JSON 직렬화 가능한 형태로 변환
    
    Args:
        obj: 변환할 객체
        
    Returns:
        직렬화 가능한 객체
    """
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
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


def format_time(seconds: float) -> str:
    """
    초를 읽기 쉬운 시간 형식으로 변환
    
    Args:
        seconds: 초 단위 시간
        
    Returns:
        포맷된 시간 문자열
    """
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}분 {secs:.1f}초"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}시간 {int(minutes)}분"


def get_git_info() -> Dict[str, str]:
    """
    현재 Git 상태 정보 수집
    
    Returns:
        Git 정보 딕셔너리
    """
    git_info = {}
    
    try:
        import subprocess
        
        # 현재 커밋 해시
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        git_info['commit_hash'] = result.stdout.strip()
        
        # 브랜치 이름
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        git_info['branch'] = result.stdout.strip()
        
        # 변경사항 여부
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, check=True
        )
        git_info['has_changes'] = bool(result.stdout.strip())
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_info = {'error': 'Git 정보를 가져올 수 없습니다.'}
    
    return git_info


def create_experiment_summary(
    experiment_dir: Union[str, Path],
    config: Dict[str, Any],
    results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    실험 요약 정보 생성
    
    Args:
        experiment_dir: 실험 디렉토리
        config: 설정 딕셔너리
        results: 결과 딕셔너리
        
    Returns:
        실험 요약 딕셔너리
    """
    experiment_dir = Path(experiment_dir)
    
    summary = {
        'experiment_name': experiment_dir.name,
        'created_at': datetime.now().isoformat(),
        'config': {
            'model': config.get('model', {}),
            'task': config.get('task', {}),
            'training': config.get('training', {})
        },
        'git_info': get_git_info(),
        'files': {
            'config': 'config.yaml',
            'logs': list(str(p.relative_to(experiment_dir)) 
                        for p in experiment_dir.glob('logs/*.log')),
            'checkpoints': list(str(p.relative_to(experiment_dir)) 
                              for p in experiment_dir.glob('checkpoints/*.pt'))
        }
    }
    
    if results:
        summary['results'] = results
    
    return summary


class ProgressTracker:
    """진행 상황 추적을 위한 유틸리티 클래스"""
    
    def __init__(self, total_steps: int, description: str = ""):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, step: Optional[int] = None, description: Optional[str] = None):
        """진행 상황 업데이트"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if description is not None:
            self.description = description
            
        self._print_progress()
        
    def _print_progress(self):
        """진행 상황 출력"""
        percent = (self.current_step / self.total_steps) * 100
        elapsed = datetime.now() - self.start_time
        
        if self.current_step > 0:
            eta = elapsed * (self.total_steps - self.current_step) / self.current_step
            eta_str = format_time(eta.total_seconds())
        else:
            eta_str = "알 수 없음"
            
        print(f"\r{self.description} [{self.current_step}/{self.total_steps}] "
              f"{percent:.1f}% | ETA: {eta_str}", end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # 줄바꿈
