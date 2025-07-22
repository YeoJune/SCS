# utils/experiment_utils.py
"""실험 관련 유틸리티"""

import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union, Optional


def get_git_info() -> Dict[str, str]:
    """현재 Git 상태 정보 수집"""
    git_info = {}
    
    try:
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
    """실험 요약 정보 생성"""
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