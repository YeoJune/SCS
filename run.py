# run.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
로컬 개발 및 빠른 실험을 위한 실행 스크립트.

이 스크립트는 'pip install -e .' 없이도 즉시 실행 및 디버깅이 가능하도록
경로를 설정하고, 실제 로직은 `src/scs/cli.py`에 위임합니다.
"""

import sys
from pathlib import Path

def main():
    """
    프로젝트의 공식 CLI 진입점인 `src/scs/cli.py`의 main 함수를 호출합니다.
    
    이 래퍼 스크립트의 주된 역할은, 패키지가 정식으로 설치되지 않은
    개발 환경에서도 `src` 디렉토리를 파이썬 경로에 추가하여
    모듈 임포트가 원활하게 이루어지도록 하는 것입니다.
    """
    # 1. 경로 설정: 이 파일의 위치를 기준으로 프로젝트 루트를 찾고,
    #    'src' 디렉토리를 파이썬의 모듈 검색 경로에 추가합니다.
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        
    # 2. 실제 로직 위임: 경로 설정이 완료되었으므로,
    #    모든 로직을 담고 있는 공식 CLI 스크립트의 main 함수를 import하여 실행합니다.
    try:
        from src.scs.cli import main as cli_main
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}.")
        print("   'src' 디렉토리 구조가 올바른지 확인해주세요.")
        sys.exit(1)
        
    # 터미널에서 `python run.py ...`로 전달된 인자를 그대로 사용하여
    # 공식 CLI를 실행합니다.
    cli_main()


if __name__ == "__main__":
    main()