#!/bin/bash

# SCS 프로젝트 초기 환경 설정 스크립트
# 새로운 환경에서 프로젝트를 시작할 때 필요한 모든 준비를 자동화

set -e  # 에러 발생 시 스크립트 중단

echo "🧠 SCS (Spike-Based Cognitive System) 초기 설정을 시작합니다..."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 진행 상황 출력 함수
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Python 버전 확인
print_status "Python 버전을 확인합니다..."
python_version=$(python --version 2>&1 | cut -d' ' -f2)
required_version="3.8"

if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    print_success "Python $python_version 확인됨 (>=3.8 요구사항 만족)"
else
    print_error "Python 3.8 이상이 필요합니다. 현재 버전: $python_version"
    exit 1
fi

# 2. 가상환경 생성 및 활성화
print_status "Python 가상환경을 설정합니다..."

# 가상환경이 이미 존재하는지 확인
if [ ! -d "venv" ]; then
    print_status "새로운 가상환경을 생성합니다..."
    python -m venv venv
    print_success "가상환경이 생성되었습니다."
else
    print_warning "가상환경이 이미 존재합니다."
fi

# 운영체제에 따른 활성화 명령어 선택
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi

print_success "가상환경이 활성화되었습니다."

# 3. pip 업그레이드
print_status "pip을 최신 버전으로 업그레이드합니다..."
pip install --upgrade pip
print_success "pip 업그레이드 완료"

# 4. 프로젝트 의존성 설치
print_status "프로젝트 의존성을 설치합니다..."

# editable install로 프로젝트 설치
pip install -e .

# 개발 의존성 설치
print_status "개발 도구를 설치합니다..."
pip install -e ".[dev]"

print_success "모든 의존성 설치 완료"

# 5. 필요한 디렉토리 생성
print_status "프로젝트 디렉토리 구조를 생성합니다..."

# experiments 하위 디렉토리들
mkdir -p experiments/phase1_logic_ops
mkdir -p experiments/phase2_clutrr  
mkdir -p experiments/phase3_gsm8k

# 데이터 디렉토리
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/cache

# 로그 디렉토리
mkdir -p logs/tensorboard
mkdir -p logs/wandb

# 모델 체크포인트 디렉토리
mkdir -p checkpoints

print_success "디렉토리 구조 생성 완료"

# 6. Git 설정 확인
print_status "Git 설정을 확인합니다..."

if git rev-parse --git-dir > /dev/null 2>&1; then
    print_success "Git 저장소가 이미 초기화되어 있습니다."
else
    print_status "Git 저장소를 초기화합니다..."
    git init
    
    # .gitignore가 있는지 확인하고 첫 커밋
    if [ -f ".gitignore" ]; then
        git add .gitignore README.md pyproject.toml
        git commit -m "Initial commit: Project structure setup"
        print_success "초기 커밋 완료"
    fi
fi

# 7. 개발 도구 설정
print_status "개발 도구를 설정합니다..."

# pre-commit hooks 설치 (선택사항)
if command -v pre-commit &> /dev/null; then
    print_status "pre-commit hooks을 설치합니다..."
    pre-commit install
    print_success "pre-commit hooks 설치 완료"
else
    print_warning "pre-commit이 설치되지 않았습니다. 코드 품질 도구를 수동으로 실행해야 합니다."
fi

# 8. 데이터셋 다운로드 (선택사항)
read -p "데이터셋을 지금 다운로드하시겠습니까? (y/N): " download_data

if [[ $download_data =~ ^[Yy]$ ]]; then
    print_status "데이터셋을 다운로드합니다..."
    
    # CLUTRR 데이터셋 다운로드
    print_status "CLUTRR 데이터셋을 다운로드합니다..."
    if command -v wget &> /dev/null; then
        wget -P data/raw/ "https://github.com/facebookresearch/clutrr/archive/master.zip"
        unzip data/raw/master.zip -d data/raw/
        mv data/raw/clutrr-master data/raw/clutrr
        rm data/raw/master.zip
    else
        print_warning "wget이 설치되지 않았습니다. 데이터셋을 수동으로 다운로드해주세요."
    fi
    
    # GSM8K 데이터셋 다운로드
    print_status "GSM8K 데이터셋을 다운로드합니다..."
    python -c "
from datasets import load_dataset
import os

# GSM8K 데이터셋 로드 및 저장
dataset = load_dataset('gsm8k', 'main')
dataset.save_to_disk('data/raw/gsm8k')
print('GSM8K 데이터셋 다운로드 완료')
" 2>/dev/null || print_warning "GSM8K 데이터셋 다운로드에 실패했습니다. 나중에 수동으로 다운로드해주세요."

    print_success "데이터셋 다운로드 완료"
else
    print_status "데이터셋 다운로드를 건너뜁니다."
fi

# 9. 환경 변수 설정 파일 생성
print_status "환경 설정 파일을 생성합니다..."

cat > .env << EOF
# SCS 프로젝트 환경 변수
SCS_PROJECT_ROOT=$(pwd)
SCS_DATA_DIR=\${SCS_PROJECT_ROOT}/data
SCS_EXPERIMENTS_DIR=\${SCS_PROJECT_ROOT}/experiments
SCS_LOGS_DIR=\${SCS_PROJECT_ROOT}/logs

# 학습 설정
CUDA_VISIBLE_DEVICES=0
WANDB_PROJECT=SCS_Development
WANDB_ENTITY=your_team

# 개발 설정
PYTHONPATH=\${SCS_PROJECT_ROOT}/src:\$PYTHONPATH
EOF

print_success "환경 설정 파일 (.env) 생성 완료"

# 10. 설치 검증
print_status "설치를 검증합니다..."

# Python 패키지 import 테스트
python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')

try:
    import transformers
    print(f'Transformers 버전: {transformers.__version__}')
except ImportError:
    print('Warning: transformers 설치 확인 필요')

try:
    import yaml
    print('YAML 파서 설치 확인')
except ImportError:
    print('Warning: PyYAML 설치 확인 필요')
    
print('기본 의존성 확인 완료')
" 2>/dev/null || print_warning "일부 패키지 import에 실패했습니다."

# 11. 완료 메시지 및 다음 단계 안내
print_success "🎉 SCS 프로젝트 초기 설정이 완료되었습니다!"

echo ""
echo "📋 다음 단계:"
echo "1. 가상환경 활성화: source venv/bin/activate (Linux/macOS) 또는 venv\\Scripts\\activate (Windows)"
echo "2. 환경 변수 로드: source .env"
echo "3. Phase 1 실험 실행: python run.py --mode train --config configs/phase1_logic_ops.yaml"
echo "4. 또는 스크립트 사용: bash scripts/run_experiment.sh phase1_logic_ops configs/phase1_logic_ops.yaml"
echo ""

echo "📚 추가 정보:"
echo "- README.md: 프로젝트 개요 및 사용법"
echo "- docs/proposal.md: 연구 제안서"
echo "- docs/architecture_spec.md: 기술 명세서"
echo ""

echo "🔧 개발 도구:"
echo "- 코드 포매팅: black src/"
echo "- 타입 체킹: mypy src/"
echo "- 테스트 실행: pytest"
echo ""

print_success "Happy coding! 🚀"
