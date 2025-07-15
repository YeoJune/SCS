#!/bin/bash

# 단일 실험을 실행하고 결과를 정리하는 스크립트
# 사용법: bash scripts/run_experiment.sh <experiment_name> <config_path> [additional_args]

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 인자 확인
if [ $# -lt 2 ]; then
    print_error "사용법: $0 <experiment_name> <config_path> [additional_args]"
    echo "예시: $0 clutrr_run_01 configs/phase2_clutrr.yaml --debug"
    exit 1
fi

EXPERIMENT_NAME=$1
CONFIG_PATH=$2
shift 2  # 처음 두 인자 제거
ADDITIONAL_ARGS="$@"

# 설정 파일 존재 확인
if [ ! -f "$CONFIG_PATH" ]; then
    print_error "설정 파일을 찾을 수 없습니다: $CONFIG_PATH"
    exit 1
fi

# 실험 디렉토리 설정
EXPERIMENT_DIR="experiments/${EXPERIMENT_NAME}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
EXPERIMENT_FULL_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"
EXPERIMENT_FULL_DIR="experiments/${EXPERIMENT_FULL_NAME}"

print_status "실험을 시작합니다: $EXPERIMENT_FULL_NAME"
print_status "설정 파일: $CONFIG_PATH"

# 실험 디렉토리 생성
mkdir -p "$EXPERIMENT_FULL_DIR"
mkdir -p "${EXPERIMENT_FULL_DIR}/logs"
mkdir -p "${EXPERIMENT_FULL_DIR}/checkpoints"
mkdir -p "${EXPERIMENT_FULL_DIR}/figures"
mkdir -p "${EXPERIMENT_FULL_DIR}/analysis"

# 설정 파일 복사 (재현성을 위해)
cp "$CONFIG_PATH" "${EXPERIMENT_FULL_DIR}/config.yaml"

# Git 정보 저장 (재현성을 위해)
if git rev-parse --git-dir > /dev/null 2>&1; then
    git rev-parse HEAD > "${EXPERIMENT_FULL_DIR}/git_commit.txt"
    git diff > "${EXPERIMENT_FULL_DIR}/git_diff.patch"
    git status --porcelain > "${EXPERIMENT_FULL_DIR}/git_status.txt"
fi

# 환경 정보 저장
echo "Python 버전: $(python --version)" > "${EXPERIMENT_FULL_DIR}/environment.txt"
echo "PyTorch 버전: $(python -c 'import torch; print(torch.__version__)')" >> "${EXPERIMENT_FULL_DIR}/environment.txt"
echo "실행 시간: $(date)" >> "${EXPERIMENT_FULL_DIR}/environment.txt"
echo "호스트명: $(hostname)" >> "${EXPERIMENT_FULL_DIR}/environment.txt"
pip freeze > "${EXPERIMENT_FULL_DIR}/requirements_frozen.txt"

# CUDA 정보 (GPU 사용 시)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > "${EXPERIMENT_FULL_DIR}/gpu_info.txt" 2>/dev/null || true
fi

print_status "실험 환경 정보를 저장했습니다."

# 로그 파일 설정
LOG_FILE="${EXPERIMENT_FULL_DIR}/logs/train.log"
ERROR_LOG="${EXPERIMENT_FULL_DIR}/logs/error.log"

# 실험 실행 함수
run_experiment() {
    local mode=$1
    local log_suffix=$2
    
    print_status "${mode} 모드로 실험을 실행합니다..."
    
    # 실행 명령어 구성
    CMD="python run.py --mode ${mode} --config ${EXPERIMENT_FULL_DIR}/config.yaml --experiment_dir ${EXPERIMENT_FULL_DIR} ${ADDITIONAL_ARGS}"
    
    print_status "실행 명령어: $CMD"
    
    # 명령어를 실험 디렉토리에 저장
    echo "$CMD" > "${EXPERIMENT_FULL_DIR}/run_command_${mode}.txt"
    
    # 실행 시작 시간 기록
    echo "$(date): ${mode} 시작" >> "${EXPERIMENT_FULL_DIR}/execution_log.txt"
    
    # 실험 실행 (로그 파일로 출력 저장)
    if eval "$CMD" 2>&1 | tee "${EXPERIMENT_FULL_DIR}/logs/${mode}_${log_suffix}.log"; then
        print_success "${mode} 완료"
        echo "$(date): ${mode} 완료" >> "${EXPERIMENT_FULL_DIR}/execution_log.txt"
        return 0
    else
        print_error "${mode} 실패"
        echo "$(date): ${mode} 실패" >> "${EXPERIMENT_FULL_DIR}/execution_log.txt"
        return 1
    fi
}

# 실험 시작 시간 기록
START_TIME=$(date +%s)
echo "실험 시작: $(date)" > "${EXPERIMENT_FULL_DIR}/execution_log.txt"

# 메인 실험 실행
success=true

# 1. 학습 실행
if ! run_experiment "train" "train"; then
    success=false
fi

# 2. 평가 실행 (학습이 성공한 경우만)
if [ "$success" = true ]; then
    if ! run_experiment "evaluate" "eval"; then
        success=false
    fi
fi

# 3. 분석 실행 (평가가 성공한 경우만)
if [ "$success" = true ]; then
    if ! run_experiment "analyze" "analyze"; then
        print_warning "분석 단계에서 오류가 발생했지만 계속 진행합니다."
    fi
fi

# 실험 종료 시간 기록
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "실험 종료: $(date)" >> "${EXPERIMENT_FULL_DIR}/execution_log.txt"
echo "총 소요 시간: ${DURATION}초 ($(($DURATION / 60))분 $(($DURATION % 60))초)" >> "${EXPERIMENT_FULL_DIR}/execution_log.txt"

# 결과 정리
print_status "실험 결과를 정리합니다..."

# 로그 파일 압축 (용량 절약)
if command -v gzip &> /dev/null; then
    find "${EXPERIMENT_FULL_DIR}/logs" -name "*.log" -size +10M -exec gzip {} \;
fi

# 결과 요약 생성
SUMMARY_FILE="${EXPERIMENT_FULL_DIR}/experiment_summary.txt"
cat > "$SUMMARY_FILE" << EOF
실험 요약: $EXPERIMENT_FULL_NAME
=====================================

실험 설정:
- 실험명: $EXPERIMENT_NAME
- 타임스탬프: $TIMESTAMP
- 설정 파일: $CONFIG_PATH
- 추가 인자: $ADDITIONAL_ARGS

실행 정보:
- 시작 시간: $(head -1 "${EXPERIMENT_FULL_DIR}/execution_log.txt")
- 종료 시간: $(tail -1 "${EXPERIMENT_FULL_DIR}/execution_log.txt")
- 총 소요 시간: ${DURATION}초

실행 결과:
- 학습: $([ "$success" = true ] && echo "성공" || echo "실패")
- 평가: $([ -f "${EXPERIMENT_FULL_DIR}/logs/evaluate_eval.log" ] && echo "완료" || echo "실행되지 않음")
- 분석: $([ -f "${EXPERIMENT_FULL_DIR}/logs/analyze_analyze.log" ] && echo "완료" || echo "실행되지 않음")

파일 구조:
$(find "$EXPERIMENT_FULL_DIR" -type f | head -20)
$([ $(find "$EXPERIMENT_FULL_DIR" -type f | wc -l) -gt 20 ] && echo "... (총 $(find "$EXPERIMENT_FULL_DIR" -type f | wc -l)개 파일)")

EOF

# 최신 실험으로 심볼릭 링크 생성 (편의성을 위해)
LATEST_LINK="experiments/latest_${EXPERIMENT_NAME}"
if [ -L "$LATEST_LINK" ]; then
    rm "$LATEST_LINK"
fi
ln -s "$(basename "$EXPERIMENT_FULL_DIR")" "$LATEST_LINK"

# 실험 결과 요약 출력
print_success "실험이 완료되었습니다!"
echo ""
echo "📁 실험 디렉토리: $EXPERIMENT_FULL_DIR"
echo "📊 결과 요약: $SUMMARY_FILE"
echo "🔗 최신 링크: $LATEST_LINK"
echo ""

if [ "$success" = true ]; then
    print_success "✅ 모든 단계가 성공적으로 완료되었습니다."
    
    # 주요 결과 파일들 확인
    echo "📈 주요 결과 파일들:"
    [ -f "${EXPERIMENT_FULL_DIR}/model_final.pt" ] && echo "  - 최종 모델: model_final.pt"
    [ -f "${EXPERIMENT_FULL_DIR}/results.json" ] && echo "  - 평가 결과: results.json"
    [ -d "${EXPERIMENT_FULL_DIR}/figures" ] && echo "  - 시각화: figures/ 디렉토리"
    [ -d "${EXPERIMENT_FULL_DIR}/analysis" ] && echo "  - 분석 결과: analysis/ 디렉토리"
    
else
    print_error "❌ 실험 중 오류가 발생했습니다."
    echo "🔍 로그 파일을 확인해주세요:"
    echo "  - 전체 로그: ${EXPERIMENT_FULL_DIR}/logs/"
    echo "  - 오류 로그: ${ERROR_LOG} (있는 경우)"
fi

echo ""
echo "🔄 결과 확인 명령어:"
echo "  python run.py --mode analyze --experiment_dir $EXPERIMENT_FULL_DIR"
echo ""
echo "📋 다음 실험 실행:"
echo "  bash scripts/run_experiment.sh <new_experiment_name> <config_path>"

# 성공 여부에 따른 종료 코드 반환
if [ "$success" = true ]; then
    exit 0
else
    exit 1
fi
