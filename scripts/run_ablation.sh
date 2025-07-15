#!/bin/bash

# Ablation Study와 하이퍼파라미터 탐색을 위한 다중 실험 자동 실행 스크립트
# 여러 설정 파일을 순차적으로 실행하여 체계적인 실험을 수행

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

# 사용법 출력
show_usage() {
    echo "사용법: $0 [옵션] <실험_타입> [실험_설정]"
    echo ""
    echo "실험 타입:"
    echo "  ablation    - Ablation study 실행"
    echo "  hyperopt    - 하이퍼파라미터 최적화"
    echo "  comparison  - 베이스라인 비교"
    echo "  sensitivity - 민감도 분석"
    echo "  custom      - 커스텀 설정 파일들 실행"
    echo ""
    echo "옵션:"
    echo "  -p, --parallel <num>    병렬 실행 수 (기본: 1)"
    echo "  -r, --resume            중단된 실험부터 재시작"
    echo "  -d, --dry-run           실제 실행 없이 계획만 출력"
    echo "  -h, --help              도움말 출력"
    echo ""
    echo "예시:"
    echo "  $0 ablation phase2_clutrr"
    echo "  $0 hyperopt phase1_logic_ops --parallel 2"
    echo "  $0 custom configs/custom_experiments/"
}

# 기본 설정
PARALLEL_JOBS=1
RESUME=false
DRY_RUN=false
EXPERIMENT_TYPE=""
EXPERIMENT_SETTING=""

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            if [ -z "$EXPERIMENT_TYPE" ]; then
                EXPERIMENT_TYPE="$1"
            elif [ -z "$EXPERIMENT_SETTING" ]; then
                EXPERIMENT_SETTING="$1"
            else
                print_error "알 수 없는 인자: $1"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# 필수 인자 확인
if [ -z "$EXPERIMENT_TYPE" ]; then
    print_error "실험 타입을 지정해주세요."
    show_usage
    exit 1
fi

# 실험 계획 파일 생성
EXPERIMENT_PLAN_DIR="experiments/ablation_plans"
mkdir -p "$EXPERIMENT_PLAN_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
PLAN_FILE="${EXPERIMENT_PLAN_DIR}/${EXPERIMENT_TYPE}_${EXPERIMENT_SETTING}_${TIMESTAMP}.txt"

print_status "다중 실험 계획을 생성합니다: $EXPERIMENT_TYPE"

# 실험 타입별 설정 생성
generate_ablation_configs() {
    local base_config=$1
    local base_name=${EXPERIMENT_SETTING:-"ablation"}
    
    if [ ! -f "$base_config" ]; then
        print_error "기본 설정 파일을 찾을 수 없습니다: $base_config"
        return 1
    fi
    
    print_status "Ablation study 설정을 생성합니다..."
    
    # Ablation 설정 디렉토리 생성
    local ablation_dir="configs/ablation/${base_name}_${TIMESTAMP}"
    mkdir -p "$ablation_dir"
    
    # 1. 모듈별 제거 실험
    local modules=("PFC" "ACC" "IPL" "MTL")
    for module in "${modules[@]}"; do
        local config_name="${ablation_dir}/without_${module}.yaml"
        cp "$base_config" "$config_name"
        
        # YAML 파일에서 해당 모듈 비활성화
        python -c "
import yaml
with open('$config_name', 'r') as f:
    config = yaml.safe_load(f)
config['model']['brain_regions']['$module']['enabled'] = False
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
        echo "${base_name}_without_${module} $config_name" >> "$PLAN_FILE"
    done
    
    # 2. 연결성 제거 실험
    local connections=("internal" "axonal" "multi_scale")
    for conn in "${connections[@]}"; do
        local config_name="${ablation_dir}/without_${conn}.yaml"
        cp "$base_config" "$config_name"
        
        # 연결성 설정 수정
        case $conn in
            "internal")
                python -c "
import yaml
with open('$config_name', 'r') as f:
    config = yaml.safe_load(f)
config['model']['connectivity']['internal']['max_distance'] = 0
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
                ;;
            "axonal")
                python -c "
import yaml
with open('$config_name', 'r') as f:
    config = yaml.safe_load(f)
config['model']['connectivity']['axonal']['connection_probability'] = 0.0
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
                ;;
            "multi_scale")
                python -c "
import yaml
with open('$config_name', 'r') as f:
    config = yaml.safe_load(f)
for scale in ['fine', 'medium', 'coarse']:
    config['model']['connectivity']['axonal']['multi_scale_grids'][scale]['weight'] = 0.0
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
                ;;
        esac
        echo "${base_name}_without_${conn} $config_name" >> "$PLAN_FILE"
    done
    
    # 3. 학습 메커니즘 제거 실험
    local mechanisms=("surrogate_gradient" "k_hop_backprop" "neuromodulation")
    for mech in "${mechanisms[@]}"; do
        local config_name="${ablation_dir}/without_${mech}.yaml"
        cp "$base_config" "$config_name"
        
        # 학습 메커니즘 비활성화
        python -c "
import yaml
with open('$config_name', 'r') as f:
    config = yaml.safe_load(f)
if '$mech' == 'k_hop_backprop':
    config['learning']['k_hop']['max_hops'] = 0
elif '$mech' == 'neuromodulation':
    config['learning']['neuromodulation']['dopamine_sensitivity'] = 0.0
    config['learning']['neuromodulation']['acetylcholine_sensitivity'] = 0.0
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
        echo "${base_name}_without_${mech} $config_name" >> "$PLAN_FILE"
    done
    
    print_success "Ablation 설정 생성 완료: $(wc -l < "$PLAN_FILE")개 실험"
}

generate_hyperopt_configs() {
    local base_config=$1
    local base_name=${EXPERIMENT_SETTING:-"hyperopt"}
    
    print_status "하이퍼파라미터 최적화 설정을 생성합니다..."
    
    local hyperopt_dir="configs/hyperopt/${base_name}_${TIMESTAMP}"
    mkdir -p "$hyperopt_dir"
    
    # 학습률 탐색
    local learning_rates=(1e-2 5e-3 1e-3 5e-4 1e-4)
    for lr in "${learning_rates[@]}"; do
        local config_name="${hyperopt_dir}/lr_${lr}.yaml"
        cp "$base_config" "$config_name"
        
        python -c "
import yaml
with open('$config_name', 'r') as f:
    config = yaml.safe_load(f)
config['training']['optimizer']['lr'] = $lr
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
        echo "${base_name}_lr_${lr} $config_name" >> "$PLAN_FILE"
    done
    
    # 배치 크기 탐색
    local batch_sizes=(8 16 32 64)
    for bs in "${batch_sizes[@]}"; do
        local config_name="${hyperopt_dir}/bs_${bs}.yaml"
        cp "$base_config" "$config_name"
        
        python -c "
import yaml
with open('$config_name', 'r') as f:
    config = yaml.safe_load(f)
config['training']['batch_size'] = $bs
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
        echo "${base_name}_bs_${bs} $config_name" >> "$PLAN_FILE"
    done
    
    # 네트워크 크기 탐색
    local size_factors=(0.5 0.75 1.0 1.25 1.5)
    for factor in "${size_factors[@]}"; do
        local config_name="${hyperopt_dir}/size_${factor}.yaml"
        cp "$base_config" "$config_name"
        
        python -c "
import yaml
with open('$config_name', 'r') as f:
    config = yaml.safe_load(f)
for region in ['PFC', 'ACC', 'IPL', 'MTL']:
    if region in config['model']['brain_regions']:
        neurons = config['model']['brain_regions'][region]['neurons_per_layer']
        config['model']['brain_regions'][region]['neurons_per_layer'] = [int(n * $factor) for n in neurons]
with open('$config_name', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
        echo "${base_name}_size_${factor} $config_name" >> "$PLAN_FILE"
    done
    
    print_success "하이퍼파라미터 설정 생성 완료: $(wc -l < "$PLAN_FILE")개 실험"
}

generate_comparison_configs() {
    local base_name=${EXPERIMENT_SETTING:-"comparison"}
    
    print_status "베이스라인 비교 설정을 생성합니다..."
    
    local comparison_dir="configs/comparison/${base_name}_${TIMESTAMP}"
    mkdir -p "$comparison_dir"
    
    # 각 Phase별 베이스라인 설정들 생성
    local baselines=("transformer_baseline" "snn_baseline" "random_baseline")
    
    for baseline in "${baselines[@]}"; do
        local config_name="${comparison_dir}/${baseline}.yaml"
        
        # 베이스라인별 설정 생성
        case $baseline in
            "transformer_baseline")
                cat > "$config_name" << EOF
# Transformer 베이스라인
defaults:
  - base_model

model:
  name: "TransformerBaseline"
  type: "transformer"
  
  transformer:
    hidden_size: 512
    num_layers: 6
    num_attention_heads: 8
    intermediate_size: 2048
    
training:
  batch_size: 16
  max_epochs: 50
  optimizer:
    name: "AdamW"
    lr: 1e-3
EOF
                ;;
            "snn_baseline")
                cat > "$config_name" << EOF
# 기본 SNN 베이스라인
defaults:
  - base_model

model:
  name: "SNNBaseline"
  type: "feedforward_snn"
  
  # 단순한 피드포워드 SNN
  brain_regions:
    # 단일 모듈만 사용
    SINGLE:
      enabled: true
      neurons_per_layer: [500, 500, 500]
      
  connectivity:
    # 피드포워드 연결만
    internal:
      enabled: false
    axonal:
      connection_probability: 0.0
EOF
                ;;
            "random_baseline")
                cat > "$config_name" << EOF
# 랜덤 베이스라인
defaults:
  - base_model

model:
  name: "RandomBaseline"
  type: "random"
  
training:
  max_epochs: 1  # 학습 없이 랜덤 성능만 측정
EOF
                ;;
        esac
        
        echo "${base_name}_${baseline} $config_name" >> "$PLAN_FILE"
    done
    
    print_success "베이스라인 설정 생성 완료: $(wc -l < "$PLAN_FILE")개 실험"
}

# 실험 타입에 따른 설정 생성
case $EXPERIMENT_TYPE in
    "ablation")
        if [ -z "$EXPERIMENT_SETTING" ]; then
            print_error "Ablation study를 위한 기본 설정을 지정해주세요."
            echo "예시: $0 ablation phase2_clutrr"
            exit 1
        fi
        
        base_config="configs/${EXPERIMENT_SETTING}.yaml"
        if [ ! -f "$base_config" ]; then
            base_config="configs/phase2_clutrr.yaml"  # 기본값
        fi
        
        generate_ablation_configs "$base_config"
        ;;
        
    "hyperopt")
        if [ -z "$EXPERIMENT_SETTING" ]; then
            EXPERIMENT_SETTING="phase1_logic_ops"
        fi
        
        base_config="configs/${EXPERIMENT_SETTING}.yaml"
        generate_hyperopt_configs "$base_config"
        ;;
        
    "comparison")
        generate_comparison_configs
        ;;
        
    "sensitivity")
        print_status "민감도 분석 설정을 생성합니다..."
        # TODO: 파라미터 민감도 분석 구현
        print_warning "민감도 분석은 아직 구현되지 않았습니다."
        exit 1
        ;;
        
    "custom")
        if [ -z "$EXPERIMENT_SETTING" ]; then
            print_error "커스텀 설정 디렉토리를 지정해주세요."
            exit 1
        fi
        
        if [ ! -d "$EXPERIMENT_SETTING" ]; then
            print_error "디렉토리를 찾을 수 없습니다: $EXPERIMENT_SETTING"
            exit 1
        fi
        
        print_status "커스텀 설정에서 실험 계획을 생성합니다..."
        find "$EXPERIMENT_SETTING" -name "*.yaml" | while read config_file; do
            experiment_name=$(basename "$config_file" .yaml)
            echo "custom_${experiment_name} $config_file" >> "$PLAN_FILE"
        done
        ;;
        
    *)
        print_error "알 수 없는 실험 타입: $EXPERIMENT_TYPE"
        show_usage
        exit 1
        ;;
esac

# 실험 계획 출력
total_experiments=$(wc -l < "$PLAN_FILE")
print_status "총 $total_experiments 개의 실험이 계획되었습니다."

if [ "$DRY_RUN" = true ]; then
    print_status "실험 계획 (실제 실행하지 않음):"
    cat "$PLAN_FILE" | nl
    exit 0
fi

# 실험 상태 추적 파일
STATUS_FILE="${EXPERIMENT_PLAN_DIR}/status_${EXPERIMENT_TYPE}_${EXPERIMENT_SETTING}_${TIMESTAMP}.txt"
RESULTS_DIR="experiments/ablation_results/${EXPERIMENT_TYPE}_${EXPERIMENT_SETTING}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# 중단된 실험 재시작 처리
if [ "$RESUME" = true ] && [ -f "$STATUS_FILE" ]; then
    print_status "중단된 실험을 재시작합니다..."
    completed_count=$(grep "COMPLETED" "$STATUS_FILE" | wc -l)
    print_status "이미 완료된 실험: $completed_count/$total_experiments"
else
    # 상태 파일 초기화
    rm -f "$STATUS_FILE"
fi

# 실험 실행 함수
run_single_experiment() {
    local experiment_name=$1
    local config_path=$2
    local experiment_id=$3
    
    # 이미 완료된 실험인지 확인
    if [ -f "$STATUS_FILE" ] && grep -q "$experiment_name COMPLETED" "$STATUS_FILE"; then
        print_status "[$experiment_id/$total_experiments] $experiment_name - 이미 완료됨"
        return 0
    fi
    
    print_status "[$experiment_id/$total_experiments] $experiment_name 실행 중..."
    echo "$experiment_name RUNNING $(date)" >> "$STATUS_FILE"
    
    # 실험 실행
    if bash scripts/run_experiment.sh "$experiment_name" "$config_path" --no-interactive; then
        echo "$experiment_name COMPLETED $(date)" >> "$STATUS_FILE"
        print_success "[$experiment_id/$total_experiments] $experiment_name 완료"
        
        # 결과 요약을 전체 결과 디렉토리로 복사
        if [ -f "experiments/latest_${experiment_name}/experiment_summary.txt" ]; then
            cp "experiments/latest_${experiment_name}/experiment_summary.txt" "${RESULTS_DIR}/${experiment_name}_summary.txt"
        fi
        
        return 0
    else
        echo "$experiment_name FAILED $(date)" >> "$STATUS_FILE"
        print_error "[$experiment_id/$total_experiments] $experiment_name 실패"
        return 1
    fi
}

# 병렬 실행 관리
if [ "$PARALLEL_JOBS" -gt 1 ]; then
    print_status "$PARALLEL_JOBS 개의 병렬 작업으로 실험을 실행합니다..."
    
    # GNU Parallel 사용 (설치되어 있는 경우)
    if command -v parallel &> /dev/null; then
        cat "$PLAN_FILE" | parallel -j "$PARALLEL_JOBS" --colsep ' ' --line-buffer \
            bash scripts/run_experiment.sh {1} {2} --no-interactive
    else
        print_warning "GNU Parallel이 설치되지 않았습니다. 순차 실행으로 전환합니다."
        PARALLEL_JOBS=1
    fi
fi

# 순차 실행
if [ "$PARALLEL_JOBS" -eq 1 ]; then
    print_status "순차적으로 실험을 실행합니다..."
    
    experiment_id=1
    while IFS=' ' read -r experiment_name config_path; do
        run_single_experiment "$experiment_name" "$config_path" "$experiment_id"
        experiment_id=$((experiment_id + 1))
    done < "$PLAN_FILE"
fi

# 최종 결과 요약
print_status "모든 실험이 완료되었습니다. 결과를 요약합니다..."

completed_count=$(grep "COMPLETED" "$STATUS_FILE" 2>/dev/null | wc -l)
failed_count=$(grep "FAILED" "$STATUS_FILE" 2>/dev/null | wc -l)

# 종합 결과 보고서 생성
FINAL_REPORT="${RESULTS_DIR}/final_report.txt"
cat > "$FINAL_REPORT" << EOF
다중 실험 최종 결과 보고서
==========================

실험 정보:
- 실험 타입: $EXPERIMENT_TYPE
- 실험 설정: $EXPERIMENT_SETTING
- 실행 시간: $(date)
- 병렬 작업 수: $PARALLEL_JOBS

실행 결과:
- 총 실험 수: $total_experiments
- 완료된 실험: $completed_count
- 실패한 실험: $failed_count
- 성공률: $(echo "scale=2; $completed_count * 100 / $total_experiments" | bc -l)%

실험 목록:
$(cat "$PLAN_FILE")

실행 상태:
$(cat "$STATUS_FILE" 2>/dev/null)

결과 파일 위치: $RESULTS_DIR
EOF

print_success "🎉 다중 실험 완료!"
print_success "✅ 성공: $completed_count/$total_experiments 실험"
if [ "$failed_count" -gt 0 ]; then
    print_warning "❌ 실패: $failed_count 실험"
fi

echo ""
echo "📁 결과 디렉토리: $RESULTS_DIR"
echo "📊 최종 보고서: $FINAL_REPORT"
echo "📋 실험 상태: $STATUS_FILE"
echo ""
echo "🔍 결과 분석 명령어:"
echo "  python run.py --mode compare --experiments_dir $RESULTS_DIR"

# 성공 여부에 따른 종료 코드
if [ "$failed_count" -eq 0 ]; then
    exit 0
else
    exit 1
fi
