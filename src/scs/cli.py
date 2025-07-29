# src/scs/cli.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCS (Spike-Based Cognitive System) 공식 CLI 실행 진입점
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import torch

# --- 프로젝트 모듈 Import ---
# 이 파일은 패키지 내부에 있으므로, 상대/절대 경로로 모듈을 import 합니다.
try:
    from scs.architecture import SCSSystem
    from scs.training import SCSTrainer, TrainingConfig, MultiObjectiveLoss, OptimizerFactory
    from scs.data import create_dataloader, SCSTokenizer
    from scs.utils import (
        setup_logging, load_config, save_config, set_random_seed,
        get_device, ModelBuilder
    )
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}. 패키지가 올바르게 설치되었는지 확인해주세요. ('pip install -e .')")
    sys.exit(1)


# --- 명령행 인자 및 유효성 검사 ---
def setup_args() -> argparse.ArgumentParser:
    """CLI 인자 설정"""
    parser = argparse.ArgumentParser(
        description="SCS 실행 스크립트 (선언적 조립 구조 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 학습 모드
  scs --mode train --config configs/phase2_logiqa_small.yaml
  
  # 평가 모드  
  scs --mode evaluate --experiment_dir experiments/phase2_20241201_1430
  
  # 설정 파일 검증
  scs --mode validate --config configs/my_experiment.yaml
        """
    )
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["train", "evaluate", "validate"], 
                       help="실행 모드")
    parser.add_argument("--config", type=str, 
                       help="설정 파일 경로 (train/validate 모드 필수)")
    parser.add_argument("--experiment_dir", type=str, 
                       help="실험 디렉토리 경로 (evaluate 모드 필수)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="연산 장치 선택 (cuda, cpu, mps)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="재현성을 위한 랜덤 시드")
    parser.add_argument("--debug", action="store_true", 
                       help="디버그 모드 (상세 로깅)")
    return parser

def validate_args(args: argparse.Namespace):
    """CLI 인자 유효성 검사"""
    if args.mode in ["train", "validate"] and not args.config:
        raise ValueError(f"{args.mode} 모드에서는 --config 인자가 필수입니다.")
    if args.mode == "evaluate" and not args.experiment_dir:
        raise ValueError("evaluate 모드에서는 --experiment_dir 인자가 필수입니다.")
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {args.config}")
    if args.experiment_dir and not Path(args.experiment_dir).exists():
        raise FileNotFoundError(f"실험 디렉토리를 찾을 수 없습니다: {args.experiment_dir}")


# --- Conv2d 출력 차원 계산 함수 ---
def calculate_conv2d_output_size(input_size: int, kernel_size: int, stride: int = 1, 
                                padding: int = 0, dilation: int = 1) -> int:
    """Conv2d 출력 크기 계산"""
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def validate_axonal_connections(config: Dict[str, Any]) -> List[str]:
    """축삭 연결의 차원 호환성 검증"""
    errors = []
    
    if "brain_regions" not in config or "axonal_connections" not in config:
        return errors
    
    brain_regions = config["brain_regions"]
    connections = config["axonal_connections"].get("connections", [])
    
    # 각 타겟 노드별로 연결들을 그룹화
    target_connections = {}
    for conn in connections:
        source = conn.get("source")
        target = conn.get("target")
        
        if not source or not target:
            continue
            
        if target not in target_connections:
            target_connections[target] = []
        target_connections[target].append(conn)
    
    # 각 타겟 노드에 대해 차원 검증
    for target, target_conns in target_connections.items():
        if target not in brain_regions:
            continue
            
        target_grid_size = brain_regions[target].get("grid_size")
        if not target_grid_size or len(target_grid_size) != 2:
            continue
            
        target_h, target_w = target_grid_size
        
        for conn in target_conns:
            source = conn["source"]
            if source not in brain_regions:
                continue
                
            source_grid_size = brain_regions[source].get("grid_size")
            if not source_grid_size or len(source_grid_size) != 2:
                continue
                
            source_h, source_w = source_grid_size
            
            # Conv2d 파라미터 추출
            kernel_size = conn.get("kernel_size", 1)
            stride = conn.get("stride", 1)
            padding = conn.get("padding", 0)
            dilation = conn.get("dilation", 1)
            
            # 출력 크기 계산
            try:
                output_h = calculate_conv2d_output_size(source_h, kernel_size, stride, padding, dilation)
                output_w = calculate_conv2d_output_size(source_w, kernel_size, stride, padding, dilation)
                
                # 타겟 크기와 비교
                if output_h != target_h or output_w != target_w:
                    errors.append(
                        f"축삥 연결 차원 불일치: {source}→{target}\n"
                        f"   소스 크기: {source_grid_size}, 타겟 크기: {target_grid_size}\n"
                        f"   Conv2d 파라미터: kernel_size={kernel_size}, stride={stride}, padding={padding}, dilation={dilation}\n"
                        f"   계산된 출력 크기: [{output_h}, {output_w}] (예상: [{target_h}, {target_w}])\n"
                        f"   해결책: padding을 조정하거나 다른 Conv2d 파라미터를 수정하세요."
                    )
                    
            except Exception as calc_error:
                errors.append(
                    f"축삥 연결 {source}→{target}의 차원 계산 중 오류: {calc_error}"
                )
    
    return errors


# --- 설정 파일 검증 모드 ---
def validate_mode(args: argparse.Namespace):
    """설정 파일 구조 검증 모드"""
    print("🔍 설정 파일 구조 검증을 시작합니다...")
    
    try:
        # 설정 파일 로드
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config = load_config(config_path)
        
        # ModelBuilder를 통한 기본 구조 검증
        validation_errors = ModelBuilder.validate_config_structure(config)
        
        # 축삥 연결 차원 검증 추가
        dimension_errors = validate_axonal_connections(config)
        
        all_errors = validation_errors + dimension_errors
        
        if not all_errors:
            print("✅ 설정 파일 구조가 올바릅니다!")
            
            # 간단한 모델 생성 테스트 (실제 디바이스 사용 안함)
            try:
                model = ModelBuilder.build_scs_from_config(config, device="cpu")
                total_params = sum(p.numel() for p in model.parameters())
                print(f"📊 모델 정보:")
                print(f"   - 총 매개변수: {total_params:,}")
                print(f"   - 뇌 영역 수: {len(config['brain_regions'])}")
                print(f"   - 축삭 연결 수: {len(config['axonal_connections']['connections'])}")
                print(f"   - 입력 노드: {config['system_roles']['input_node']}")
                print(f"   - 출력 노드: {config['system_roles']['output_node']}")
                
                # 축별 연결 차원 정보 출력
                print(f"📐 축삭 연결 차원 검증:")
                for conn in config['axonal_connections']['connections']:
                    source = conn['source']
                    target = conn['target']
                    source_size = config['brain_regions'][source]['grid_size']
                    target_size = config['brain_regions'][target]['grid_size']
                    
                    kernel_size = conn.get('kernel_size', 1)
                    stride = conn.get('stride', 1)
                    padding = conn.get('padding', 0)
                    dilation = conn.get('dilation', 1)
                    
                    output_h = calculate_conv2d_output_size(source_size[0], kernel_size, stride, padding, dilation)
                    output_w = calculate_conv2d_output_size(source_size[1], kernel_size, stride, padding, dilation)
                    
                    print(f"   - {source}→{target}: {source_size} → [{output_h}, {output_w}] (타겟: {target_size})")
                
                print("✅ 모델 생성 및 차원 검증 테스트 성공!")
            except Exception as model_error:
                print(f"⚠️  모델 생성 테스트 실패: {model_error}")
                return False
                
        else:
            print("❌ 설정 파일에서 다음 오류들이 발견되었습니다:")
            for i, error in enumerate(all_errors, 1):
                print(f"   {i}. {error}")
            
            # 차원 오류가 있을 경우 도움말 제공
            if dimension_errors:
                print("\n💡 축삭 연결 차원 문제 해결 가이드:")
                print("   1. Conv2d 출력 크기 공식: floor((Input + 2*Padding - Dilation*(Kernel-1) - 1) / Stride + 1)")
                print("   2. 동일한 크기 유지: stride=1일 때 padding = (kernel_size-1)/2")
                print("   3. 크기 축소: stride > 1 또는 padding을 줄이세요")
                print("   4. 온라인 계산기: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html")
            
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 설정 파일 검증 중 오류 발생: {e}")
        return False


# --- 데이터셋 이름 추출 헬퍼 ---
def get_dataset_name_from_config(config: Dict[str, Any], logger) -> str:
    """설정 파일에서 데이터셋 이름 추출 (다양한 형식 지원)"""
    dataset_name = None
    
    # 우선순위에 따라 데이터셋 이름 탐색
    if "task" in config and "dataset_name" in config["task"]:
        dataset_name = config["task"]["dataset_name"]
    elif "data" in config and "dataset_name" in config["data"]:
        dataset_name = config["data"]["dataset_name"]  
    elif "dataset_name" in config:
        dataset_name = config["dataset_name"]
    else:
        # 기본값 사용
        dataset_name = "datatune/LogiQA2.0"
        logger.warning(f"dataset_name not found in config, using default: {dataset_name}")
    
    return dataset_name


# --- 학습 설정 추출 및 정규화 헬퍼 ---
def extract_and_normalize_training_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """설정에서 학습 파라미터 추출 및 정규화"""
    # config 매핑 - base_model.yaml은 "learning", phase2는 "training" 사용
    raw_config = config.get("learning", config.get("training", {})).copy()
    
    # 파라미터 이름 정규화
    param_mapping = {
        "base_learning_rate": "learning_rate",
        "max_grad_norm": "gradient_clip_norm", 
        "eval_every_n_epochs": "eval_every",
        "save_every_n_epochs": "save_every"
    }
    
    for old_name, new_name in param_mapping.items():
        if old_name in raw_config:
            raw_config[new_name] = raw_config.pop(old_name)
    
    # TrainingConfig가 허용하는 파라미터만 필터링
    valid_params = {
        "epochs", "learning_rate", "weight_decay", "gradient_clip_norm",
        "eval_every", "save_every", "early_stopping_patience", "max_clk_training"
    }
    filtered_config = {k: v for k, v in raw_config.items() if k in valid_params}
    
    # 타입 변환 - YAML에서 문자열로 로드된 숫자 값들을 적절한 타입으로 변환
    float_params = ["learning_rate", "weight_decay", "gradient_clip_norm"]
    int_params = ["epochs", "eval_every", "save_every", "early_stopping_patience", "max_clk_training"]
    
    for param in float_params:
        if param in filtered_config:
            filtered_config[param] = float(filtered_config[param])
    
    for param in int_params:
        if param in filtered_config:
            filtered_config[param] = int(filtered_config[param])
    
    return filtered_config, raw_config


# --- 모드별 실행 함수 ---
def train_mode(args: argparse.Namespace, config: Dict[str, Any]):
    """학습 모드 실행 (새로운 선언적 조립 구조 지원)"""
    # 1. 실험 환경 설정
    experiment_name = f"{Path(args.config).stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir / "config.yaml")
    setup_logging(log_dir=experiment_dir / "logs", level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    set_random_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"🚀 실험 '{experiment_name}' 시작 | 디바이스: {device}")

    try:
        # 2. 설정 파일 사전 검증 (차원 검증 포함)
        logger.info("📋 설정 파일 구조 및 차원 검증 중...")
        validation_errors = ModelBuilder.validate_config_structure(config)
        dimension_errors = validate_axonal_connections(config)
        all_errors = validation_errors + dimension_errors
        
        if all_errors:
            logger.error("❌ 설정 파일 검증 실패:")
            for error in all_errors:
                logger.error(f"   - {error}")
            raise ValueError("설정 파일에 오류가 있습니다. 위 메시지를 확인해주세요.")
        logger.info("✅ 설정 파일 구조 및 차원 검증 완료")

        # 3. 데이터 로더 생성
        logger.info("📊 데이터 로더 생성 중...")
        tokenizer = SCSTokenizer(config["data_loading"]["tokenizer"]["name"])
        dataset_name = get_dataset_name_from_config(config, logger)
        
        task_config = config.get("task", {})
        max_samples_config = task_config.get("max_samples", {})

        train_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="train", 
            batch_size=config["data_loading"]["batch_size"], 
            max_length=config["data_loading"]["tokenizer"]["max_length"], 
            tokenizer=tokenizer,
            max_samples=max_samples_config.get("train", None)
        )

        val_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="validation", 
            batch_size=1, 
            max_length=config["data_loading"]["tokenizer"]["max_length"], 
            tokenizer=tokenizer,
            max_samples=max_samples_config.get("validation", None)
        )
        logger.info(f"✅ 데이터 로더 생성 완료 (데이터셋: {dataset_name})")

        # 4. 모델 인스턴스화 (새로운 선언적 조립 방식)
        logger.info("🧠 SCS 모델 생성 중...")
        config["io_system"]["input_interface"]["vocab_size"] = tokenizer.vocab_size
        config["io_system"]["output_interface"]["vocab_size"] = tokenizer.vocab_size
        model = ModelBuilder.build_scs_from_config(config, device=device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ 모델 생성 완료")
        logger.info(f"   - 총 매개변수: {total_params:,}")
        logger.info(f"   - 뇌 영역: {list(config['brain_regions'].keys())}")
        logger.info(f"   - 입력→출력: {config['system_roles']['input_node']} → {config['system_roles']['output_node']}")

        # 5. 학습 시스템 구성
        logger.info("⚙️ 학습 시스템 구성 중...")
        pad_token_id = tokenizer.tokenizer.pad_token_id
        
        filtered_config, raw_config = extract_and_normalize_training_config(config)
        
        training_config = TrainingConfig(pad_token_id=pad_token_id, device=device, **filtered_config)
        loss_fn = MultiObjectiveLoss(pad_token_id=pad_token_id)
        
        # 옵티마이저 생성
        optimizer_type = raw_config.get("optimizer", "adamw").lower()
        optimizer = OptimizerFactory.create(optimizer_type=optimizer_type, model=model, config=training_config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.epochs)
        
        logger.info(f"✅ 학습 시스템 구성 완료 (옵티마이저: {optimizer_type})")

        # 6. 트레이너 생성 및 학습
        logger.info("🎯 학습 시작...")
        trainer = SCSTrainer(
            model=model, 
            config=training_config, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            tokenizer=tokenizer
        )
        trainer.train(train_loader, val_loader, save_path=str(experiment_dir / "checkpoints"))

        # 7. 최종 평가
        logger.info("📈 최종 평가 시작...")
        test_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="test", 
            batch_size=1, 
            max_length=config["data_loading"]["tokenizer"]["max_length"], 
            tokenizer=tokenizer,
            max_samples=max_samples_config.get("test", None)
        )
        test_results = trainer.evaluate(test_loader)
        save_config(test_results, experiment_dir / "results.yaml")
        
        logger.info("🎉 학습 및 평가가 성공적으로 완료되었습니다!")
        logger.info(f"📂 결과 저장 위치: {experiment_dir}")

    except Exception as e:
        logger.error(f"❌ 학습 중 치명적인 오류 발생: {e}", exc_info=True)
        raise

def evaluate_mode(args: argparse.Namespace):
    """평가 모드 실행 (새로운 선언적 조립 구조 지원)"""
    # 1. 환경 설정
    experiment_dir = Path(args.experiment_dir)
    config_path = experiment_dir / "config.yaml"
    checkpoint_path = experiment_dir / "checkpoints" / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")

    config = load_config(config_path)
    setup_logging(log_dir=experiment_dir / "logs" / "eval")
    logger = logging.getLogger(__name__)
    device = get_device(args.device)
    logger.info(f"📊 평가 모드 시작 | 디바이스: {device}")
    
    try:
        # 2. 설정 파일 검증 (저장된 실험의 무결성 확인)
        logger.info("📋 저장된 설정 파일 검증 중...")
        validation_errors = ModelBuilder.validate_config_structure(config)
        dimension_errors = validate_axonal_connections(config)
        all_errors = validation_errors + dimension_errors
        
        if all_errors:
            logger.warning("⚠️ 저장된 설정 파일에 일부 문제가 있지만 평가를 계속합니다:")
            for error in all_errors[:3]:  # 처음 3개만 표시
                logger.warning(f"   - {error}")

        # 3. 데이터 및 모델 로드
        logger.info("📊 데이터 로더 생성 중...")
        tokenizer = SCSTokenizer(config["data_loading"]["tokenizer"]["name"])
        dataset_name = get_dataset_name_from_config(config, logger)
        
        test_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="test", 
            batch_size=1, 
            max_length=config["data_loading"]["tokenizer"]["max_length"], 
            tokenizer=tokenizer
        )
        
        logger.info("🧠 모델 복원 중...")
        config["io_system"]["input_interface"]["vocab_size"] = tokenizer.vocab_size
        config["io_system"]["output_interface"]["vocab_size"] = tokenizer.vocab_size
        model = ModelBuilder.build_scs_from_config(config, device=device)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ 모델 복원 완료")

        # 4. 트레이너 생성 및 평가
        logger.info("📈 평가 실행 중...")
        pad_token_id = tokenizer.tokenizer.pad_token_id
        
        filtered_config, _ = extract_and_normalize_training_config(config)
        
        training_config = TrainingConfig(pad_token_id=pad_token_id, device=device, **filtered_config)
        trainer = SCSTrainer(model=model, config=training_config, tokenizer=tokenizer)
        results = trainer.evaluate(test_loader)
        
        # 결과 저장
        results_path = experiment_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.yaml"
        save_config(results, results_path)
        
        logger.info("🎉 평가가 성공적으로 완료되었습니다!")
        logger.info(f"📂 결과 저장 위치: {results_path}")

    except Exception as e:
        logger.error(f"❌ 평가 중 오류 발생: {e}", exc_info=True)
        raise


def main():
    """메인 CLI 함수 (새로운 선언적 조립 구조 지원)"""
    parser = setup_args()
    args = parser.parse_args()
    
    try:
        validate_args(args)
        
        if args.mode == "validate":
            # 설정 파일 검증 모드
            success = validate_mode(args)
            sys.exit(0 if success else 1)
            
        elif args.mode == "train":
            # 학습 모드
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = Path.cwd() / config_path
            config = load_config(config_path)
            train_mode(args, config)
            
        elif args.mode == "evaluate":
            # 평가 모드
            evaluate_mode(args)
            
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ 입력 오류: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).critical(f"❌ 실행 실패: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # 이 파일이 직접 실행될 때는 아무 일도 일어나지 않음.
    # scs 명령어 또는 run.py를 통해 main()이 호출되어야 함.
    pass