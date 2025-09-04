# src/scs/cli.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCS (Spike-Based Cognitive System) 공식 CLI 실행 진입점 (간소화된 오케스트레이터)
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any
import torch

# 프로젝트 모듈 Import (리팩토링된 경로)
try:
    from scs.config import load_and_validate_config, ModelBuilder, AppConfig, LearningConfig
    from scs.training import SCSTrainer, MultiObjectiveLoss, TimingLoss, OptimizerFactory
    from scs.evaluation import SCSVisualizer, analyze_io_pipeline
    from scs.data import create_dataloader, SCSTokenizer
    from scs.utils import (
        setup_logging, save_config, set_random_seed, get_device
    )
except ImportError as e:
    print(f"⚠ 모듈 import 오류: {e}. 패키지가 올바르게 설치되었는지 확인해주세요. ('pip install -e .')")
    sys.exit(1)

def setup_args() -> argparse.ArgumentParser:
    """CLI 인자 설정"""
    parser = argparse.ArgumentParser(
        description="SCS 실행 스크립트 (리팩토링된 구조)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 학습 모드
  scs --mode train --config configs/phase2_logiqa_small.yaml
  
  # TensorBoard와 함께 학습
  scs --mode train --config configs/phase2_logiqa_small.yaml --tensorboard --tb-launch
  
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
    parser.add_argument("--tensorboard", action="store_true", 
                       help="TensorBoard 로깅 활성화")
    parser.add_argument("--tb-port", type=int, default=6006, 
                       help="TensorBoard 서버 포트")
    parser.add_argument("--tb-launch", action="store_true", 
                       help="TensorBoard 서버 자동 시작 및 브라우저 열기")
    
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


def validate_mode(args: argparse.Namespace):
    """설정 파일 구조 검증 모드 (config 패키지 사용)"""
    print("🔍 설정 파일 구조 검증을 시작합니다...")
    
    try:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        
        # config 패키지의 검증 기능 사용
        app_config = load_and_validate_config(config_path)
        
        print("✅ 설정 파일 구조가 올바릅니다!")
        
        # 간단한 모델 생성 테스트
        try:
            model = ModelBuilder.build_model(app_config, device="cpu")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 모델 정보:")
            print(f"   - 총 매개변수: {total_params:,}")
            print(f"   - 뇌 영역 수: {len(app_config.brain_regions)}")
            print(f"   - 축삭 연결 수: {len(app_config.axonal_connections.connections)}")
            print(f"   - 입력→출력: {app_config.system_roles.input_node} → {app_config.system_roles.output_node}")
            print("✅ 모델 생성 및 차원 검증 테스트 성공!")
        except Exception as model_error:
            print(f"⚠️ 모델 생성 테스트 실패: {model_error}")
            return False
            
        return True
        
    except Exception as e:
        print(f"⚠ 설정 파일 검증 중 오류 발생: {e}")
        return False


def train_mode(args: argparse.Namespace):
    """학습 모드 실행 (간소화된 오케스트레이터)"""
    # 1. 기본 설정
    experiment_name = f"{Path(args.config).stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir=experiment_dir / "logs", level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    set_random_seed(args.seed)
    device = get_device(args.device)
    
    logger.info(f"🚀 실험 '{experiment_name}' 시작 | 디바이스: {device}")

    try:
        # 2. 설정 로딩 및 검증 (config 패키지 사용)
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        
        app_config = load_and_validate_config(config_path)
        
        # TensorBoard 설정 오버라이드 (새로 추가)
        if args.tensorboard:
            app_config.logging.tensorboard.enabled = True
            app_config.logging.tensorboard.port = args.tb_port
            app_config.logging.tensorboard.auto_launch = args.tb_launch
            logger.info(f"📊 TensorBoard 활성화: 포트 {args.tb_port}, 자동 시작: {args.tb_launch}")

        save_config(app_config.model_dump(), experiment_dir / "config.yaml")
        logger.info("✅ 설정 파일 로딩 및 검증 완료")

        # 3. 데이터 로더 생성
        logger.info("📊 데이터 로더 생성 중...")
        tokenizer = SCSTokenizer(app_config.data_loading.tokenizer.name)
        
        # 토크나이저 설정 업데이트
        app_config.data_loading.tokenizer.pad_token_id = getattr(tokenizer.tokenizer, 'pad_token_id', 0)
        app_config.data_loading.tokenizer.eos_token_id = getattr(tokenizer.tokenizer, 'eos_token_id', 1)
        
        dataset_name = app_config.task.dataset_name
        learning_style = app_config.task.learning_style
        bert_config = app_config.task.bert_config
        
        train_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="train", 
            batch_size=app_config.data_loading.batch_size,
            max_length=app_config.data_loading.tokenizer.max_length,
            tokenizer=tokenizer,
            num_samples=app_config.data.train_samples,
            task_id=app_config.task.task_id,
            learning_style=learning_style,
            bert_config=bert_config
        )

        val_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="validation", 
            batch_size=app_config.data_loading.batch_size,
            max_length=app_config.data_loading.tokenizer.max_length,
            tokenizer=tokenizer,
            num_samples=app_config.data.val_samples,
            task_id=app_config.task.task_id,
            learning_style=learning_style,
            bert_config=bert_config
        )
        
        logger.info(f"✅ 데이터 로더 생성 완료 (데이터셋: {dataset_name}, 스타일: {learning_style})")

        # 4. 모델 생성 (config 패키지 사용)
        logger.info("🧠 SCS 모델 생성 중...")
        app_config.io_system.input_interface.vocab_size = tokenizer.vocab_size
        app_config.io_system.output_interface.vocab_size = tokenizer.vocab_size

        model = ModelBuilder.build_model(app_config, device=device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ 모델 생성 완료")
        logger.info(f"   - 총 매개변수: {total_params:,}")
        logger.info(f"   - 뇌 영역: {list(app_config.brain_regions.keys())}")

        # 5. 학습 시스템 구성
        logger.info("⚙️ 학습 시스템 구성 중...")
        
        # 학습 설정 준비
        learning_config = app_config.learning
        learning_config.device = device
        learning_config.pad_token_id = app_config.data_loading.tokenizer.pad_token_id

        # 손실 함수 생성
        loss_fn = TimingLoss(
            pad_token_id=app_config.data_loading.tokenizer.pad_token_id,
            guide_sep_token_id=learning_config.guide_sep_token_id,
            max_clk=learning_config.max_clk_training,
            guide_weight=learning_config.guide_weight,
            length_penalty_weight=learning_config.length_penalty_weight,
            orthogonal_reg_weight=learning_config.orthogonal_reg_weight,
            spike_reg_weight=learning_config.spike_reg_weight,
            target_spike_rate=learning_config.target_spike_rate,
            use_temporal_weighting=learning_config.use_temporal_weighting,
            initial_temporal_weight=learning_config.initial_temporal_weight,
            final_temporal_weight=learning_config.final_temporal_weight,
            timing_weight=learning_config.timing_weight,
            sync_target_start=learning_config.sync_target_start,
            sync_target_end=learning_config.sync_target_end,
            gate_pruning_weight=learning_config.gate_pruning_weight,
            gate_temperature=learning_config.gate_temperature,
            inner_pruning_weight=learning_config.inner_pruning_weight,
            inner_temperature=learning_config.inner_temperature,
            axon_strength_reg_weight=learning_config.axon_strength_reg_weight
        )
        
        # 옵티마이저 생성
        optimizer = OptimizerFactory.create(
            optimizer_type=learning_config.optimizer, 
            model=model, 
            config=learning_config
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=learning_config.epochs,
            eta_min=learning_config.eta_min
        )
        
        logger.info(f"✅ 학습 시스템 구성 완료 (옵티마이저: {learning_config.optimizer})")

        # 6. 트레이너 생성 및 학습
        logger.info("🎯 학습 시작...")
        
        # 점진적 해제 설정
        unfreezing_config = learning_config.gradual_unfreezing.model_dump() if learning_config.gradual_unfreezing else None
        
        trainer = SCSTrainer(
            model=model, 
            config=learning_config, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            tokenizer=tokenizer,
            unfreezing_config=unfreezing_config,
            tensorboard_config=app_config.logging.tensorboard.model_dump(),
            experiment_dir=experiment_dir
        )
        trainer.train(train_loader, val_loader, save_path=str(experiment_dir / "checkpoints"))

        # 7. 최종 평가
        logger.info("📈 최종 평가 시작...")
        test_loader = create_dataloader(
            dataset_name=dataset_name, 
            split="test", 
            batch_size=app_config.data_loading.batch_size,
            max_length=app_config.data_loading.tokenizer.max_length,
            tokenizer=tokenizer,
            num_samples=app_config.data.test_samples,
            task_id=app_config.task.task_id,
            learning_style=learning_style,
            bert_config=bert_config
        )
        
        test_results = trainer.evaluate(test_loader, save_examples=app_config.evaluation.save_examples)
        
        # 결과 저장
        results_path = experiment_dir / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M')}.yaml"
        save_config(test_results, results_path)
        
        logger.info("🎉 학습 및 평가가 성공적으로 완료되었습니다!")
        logger.info("📊 최종 평가 결과:")
        for key, value in test_results.items():
            if key not in ['examples']:
                if isinstance(value, float):
                    logger.info(f"   - {key}: {value:.4f}")
                else:
                    logger.info(f"   - {key}: {value}")
        
        logger.info(f"📂 결과 저장 위치: {experiment_dir}")

        # 8. 시각화 및 분석 (evaluation 패키지 사용)
        logger.info("🎨 시각화 생성 중...")
        visualizer = SCSVisualizer()
        visualizer.generate_all_visualizations(model, test_loader, experiment_dir)

        logger.info("🔬 IO 파이프라인 분석 중...")
        analyze_io_pipeline(model, test_loader, experiment_dir, device)

    except Exception as e:
        logger.error(f"⚠ 학습 중 치명적인 오류 발생: {e}", exc_info=True)
        raise


def evaluate_mode(args: argparse.Namespace):
    """평가 모드 실행 (간소화된 오케스트레이터)"""
    # 1. 기본 설정
    experiment_dir = Path(args.experiment_dir)
    config_path = experiment_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    setup_logging(log_dir=experiment_dir / "logs" / "eval", level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    device = get_device(args.device)
    logger.info(f"📊 평가 모드 시작 | 디바이스: {device}")
    
    try:
        # 2. 설정 로딩 (config 패키지 사용)
        app_config = load_and_validate_config(config_path)
        
        # TensorBoard 설정 오버라이드 (평가 모드에서도 로깅 가능)
        if args.tensorboard:
            app_config.logging.tensorboard.enabled = True
            app_config.logging.tensorboard.port = args.tb_port
            app_config.logging.tensorboard.auto_launch = args.tb_launch
            logger.info(f"📊 평가 모드 TensorBoard 활성화: 포트 {args.tb_port}")
        
        logger.info("✅ 저장된 설정 파일 로딩 완료")
        
        # 3. 체크포인트 찾기
        checkpoint_dir = experiment_dir / "checkpoints"
        best_model_path = checkpoint_dir / "best_model.pt"
        if not best_model_path.exists():
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_dir}")
            best_model_path = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
        
        logger.info(f"사용할 체크포인트: {best_model_path}")

        # 4. 데이터 로더 생성
        tokenizer = SCSTokenizer(app_config.data_loading.tokenizer.name)
        app_config.io_system.input_interface.vocab_size = tokenizer.vocab_size
        app_config.io_system.output_interface.vocab_size = tokenizer.vocab_size
        
        test_loader = create_dataloader(
            dataset_name=app_config.task.dataset_name,
            split="test",
            batch_size=app_config.data_loading.batch_size,
            max_length=app_config.data_loading.tokenizer.max_length,
            tokenizer=tokenizer,
            num_samples=app_config.data.test_samples,
            task_id=app_config.task.task_id,
            learning_style=app_config.task.learning_style,
            bert_config=app_config.task.bert_config
        )

        # 5. 모델 로드 (config 패키지 사용)
        logger.info("🧠 모델 복원 중...")
        model = ModelBuilder.build_model(app_config, device=device)
        
        # 체크포인트 로드
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("✅ 모델 복원 완료")

        # 6. 평가 실행
        logger.info("📈 평가 실행 중...")
        learning_config = app_config.learning
        learning_config.device = device
        learning_config.pad_token_id = app_config.data_loading.tokenizer.pad_token_id
        
        trainer = SCSTrainer(
            model=model, 
            config=learning_config, 
            tokenizer=tokenizer,
            tensorboard_config=app_config.logging.tensorboard.model_dump(),
            experiment_dir=experiment_dir
        )
        results = trainer.evaluate(test_loader, save_examples=app_config.evaluation.save_examples)
        
        # 결과 저장
        results_path = experiment_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.yaml"
        save_config(results, results_path)
        
        logger.info("🎉 평가가 성공적으로 완료되었습니다!")
        logger.info("📊 평가 결과:")
        for key, value in results.items():
            if key not in ['examples']:
                if isinstance(value, float):
                    logger.info(f"   - {key}: {value:.4f}")
                else:
                    logger.info(f"   - {key}: {value}")
        
        logger.info(f"📂 결과 저장 위치: {results_path}")
        
        # 7. 시각화 및 분석 (evaluation 패키지 사용)
        logger.info("🎨 시각화 생성 중...")
        visualizer = SCSVisualizer()
        visualizer.generate_visualizations(model, test_loader, experiment_dir)

        logger.info("🔬 IO 파이프라인 분석 중...")
        analyze_io_pipeline(model, test_loader, experiment_dir, device)

    except Exception as e:
        logger.error(f"⚠ 평가 중 오류 발생: {e}", exc_info=True)
        raise


def main():
    """메인 CLI 함수 (간소화된 오케스트레이터)"""
    parser = setup_args()
    args = parser.parse_args()
    
    try:
        validate_args(args)
        
        if args.mode == "validate":
            success = validate_mode(args)
            sys.exit(0 if success else 1)
            
        elif args.mode == "train":
            train_mode(args)
            
        elif args.mode == "evaluate":
            evaluate_mode(args)
            
    except (ValueError, FileNotFoundError) as e:
        print(f"⚠ 입력 오류: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).critical(f"⚠ 실행 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    pass