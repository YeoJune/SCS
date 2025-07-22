#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCS (Spike-Based Cognitive System) 공식 CLI 실행 진입점

이 스크립트는 패키지 설치 후 `scs` 명령어를 통해 호출되며,
실험의 전체 파이프라인을 관장하는 중앙 오케스트레이터입니다.
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any
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
    parser = argparse.ArgumentParser(description="SCS 실행 스크립트")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate"], help="실행 모드")
    parser.add_argument("--config", type=str, help="설정 파일 경로 (train 모드 필수)")
    parser.add_argument("--experiment_dir", type=str, help="실험 디렉토리 경로 (evaluate 모드 필수)")
    parser.add_argument("--device", type=str, default="auto", help="연산 장치 선택 (cuda, cpu, mps)")
    parser.add_argument("--seed", type=int, default=42, help="재현성을 위한 랜덤 시드")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 (상세 로깅)")
    return parser

def validate_args(args: argparse.Namespace):
    if args.mode == "train" and not args.config:
        raise ValueError("train 모드에서는 --config 인자가 필수입니다.")
    if args.mode == "evaluate" and not args.experiment_dir:
        raise ValueError("evaluate 모드에서는 --experiment_dir 인자가 필수입니다.")
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {args.config}")
    if args.experiment_dir and not Path(args.experiment_dir).exists():
        raise FileNotFoundError(f"실험 디렉토리를 찾을 수 없습니다: {args.experiment_dir}")


# --- 모드별 실행 함수 ---
def train_mode(args: argparse.Namespace, config: Dict[str, Any]):
    """학습 모드 실행"""
    # 1. 실험 환경 설정
    experiment_name = f"{Path(args.config).stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir / "config.yaml")
    setup_logging(log_dir=experiment_dir / "logs", level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    set_random_seed(args.seed)
    device = get_device(args.device)
    logger.info(f"실험 '{experiment_name}' 시작 | 디바이스: {device}")

    try:
        # 2. 데이터 로더 생성
        tokenizer = SCSTokenizer(config["data_loading"]["tokenizer"]["name"])
        
        # 데이터셋 이름 가져오기 - config 형식에 따라 다른 위치 확인
        dataset_name = None
        if "task" in config and "dataset_name" in config["task"]:
            dataset_name = config["task"]["dataset_name"]
        elif "data" in config and "dataset_name" in config["data"]:
            dataset_name = config["data"]["dataset_name"]  
        elif "dataset_name" in config:
            dataset_name = config["dataset_name"]
        else:
            # base_model.yaml 같은 경우 기본값 사용
            dataset_name = "datatune/LogiQA2.0"
            logger.warning(f"dataset_name not found in config, using default: {dataset_name}")
        
        train_loader = create_dataloader(dataset_name=dataset_name, split="train", batch_size=config["data_loading"]["batch_size"], max_length=config["data_loading"]["tokenizer"]["max_length"], tokenizer=tokenizer)
        val_loader = create_dataloader(dataset_name=dataset_name, split="validation", batch_size=1, max_length=config["data_loading"]["tokenizer"]["max_length"], tokenizer=tokenizer)

        # 3. 모델 인스턴스화
        config["io_system"]["input_interface"]["vocab_size"] = tokenizer.vocab_size
        config["io_system"]["output_interface"]["vocab_size"] = tokenizer.vocab_size
        model = ModelBuilder.build_scs_from_config(config, device=device)
        logger.info(f"모델 매개변수: {sum(p.numel() for p in model.parameters()):,}")

        # 4. 학습 시스템 구성
        pad_token_id = tokenizer.tokenizer.pad_token_id
        
        # config 매핑 - base_model.yaml은 "learning", phase2는 "training" 사용
        raw_config = config.get("learning", config.get("training", {})).copy()
        
        # 파라미터 이름 정규화
        if "base_learning_rate" in raw_config:
            raw_config["learning_rate"] = raw_config.pop("base_learning_rate")
        if "max_grad_norm" in raw_config:
            raw_config["gradient_clip_norm"] = raw_config.pop("max_grad_norm")
        if "eval_every_n_epochs" in raw_config:
            raw_config["eval_every"] = raw_config.pop("eval_every_n_epochs")
        if "save_every_n_epochs" in raw_config:
            raw_config["save_every"] = raw_config.pop("save_every_n_epochs")
        
        # TrainingConfig가 허용하는 파라미터만 필터링
        valid_params = {
            "epochs", "learning_rate", "weight_decay", "gradient_clip_norm",
            "eval_every", "save_every", "early_stopping_patience", "max_clk_training"
        }
        filtered_config = {k: v for k, v in raw_config.items() if k in valid_params}
        
        # 타입 변환 - YAML에서 문자열로 로드된 숫자 값들을 적절한 타입으로 변환
        if "learning_rate" in filtered_config:
            filtered_config["learning_rate"] = float(filtered_config["learning_rate"])
        if "weight_decay" in filtered_config:
            filtered_config["weight_decay"] = float(filtered_config["weight_decay"])
        if "gradient_clip_norm" in filtered_config:
            filtered_config["gradient_clip_norm"] = float(filtered_config["gradient_clip_norm"])
        if "epochs" in filtered_config:
            filtered_config["epochs"] = int(filtered_config["epochs"])
        if "eval_every" in filtered_config:
            filtered_config["eval_every"] = int(filtered_config["eval_every"])
        if "save_every" in filtered_config:
            filtered_config["save_every"] = int(filtered_config["save_every"])
        if "early_stopping_patience" in filtered_config:
            filtered_config["early_stopping_patience"] = int(filtered_config["early_stopping_patience"])
        if "max_clk_training" in filtered_config:
            filtered_config["max_clk_training"] = int(filtered_config["max_clk_training"])
        
        training_config = TrainingConfig(pad_token_id=pad_token_id, device=device, **filtered_config)
        loss_fn = MultiObjectiveLoss(pad_token_id=pad_token_id)
        
        # 옵티마이저 타입 가져오기
        optimizer_type = raw_config.get("optimizer", "adamw").lower()
        optimizer = OptimizerFactory.create(optimizer_type=optimizer_type, model=model, config=training_config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.epochs)
        
        # 5. 트레이너 생성 및 학습
        trainer = SCSTrainer(model=model, config=training_config, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, tokenizer=tokenizer)
        trainer.train(train_loader, val_loader, save_path=str(experiment_dir / "checkpoints"))

        # 6. 최종 평가
        logger.info("최종 평가 시작...")
        test_loader = create_dataloader(dataset_name=dataset_name, split="test", batch_size=1, max_length=config["data_loading"]["tokenizer"]["max_length"], tokenizer=tokenizer)
        test_results = trainer.evaluate(test_loader)
        save_config(test_results, experiment_dir / "results.yaml")

    except Exception as e:
        logger.error(f"학습 중 치명적인 오류 발생: {e}", exc_info=True)
        raise

def evaluate_mode(args: argparse.Namespace):
    """평가 모드 실행"""
    # 1. 환경 설정
    experiment_dir = Path(args.experiment_dir)
    config_path = experiment_dir / "config.yaml"
    checkpoint_path = experiment_dir / "checkpoints" / "best_model.pt"

    config = load_config(config_path)
    setup_logging(log_dir=experiment_dir / "logs" / "eval")
    logger = logging.getLogger(__name__)
    device = get_device(args.device)
    
    try:
        # 2. 데이터 및 모델 로드
        tokenizer = SCSTokenizer(config["data_loading"]["tokenizer"]["name"])
        
        # 데이터셋 이름 가져오기 - config 형식에 따라 다른 위치 확인
        dataset_name = None
        if "task" in config and "dataset_name" in config["task"]:
            dataset_name = config["task"]["dataset_name"]
        elif "data" in config and "dataset_name" in config["data"]:
            dataset_name = config["data"]["dataset_name"]  
        elif "dataset_name" in config:
            dataset_name = config["dataset_name"]
        else:
            # base_model.yaml 같은 경우 기본값 사용
            dataset_name = "datatune/LogiQA2.0"
            logger.warning(f"dataset_name not found in config, using default: {dataset_name}")
        
        test_loader = create_dataloader(dataset_name=dataset_name, split="test", batch_size=1, max_length=config["data_loading"]["tokenizer"]["max_length"], tokenizer=tokenizer)
        
        config["io_system"]["input_interface"]["vocab_size"] = tokenizer.vocab_size
        config["io_system"]["output_interface"]["vocab_size"] = tokenizer.vocab_size
        model = ModelBuilder.build_scs_from_config(config, device=device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 3. 트레이너 생성 및 평가
        pad_token_id = tokenizer.tokenizer.pad_token_id
        
        # config 매핑 - base_model.yaml은 "learning", phase2는 "training" 사용
        raw_config = config.get("learning", config.get("training", {})).copy()
        
        # 파라미터 이름 정규화
        if "base_learning_rate" in raw_config:
            raw_config["learning_rate"] = raw_config.pop("base_learning_rate")
        if "max_grad_norm" in raw_config:
            raw_config["gradient_clip_norm"] = raw_config.pop("max_grad_norm")
        if "eval_every_n_epochs" in raw_config:
            raw_config["eval_every"] = raw_config.pop("eval_every_n_epochs")
        if "save_every_n_epochs" in raw_config:
            raw_config["save_every"] = raw_config.pop("save_every_n_epochs")
        
        # TrainingConfig가 허용하는 파라미터만 필터링
        valid_params = {
            "epochs", "learning_rate", "weight_decay", "gradient_clip_norm",
            "eval_every", "save_every", "early_stopping_patience", "max_clk_training"
        }
        filtered_config = {k: v for k, v in raw_config.items() if k in valid_params}
        
        # 타입 변환 - YAML에서 문자열로 로드된 숫자 값들을 적절한 타입으로 변환
        if "learning_rate" in filtered_config:
            filtered_config["learning_rate"] = float(filtered_config["learning_rate"])
        if "weight_decay" in filtered_config:
            filtered_config["weight_decay"] = float(filtered_config["weight_decay"])
        if "gradient_clip_norm" in filtered_config:
            filtered_config["gradient_clip_norm"] = float(filtered_config["gradient_clip_norm"])
        if "epochs" in filtered_config:
            filtered_config["epochs"] = int(filtered_config["epochs"])
        if "eval_every" in filtered_config:
            filtered_config["eval_every"] = int(filtered_config["eval_every"])
        if "save_every" in filtered_config:
            filtered_config["save_every"] = int(filtered_config["save_every"])
        if "early_stopping_patience" in filtered_config:
            filtered_config["early_stopping_patience"] = int(filtered_config["early_stopping_patience"])
        if "max_clk_training" in filtered_config:
            filtered_config["max_clk_training"] = int(filtered_config["max_clk_training"])
        
        training_config = TrainingConfig(pad_token_id=pad_token_id, device=device, **filtered_config)
        trainer = SCSTrainer(model=model, config=training_config, tokenizer=tokenizer)
        results = trainer.evaluate(test_loader)
        
        logger.info("🎉 평가가 성공적으로 완료되었습니다!")

    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}", exc_info=True)
        raise


def main():
    """메인 CLI 함수"""
    parser = setup_args()
    args = parser.parse_args()
    try:
        validate_args(args)
        if args.mode == "train":
            config_path = Path(args.config)
            if not config_path.is_absolute():
                 config_path = Path.cwd() / config_path
            config = load_config(config_path)
            train_mode(args, config)
        elif args.mode == "evaluate":
            evaluate_mode(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ 입력 오류: {e}")
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).critical(f"실행 실패: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # 이 파일이 직접 실행될 때는 아무 일도 일어나지 않음.
    # scs 명령어 또는 run.py를 통해 main()이 호출되어야 함.
    pass