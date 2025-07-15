"""
단일 실행 진입점

모든 SCS 작업을 위한 통합 인터페이스
사용법: python run.py --mode [train|evaluate|analyze] --config <config_path>
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from scs.training import SCSTrainer
    from scs.data import DataProcessor, DataLoader
    from scs.utils import setup_logging, load_config, save_config
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
    print("💡 먼저 'pip install -e .' 명령어로 패키지를 설치해주세요.")
    sys.exit(1)


def setup_args() -> argparse.ArgumentParser:
    """명령행 인자 설정"""
    parser = argparse.ArgumentParser(
        description="SCS (Spike-Based Cognitive System) 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Phase 1 실험 실행
  python run.py --mode train --config configs/phase1_logic_ops.yaml
  
  # 기존 실험 평가
  python run.py --mode evaluate --experiment_dir experiments/clutrr_run_01
  
  # 결과 분석
  python run.py --mode analyze --experiment_dir experiments/clutrr_run_01
  
  # 여러 실험 비교
  python run.py --mode compare --experiments_dir experiments/ablation_results/
        """
    )
    
    # 필수 인자
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "analyze", "compare"],
        help="실행 모드"
    )
    
    # 설정 관련
    parser.add_argument(
        "--config",
        type=str,
        help="설정 파일 경로 (train 모드에서 필수)"
    )
    
    # 실험 디렉토리 관련
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="실험 디렉토리 경로 (evaluate/analyze 모드에서 필수)"
    )
    
    parser.add_argument(
        "--experiments_dir", 
        type=str,
        help="여러 실험 디렉토리 경로 (compare 모드에서 필수)"
    )
    
    # 실험 이름 (자동 생성되지만 수동 지정 가능)
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="실험 이름 (지정하지 않으면 자동 생성)"
    )
    
    # 추가 옵션들
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 활성화"
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true", 
        help="Weights & Biases 로깅 비활성화"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="대화형 입력 비활성화 (스크립트 실행용)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="연산 장치 선택"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """인자 유효성 검사"""
    if args.mode == "train":
        if not args.config:
            raise ValueError("train 모드에서는 --config가 필수입니다.")
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {args.config}")
            
    elif args.mode in ["evaluate", "analyze"]:
        if not args.experiment_dir:
            raise ValueError(f"{args.mode} 모드에서는 --experiment_dir이 필수입니다.")
        if not os.path.exists(args.experiment_dir):
            raise FileNotFoundError(f"실험 디렉토리를 찾을 수 없습니다: {args.experiment_dir}")
            
    elif args.mode == "compare":
        if not args.experiments_dir:
            raise ValueError("compare 모드에서는 --experiments_dir이 필수입니다.")
        if not os.path.exists(args.experiments_dir):
            raise FileNotFoundError(f"실험 디렉토리를 찾을 수 없습니다: {args.experiments_dir}")


def train_mode(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """학습 모드 실행"""
    print("🚀 학습 모드를 시작합니다...")
    
    # 실험 이름 생성
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        config_name = Path(args.config).stem
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"{config_name}_{timestamp}"
    
    # 실험 디렉토리 설정
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 저장
    config_copy = experiment_dir / "config.yaml"
    save_config(config, config_copy)
    
    # 로깅 설정
    setup_logging(
        log_dir=experiment_dir / "logs",
        level=logging.DEBUG if args.debug else logging.INFO
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"실험 시작: {experiment_name}")
    logger.info(f"실험 디렉토리: {experiment_dir}")
    
    try:
        # 데이터 로더 생성
        data_processor = DataProcessor(config["task"])
        train_loader, val_loader, test_loader = data_processor.get_dataloaders()
        
        # 트레이너 생성 및 학습 실행
        trainer = SCSTrainer(
            config=config,
            experiment_dir=experiment_dir,
            device=args.device,
            debug=args.debug,
            use_wandb=not args.no_wandb
        )
        
        # 학습 실행
        trainer.train(train_loader, val_loader)
        
        # 최종 평가
        test_results = trainer.evaluate(test_loader, save_results=True)
        
        logger.info("학습이 성공적으로 완료되었습니다!")
        logger.info(f"테스트 정확도: {test_results.get('accuracy', 'N/A')}")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        raise


def evaluate_mode(args: argparse.Namespace) -> None:
    """평가 모드 실행"""
    print("📊 평가 모드를 시작합니다...")
    
    experiment_dir = Path(args.experiment_dir)
    
    # 설정 파일 로드
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    config = load_config(config_path)
    
    # 로깅 설정
    setup_logging(
        log_dir=experiment_dir / "logs",
        level=logging.DEBUG if args.debug else logging.INFO
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"평가 시작: {experiment_dir}")
    
    try:
        # 데이터 로더 생성
        data_processor = DataProcessor(config["task"])
        _, _, test_loader = data_processor.get_dataloaders()
        
        # 트레이너 로드
        trainer = SCSTrainer.load_from_checkpoint(
            experiment_dir=experiment_dir,
            device=args.device,
            debug=args.debug
        )
        
        # 평가 실행
        results = trainer.evaluate(test_loader, save_results=True)
        
        logger.info("평가가 성공적으로 완료되었습니다!")
        for metric, value in results.items():
            logger.info(f"{metric}: {value}")
            
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        raise


def analyze_mode(args: argparse.Namespace) -> None:
    """분석 모드 실행"""
    print("🔍 분석 모드를 시작합니다...")
    
    experiment_dir = Path(args.experiment_dir)
    
    # 설정 파일 로드
    config_path = experiment_dir / "config.yaml"
    config = load_config(config_path)
    
    # 로깅 설정
    setup_logging(
        log_dir=experiment_dir / "logs",
        level=logging.DEBUG if args.debug else logging.INFO
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"분석 시작: {experiment_dir}")
    
    try:
        # 분석 실행 (구현 예정)
        from scs.analysis import ExperimentAnalyzer
        
        analyzer = ExperimentAnalyzer(experiment_dir)
        
        # 다양한 분석 수행
        if config.get("analysis", {}).get("dynamics_analysis", {}).get("enabled", False):
            analyzer.analyze_dynamics()
            
        if config.get("analysis", {}).get("representation_analysis", {}).get("enabled", False):
            analyzer.analyze_representations()
            
        if config.get("analysis", {}).get("ablation_study", {}).get("enabled", False):
            analyzer.analyze_ablations()
            
        logger.info("분석이 성공적으로 완료되었습니다!")
        
    except ImportError:
        logger.warning("분석 모듈이 아직 구현되지 않았습니다.")
        logger.info("기본 분석을 수행합니다...")
        
        # 기본 분석: 로그 파일 요약
        log_files = list(experiment_dir.glob("logs/*.log"))
        logger.info(f"로그 파일 {len(log_files)}개 발견")
        
        results_file = experiment_dir / "results.json"
        if results_file.exists():
            import json
            with open(results_file) as f:
                results = json.load(f)
            logger.info("평가 결과:")
            for key, value in results.items():
                logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        raise


def compare_mode(args: argparse.Namespace) -> None:
    """비교 모드 실행"""
    print("⚖️  비교 모드를 시작합니다...")
    
    experiments_dir = Path(args.experiments_dir)
    
    # 모든 실험 디렉토리 찾기
    experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    
    print(f"발견된 실험: {len(experiment_dirs)}개")
    
    try:
        # 비교 분석 실행 (구현 예정)
        from scs.analysis import ComparisonAnalyzer
        
        analyzer = ComparisonAnalyzer(experiment_dirs)
        analyzer.generate_comparison_report()
        
        print("비교 분석이 성공적으로 완료되었습니다!")
        
    except ImportError:
        print("⚠️  비교 분석 모듈이 아직 구현되지 않았습니다.")
        
        # 기본 비교: 결과 파일들 수집
        results = {}
        for exp_dir in experiment_dirs:
            results_file = exp_dir / "results.json"
            if results_file.exists():
                import json
                with open(results_file) as f:
                    results[exp_dir.name] = json.load(f)
                    
        if results:
            print("🏆 실험 결과 요약:")
            for exp_name, exp_results in results.items():
                accuracy = exp_results.get("accuracy", "N/A")
                print(f"  {exp_name}: {accuracy}")
        else:
            print("📋 결과 파일을 찾을 수 없습니다.")


def main():
    """메인 함수"""
    parser = setup_args()
    args = parser.parse_args()
    
    try:
        # 인자 유효성 검사
        validate_args(args)
        
        # 모드별 실행
        if args.mode == "train":
            config = load_config(args.config)
            train_mode(args, config)
            
        elif args.mode == "evaluate":
            evaluate_mode(args)
            
        elif args.mode == "analyze":
            analyze_mode(args)
            
        elif args.mode == "compare":
            compare_mode(args)
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
