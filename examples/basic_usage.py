"""SCS 기본 사용 예제"""

import torch
import yaml
from pathlib import Path

from src.scs import (
    SCSSystem, SCSTrainer, DataProcessor, create_scs_datasets
)
from utils import setup_logging, load_config, set_random_seed, get_device


def main():
    logger = setup_logging("SCS_Example")
    logger.info("SCS 기본 예제 시작")
    
    # 설정 로드
    config_path = "configs/base_model.yaml"
    
    try:
        config = load_config(config_path)
        logger.info(f"설정 로드 완료: {config_path}")
    except Exception as e:
        logger.warning(f"설정 로드 실패: {e}")
        # 기본 설정 사용
        config = {
            "model": {
                "embedding_dim": 512,
                "max_tokens": 128,
                "pfc_size": 256,
                "acc_size": 128,
                "ipl_size": 192,
                "mtl_size": 128,
                "min_clk": 20,
                "max_clk": 100
            },
            "training": {
                "learning_rate": 0.001,
                "batch_size": 16,
                "num_epochs": 5,
                "optimizer": "adam",
                "weight_decay": 1e-5
            },
            "data": {
                "tokenizer": "bert-base-uncased",
                "max_length": 128,
                "train_samples": 1000,
                "val_samples": 200
            }
        }
    
    # 재현성을 위한 시드 설정
    set_random_seed(42)
    
    # 장치 설정
    device = get_device()
    logger.info(f"사용 장치: {device}")
    
    # 데이터 처리기 초기화
    data_processor = DataProcessor(
        tokenizer_name=config["data"]["tokenizer"],
        max_length=config["data"]["max_length"]
    )
    
    # 간단한 더미 데이터셋 생성 (실제로는 Hugging Face 데이터셋 사용)
    logger.info("더미 데이터셋 생성 중...")
    
    # 더미 텍스트와 라벨 생성
    dummy_texts = [
        "This is a positive example.",
        "This is a negative example.", 
        "Another positive sentence.",
        "Another negative sentence.",
        "Positive sentiment here.",
        "Negative sentiment here."
    ] * 200  # 반복하여 충분한 데이터 생성
    
    dummy_labels = [1, 0, 1, 0, 1, 0] * 200  # 긍정(1), 부정(0) 라벨
    
    # 학습/검증 분할
    train_size = config["data"]["train_samples"]
    val_size = config["data"]["val_samples"]
    
    from src.scs import SCSDataset
    
    train_dataset = SCSDataset(
        texts=dummy_texts[:train_size],
        labels=dummy_labels[:train_size],
        tokenizer=data_processor.tokenizer,
        max_length=config["data"]["max_length"]
    )
    
    val_dataset = SCSDataset(
        texts=dummy_texts[train_size:train_size+val_size],
        labels=dummy_labels[train_size:train_size+val_size],
        tokenizer=data_processor.tokenizer,
        max_length=config["data"]["max_length"]
    )
    
    # 데이터 로더 생성
    train_loader = data_processor.create_data_loader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    val_loader = data_processor.create_data_loader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )
    
    logger.info(f"학습 데이터: {len(train_dataset)} 샘플")
    logger.info(f"검증 데이터: {len(val_dataset)} 샘플")
    
    # SCS 모델 초기화
    logger.info("SCS 모델 초기화 중...")
    
    model = SCSSystem(
        vocab_size=data_processor.get_vocab_size(),
        config=config["model"],
        device=device
    ).to(device)
    
    logger.info(f"모델 매개변수 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 학습기 초기화
    trainer = SCSTrainer(
        model=model,
        config=config["training"],
        device=device
    )
    
    logger.info("학습 시작...")
    
    try:
        # 모델 학습
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["training"]["num_epochs"],
            save_dir="outputs/checkpoints"
        )
        
        logger.info("학습 완료!")
        
        # 학습 기록 저장
        history = trainer.get_training_history()
        
        # 간단한 결과 출력
        final_train_acc = history["train_accuracy"][-1] if history["train_accuracy"] else 0
        final_val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0
        final_spike_activity = history["spike_activity"][-1] if history["spike_activity"] else 0
        
        logger.info(f"최종 학습 정확도: {final_train_acc:.4f}")
        logger.info(f"최종 검증 정확도: {final_val_acc:.4f}")
        logger.info(f"최종 스파이크 활성도: {final_spike_activity:.3f}")
        
        # 모델 추론 테스트
        logger.info("추론 테스트 중...")
        
        model.eval()
        test_text = "This is a test sentence for inference."
        
        # 토큰화
        test_encoding = data_processor.tokenizer(
            test_text,
            truncation=True,
            padding="max_length",
            max_length=config["data"]["max_length"],
            return_tensors="pt"
        )
        
        test_input_ids = test_encoding["input_ids"].to(device)
        test_attention_mask = test_encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            output_probs, processing_info = model(test_input_ids, test_attention_mask)
            
            # 예측 결과
            prediction = torch.argmax(output_probs, dim=-1).item()
            confidence = torch.max(output_probs).item()
            
            logger.info(f"테스트 입력: {test_text}")
            logger.info(f"예측 라벨: {prediction}")
            logger.info(f"신뢰도: {confidence:.4f}")
            logger.info(f"처리 시간: {processing_info['processing_clk']} CLK")
            
            # 모듈별 활성도
            module_activity = processing_info.get("module_activity", {})
            for module_name, activity in module_activity.items():
                logger.info(f"{module_name} 활성도: {activity:.3f}")
    
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("SCS 기본 예제 완료")


if __name__ == "__main__":
    main()
