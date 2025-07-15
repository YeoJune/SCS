"""
SCS 리팩토링된 기본 사용 예제

개선된 설정 시스템과 모듈화된 구조를 활용한 예제입니다.
"""

import torch
from pathlib import Path

# 리팩토링된 SCS 컴포넌트 import
from src.scs import (
    SCS, SCSConfig, BrainRegion, InputOutputConfig, TimingConfig,
    SCSTrainer, DataProcessor, SCSDataset,
    setup_logger, set_random_seed, get_device,
    validate_config, ValidationUtils
)


def create_example_config(vocab_size: int) -> SCSConfig:
    """예제용 SCS 설정 생성"""
    
    # 기본 설정 생성
    config = SCSConfig.create_default(vocab_size)
    
    # 실험용으로 크기 축소
    for region in config.modules:
        for layer_type in config.modules[region].layers:
            config.modules[region].layers[layer_type].num_neurons //= 4  # 1/4 크기로 축소
    
    # 입출력 설정 조정
    config.io_config.embedding_dim = 256
    config.io_config.num_slots = 64
    config.io_config.attention_heads = 4
    
    # 타이밍 설정 조정
    config.timing_config.min_processing_clk = 10
    config.timing_config.max_processing_clk = 50
    
    return config


def create_dummy_dataset(tokenizer, size: int = 500, max_length: int = 64) -> SCSDataset:
    """더미 데이터셋 생성"""
    
    # 더미 텍스트 생성
    positive_templates = [
        "This is a great example of positive sentiment.",
        "I really love this amazing product quality.",
        "Excellent work and outstanding performance here.",
        "Wonderful experience with fantastic results.",
        "Brilliant solution and impressive achievement."
    ]
    
    negative_templates = [
        "This is a terrible example of poor quality.",
        "I really hate this disappointing product issue.",
        "Awful work and horrible performance here.",
        "Dreadful experience with devastating results.",
        "Terrible solution and disappointing failure."
    ]
    
    texts = []
    labels = []
    
    for i in range(size):
        if i % 2 == 0:
            text = positive_templates[i % len(positive_templates)]
            label = 1
        else:
            text = negative_templates[i % len(negative_templates)]
            label = 0
        
        texts.append(text)
        labels.append(label)
    
    return SCSDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )


def run_training_experiment(config: SCSConfig, train_dataset: SCSDataset, val_dataset: SCSDataset):
    """학습 실험 실행"""
    
    logger = setup_logger("SCS_Experiment")
    device = get_device()
    
    logger.info("=== SCS 리팩토링된 학습 실험 시작 ===")
    logger.info(f"장치: {device}")
    logger.info(f"어휘 크기: {config.io_config.vocab_size}")
    logger.info(f"학습 데이터: {len(train_dataset)} 샘플")
    logger.info(f"검증 데이터: {len(val_dataset)} 샘플")
    
    # 모듈별 뉴런 수 출력
    for region_name, module_config in config.modules.items():
        total_neurons = sum(
            layer_config.num_neurons 
            for layer_config in module_config.layers.values()
        )
        logger.info(f"{region_name.value} 모듈: {total_neurons} 뉴런")
    
    # SCS 모델 초기화
    logger.info("SCS 모델 초기화 중...")
    model = SCS(config=config, device=device).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"모델 매개변수 수: {param_count:,}")
    
    # 데이터 로더 생성
    data_processor = DataProcessor(max_length=config.io_config.num_slots)
    data_processor.tokenizer = train_dataset.tokenizer  # 더미 토크나이저 설정
    
    train_loader = data_processor.create_data_loader(
        train_dataset, batch_size=8, shuffle=True
    )
    val_loader = data_processor.create_data_loader(
        val_dataset, batch_size=8, shuffle=False
    )
    
    # 학습기 설정
    training_config = {
        "learning_rate": 0.001,
        "optimizer": "adam",
        "weight_decay": 1e-5,
        "gradient_clipping": True,
        "max_grad_norm": 1.0
    }
    
    trainer = SCSTrainer(
        model=model,
        config=training_config,
        device=device
    )
    
    # 학습 실행
    try:
        logger.info("학습 시작...")
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,  # 빠른 테스트를 위해 3 에포크
            save_dir="outputs/refactored_checkpoints"
        )
        
        # 학습 결과
        history = trainer.get_training_history()
        
        if history["train_accuracy"]:
            final_train_acc = history["train_accuracy"][-1]
            final_val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0
            final_spike_activity = history["spike_activity"][-1] if history["spike_activity"] else 0
            
            logger.info(f"최종 학습 정확도: {final_train_acc:.4f}")
            logger.info(f"최종 검증 정확도: {final_val_acc:.4f}")
            logger.info(f"최종 스파이크 활성도: {final_spike_activity:.3f}")
        
        # 추론 테스트
        test_inference(model, data_processor, config, logger)
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def test_inference(model, data_processor, config: SCSConfig, logger):
    """추론 테스트"""
    
    logger.info("=== 추론 테스트 ===")
    
    model.eval()
    test_texts = [
        "This is an excellent and amazing product!",
        "This is a terrible and disappointing experience.",
        "Great work with fantastic results achieved.",
        "Poor quality with awful performance shown."
    ]
    
    for i, text in enumerate(test_texts):
        logger.info(f"\n테스트 {i+1}: {text}")
        
        # 토큰화
        encoding = data_processor.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=config.io_config.num_slots,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        
        with torch.no_grad():
            # SCS 처리
            output_probs, processing_info = model(input_ids, attention_mask)
            
            # 예측 결과
            prediction = torch.argmax(output_probs, dim=-1).item()
            confidence = torch.max(output_probs).item()
            
            logger.info(f"  예측: {'긍정' if prediction == 1 else '부정'}")
            logger.info(f"  신뢰도: {confidence:.4f}")
            logger.info(f"  처리 시간: {processing_info['processing_clk']} CLK")
            logger.info(f"  수렴 달성: {processing_info.get('convergence_achieved', False)}")
            
            # 모듈별 활성도
            module_activity = processing_info.get("module_activity", {})
            for region_name, activity in module_activity.items():
                logger.info(f"  {region_name} 활성도: {activity:.3f}")
            
            # 처리 분석
            analysis = model.get_processing_analysis()
            if analysis and "activity_stability" in analysis:
                logger.info(f"  활성도 안정성: {analysis['activity_stability']:.3f}")


def main():
    """메인 실행 함수"""
    
    # 기본 설정
    set_random_seed(42)
    logger = setup_logger("SCS_Refactored_Main")
    
    logger.info("SCS 리팩토링된 기본 예제 시작")
    
    try:
        # 더미 토크나이저 생성
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 1000
            
            def __call__(self, text, **kwargs):
                # 간단한 토큰화 (단어 분할)
                tokens = text.lower().split()[:kwargs.get("max_length", 64)]
                
                # 해시 기반 토큰 ID 생성
                token_ids = [hash(token) % self.vocab_size for token in tokens]
                
                # 패딩
                max_length = kwargs.get("max_length", 64)
                if len(token_ids) < max_length:
                    token_ids.extend([0] * (max_length - len(token_ids)))
                else:
                    token_ids = token_ids[:max_length]
                
                # 어텐션 마스크
                attention_mask = [1] * min(len(tokens), max_length)
                if len(attention_mask) < max_length:
                    attention_mask.extend([0] * (max_length - len(attention_mask)))
                
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.tensor([attention_mask])
                }
        
        tokenizer = DummyTokenizer()
        
        # SCS 설정 생성
        config = create_example_config(vocab_size=tokenizer.vocab_size)
        
        # 설정 검증
        if not validate_config(config):
            logger.error("설정 검증 실패")
            return
        
        logger.info("설정 검증 통과")
        
        # 데이터셋 생성
        logger.info("데이터셋 생성 중...")
        train_dataset = create_dummy_dataset(tokenizer, size=200, max_length=config.io_config.num_slots)
        val_dataset = create_dummy_dataset(tokenizer, size=50, max_length=config.io_config.num_slots)
        
        # 학습 실험 실행
        run_training_experiment(config, train_dataset, val_dataset)
        
        logger.info("리팩토링된 SCS 예제 완료")
        
    except Exception as e:
        logger.error(f"예제 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
