"""SCS 기본 컴포넌트 테스트"""

import torch
import numpy as np
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.scs.utils import setup_logger, set_random_seed


def test_spike_node():
    print("\n=== SpikeNode 테스트 ===")
    
    try:
        from src.scs.architecture.node import SpikeNode
        
        # 기본 파라미터로 노드 생성
        node = SpikeNode(
            threshold=1.0,
            tau_membrane=20.0,
            tau_refractory=5.0,
            device="cpu"
        )
        
        # 테스트 입력 생성
        batch_size = 4
        input_current = torch.randn(batch_size, 1) * 2.0
        
        # 여러 시간 스텝 실행
        spike_history = []
        membrane_history = []
        
        for t in range(50):
            spikes, states = node(input_current)
            spike_history.append(spikes.mean().item())
            membrane_history.append(states["membrane_potential"].mean().item())
        
        avg_spike_rate = np.mean(spike_history)
        print(f"평균 스파이크 발화율: {avg_spike_rate:.3f}")
        print(f"최종 막전위: {membrane_history[-1]:.3f}")
        
        # 기본 검증
        assert len(spike_history) == 50, "시간 스텝 수가 맞지 않음"
        assert 0 <= avg_spike_rate <= 1, "스파이크 발화율이 범위를 벗어남"
        
        print("✓ SpikeNode 테스트 통과")
        return True
        
    except Exception as e:
        print(f"✗ SpikeNode 테스트 실패: {e}")
        return False


def test_cognitive_module():
    """CognitiveModule 테스트"""
    print("\n=== CognitiveModule 테스트 ===")
    
    try:
        from src.scs.architecture.module import CognitiveModule
        
        # PFC 모듈 생성
        module = CognitiveModule(
            module_name="PFC",
            num_neurons=128,
            config={
                "num_layers": 4,
                "layer_sizes": [32, 32, 32, 32],
                "connection_prob": 0.1
            },
            device="cpu"
        )
        
        # 테스트 입력
        batch_size = 2
        external_input = torch.randn(batch_size, 128) * 0.5
        
        # 모듈 실행
        output_spikes, module_states = module(external_input)
        
        print(f"출력 스파이크 형태: {output_spikes.shape}")
        print(f"스파이크 발화율: {module_states['spike_rate']:.3f}")
        print(f"층별 활성도: {[f'{act:.3f}' for act in module_states['layer_activities']]}")
        
        # 기본 검증
        assert output_spikes.shape == (batch_size, 128), "출력 형태가 맞지 않음"
        assert 0 <= module_states["spike_rate"] <= 1, "스파이크 발화율이 범위를 벗어남"
        
        print("✓ CognitiveModule 테스트 통과")
        return True
        
    except Exception as e:
        print(f"✗ CognitiveModule 테스트 실패: {e}")
        return False


def test_input_output_nodes():
    """InputNode와 OutputNode 테스트"""
    print("\n=== InputNode & OutputNode 테스트 ===")
    
    try:
        from SCS.src.scs.architecture.io_node import InputNode, OutputNode
        
        # 더미 토크나이저 클래스
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 1000
            
            def __call__(self, text, **kwargs):
                # 더미 토큰화 결과
                return {
                    "input_ids": torch.randint(0, 1000, (1, 10)),
                    "attention_mask": torch.ones(1, 10)
                }
        
        tokenizer = DummyTokenizer()
        
        # InputNode 테스트
        input_node = InputNode(
            vocab_size=1000,
            embedding_dim=256,
            num_slots=128,
            device="cpu"
        )
        
        # OutputNode 테스트
        output_node = OutputNode(
            vocab_size=1000,
            embedding_dim=256,
            num_input_neurons=128,
            device="cpu"
        )
        
        # 테스트 데이터
        batch_size = 2
        token_ids = torch.randint(0, 1000, (batch_size, 10))
        attention_mask = torch.ones(batch_size, 10)
        
        # InputNode 실행
        spike_patterns, input_states = input_node(token_ids, attention_mask)
        
        print(f"입력 스파이크 패턴 형태: {spike_patterns.shape}")
        print(f"스파이크 발화율: {input_states['spike_rate']:.3f}")
        print(f"활성 슬롯 수: {input_states['active_slots']:.1f}")
        
        # OutputNode 실행
        token_probs, output_states = output_node(spike_patterns)
        
        print(f"출력 토큰 확률 형태: {token_probs.shape}")
        print(f"최대 확률: {output_states['max_prob']:.3f}")
        print(f"엔트로피: {output_states['entropy']:.3f}")
        
        # 기본 검증
        assert spike_patterns.shape == (batch_size, 128), "스파이크 패턴 형태가 맞지 않음"
        assert token_probs.shape == (batch_size, 1000), "토큰 확률 형태가 맞지 않음"
        assert torch.allclose(token_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6), "확률 합이 1이 아님"
        
        print("✓ InputNode & OutputNode 테스트 통과")
        return True
        
    except Exception as e:
        print(f"✗ InputNode & OutputNode 테스트 실패: {e}")
        return False


def test_scs_system():
    """전체 SCS 시스템 테스트"""
    print("\n=== SCS 시스템 테스트 ===")
    
    try:
        from src.scs.architecture.system import SCS
        
        # 기본 설정
        config = {
            "embedding_dim": 256,
            "max_tokens": 64,
            "pfc_size": 64,
            "acc_size": 32,
            "ipl_size": 48,
            "mtl_size": 32,
            "min_clk": 10,
            "max_clk": 50
        }
        
        # SCS 모델 생성
        model = SCS(
            vocab_size=1000,
            config=config,
            device="cpu"
        )
        
        # 테스트 입력
        batch_size = 2
        token_ids = torch.randint(0, 1000, (batch_size, 8))
        attention_mask = torch.ones(batch_size, 8)
        
        # 순전파 실행
        output_probs, processing_info = model(token_ids, attention_mask, max_clk=20)
        
        print(f"출력 확률 형태: {output_probs.shape}")
        print(f"처리 시간: {processing_info['processing_clk']} CLK")
        print(f"수렴 달성: {processing_info['convergence_achieved']}")
        
        # 모듈 활성도 출력
        module_activity = processing_info.get("module_activity", {})
        for module_name, activity in module_activity.items():
            print(f"{module_name} 활성도: {activity:.3f}")
        
        # 처리 분석
        analysis = model.get_processing_analysis()
        if analysis:
            print(f"총 처리 시간: {analysis['total_processing_clk']} CLK")
            print(f"최종 활성도: {analysis['final_activity']:.3f}")
        
        # 기본 검증
        assert output_probs.shape == (batch_size, 1000), "출력 형태가 맞지 않음"
        assert processing_info["processing_clk"] > 0, "처리 시간이 0 이하임"
        assert len(module_activity) == 4, "모듈 수가 맞지 않음"
        
        print("✓ SCS 시스템 테스트 통과")
        return True
        
    except Exception as e:
        print(f"✗ SCS 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_processor():
    """DataProcessor 테스트"""
    print("\n=== DataProcessor 테스트 ===")
    
    try:
        from src.scs.data.dataset import DataProcessor, SCSDataset
        
        # 더미 토크나이저 클래스
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 1000
            
            def __call__(self, text, **kwargs):
                tokens = text.split()[:5]  # 최대 5 토큰
                token_ids = [hash(token) % 1000 for token in tokens]
                
                # 패딩
                max_length = kwargs.get("max_length", 10)
                if len(token_ids) < max_length:
                    token_ids.extend([0] * (max_length - len(token_ids)))
                else:
                    token_ids = token_ids[:max_length]
                
                attention_mask = [1] * min(len(tokens), max_length)
                if len(attention_mask) < max_length:
                    attention_mask.extend([0] * (max_length - len(attention_mask)))
                
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.tensor([attention_mask])
                }
        
        # DataProcessor 테스트
        processor = DataProcessor(
            tokenizer_name="dummy",
            max_length=16
        )
        processor.tokenizer = DummyTokenizer()  # 더미 토크나이저 설정
        
        # SCSDataset 테스트
        texts = [
            "This is a test sentence.",
            "Another example text.",
            "More test data here."
        ]
        labels = [0, 1, 0]
        
        dataset = SCSDataset(
            texts=texts,
            labels=labels,
            tokenizer=processor.tokenizer,
            max_length=16
        )
        
        print(f"데이터셋 크기: {len(dataset)}")
        
        # 첫 번째 아이템 확인
        item = dataset[0]
        print(f"input_ids 형태: {item['input_ids'].shape}")
        print(f"attention_mask 형태: {item['attention_mask'].shape}")
        print(f"label: {item['labels'].item()}")
        
        # 데이터 로더 생성
        dataloader = processor.create_data_loader(
            dataset,
            batch_size=2,
            shuffle=False
        )
        
        # 첫 번째 배치 확인
        batch = next(iter(dataloader))
        print(f"배치 input_ids 형태: {batch['input_ids'].shape}")
        print(f"배치 labels 형태: {batch['labels'].shape}")
        
        # 기본 검증
        assert len(dataset) == 3, "데이터셋 크기가 맞지 않음"
        assert item["input_ids"].shape == (16,), "input_ids 형태가 맞지 않음"
        assert batch["input_ids"].shape == (2, 16), "배치 형태가 맞지 않음"
        
        print("✓ DataProcessor 테스트 통과")
        return True
        
    except Exception as e:
        print(f"✗ DataProcessor 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("SCS 컴포넌트 테스트 시작")
    print("=" * 50)
    
    # 로거 설정
    logger = setup_logger("SCS_Test", level=logging.WARNING)  # 경고 이상만 출력
    
    # 재현성을 위한 시드 설정
    set_random_seed(42)
    
    # 테스트 실행
    tests = [
        test_spike_node,
        test_cognitive_module,
        test_input_output_nodes,
        test_data_processor,
        test_scs_system,  # 가장 복잡한 테스트를 마지막에
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} 테스트 중 예외 발생: {e}")
    
    print("\n" + "=" * 50)
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("✓ 모든 테스트 통과!")
        return True
    else:
        print(f"✗ {total - passed}개 테스트 실패")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
