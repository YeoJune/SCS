# utils/model_builder.py
"""
SCS 모델 빌더 - 선언적 조립 설명서 기반 모델 생성

이 모듈은 config 파일을 해석하여 동적으로 SCS 시스템을 조립합니다.
"""

from typing import Dict, Any, Tuple, List


class ModelBuilder:
    """SCS 모델 생성을 위한 빌더 클래스 - 선언적 조립 기반"""
    
    @staticmethod
    def build_scs_from_config(config: Dict[str, Any], device: str = "cpu"):
        """
        설정 파일로부터 SCS 시스템을 동적으로 생성합니다
        
        새로운 선언적 조립 방식:
        1. system_roles에서 입출력 노드 역할 정의
        2. brain_regions에서 각 노드의 사양 정의 (grid_size 배열 사용)
        3. axonal_connections에서 Conv2d 기반 연결 정의
        4. ModelBuilder가 이 정보를 해석하여 전체 시스템 조립
        
        Args:
            config: 설정 딕셔너리
            device: 연산 장치
            
        Returns:
            조립된 SCSSystem 인스턴스
        """
        try:
            # 필요한 모듈들을 동적으로 import
            from ..architecture import (
                SpikeNode, LocalConnectivity, AxonalConnections,
                InputInterface, OutputInterface, SCSSystem
            )
            from ..architecture.timing import TimingManager
            import torch.nn as nn
            
            # --- 단계 1: 뇌 영역(노드) 객체 생성 ---
            nodes = {}
            local_connections = {}
            node_grid_sizes = {}  # 노드별 그리드 크기 저장을 위한 헬퍼 딕셔너리
            
            if "brain_regions" not in config:
                raise ValueError("Config 파일에 'brain_regions' 섹션이 필요합니다.")
            
            for region_name, region_config in config["brain_regions"].items():
                if "grid_size" not in region_config:
                    raise ValueError(f"'{region_name}' 영역에 'grid_size' [H, W] 설정이 필요합니다.")
                
                grid_height, grid_width = region_config["grid_size"]
                node_grid_sizes[region_name] = (grid_height, grid_width)
                
                # SpikeNode 생성
                nodes[region_name] = SpikeNode(
                    grid_height=grid_height,
                    grid_width=grid_width,
                    decay_rate=region_config["decay_rate"],
                    spike_threshold=config["spike_dynamics"]["threshold"],
                    refractory_base=config["spike_dynamics"]["refractory_base"],
                    refractory_adaptive_factor=config["spike_dynamics"]["refractory_adaptive_factor"],
                    surrogate_beta=config["spike_dynamics"]["surrogate_beta"],
                    ema_alpha=config["spike_dynamics"]["ema_alpha"],
                    device=device
                )
                
                # LocalConnectivity 생성
                local_connections[region_name] = LocalConnectivity(
                    grid_height=grid_height,
                    grid_width=grid_width,
                    distance_tau=region_config["distance_tau"],
                    max_distance=config["connectivity"]["local"]["max_distance"],
                    device=device
                )
            
            # --- 단계 2: 축삭 연결 객체 생성 ---
            if "axonal_connections" not in config:
                raise ValueError("Config 파일에 'axonal_connections' 섹션이 필요합니다.")

            axonal_config = config["axonal_connections"]
            axonal_connections = AxonalConnections(
                connections=axonal_config.get("connections", []),  # 연결이 없는 경우도 처리
                node_grid_sizes=node_grid_sizes,  # 🔧 추가: 그리드 크기 정보 전달
                device=device
            )
            
            # --- 단계 3: 입출력 인터페이스 객체 생성 ---
            if "system_roles" not in config:
                raise ValueError("Config 파일에 'system_roles' 섹션(input_node, output_node)이 필요합니다.")
            
            input_node_name = config["system_roles"]["input_node"]
            output_node_name = config["system_roles"]["output_node"]
            
            if input_node_name not in node_grid_sizes:
                raise ValueError(f"system_roles의 input_node '{input_node_name}'이 brain_regions에 정의되지 않았습니다.")
            if output_node_name not in node_grid_sizes:
                raise ValueError(f"system_roles의 output_node '{output_node_name}'이 brain_regions에 정의되지 않았습니다.")
            
            input_h, input_w = node_grid_sizes[input_node_name]
            output_h, output_w = node_grid_sizes[output_node_name]

            tokenizer_config = config["data_loading"]["tokenizer"]
            pad_token_id = tokenizer_config.get("pad_token_id") or \
                        config.get("io_system", {}).get("output_interface", {}).get("pad_token_id", 0)
            eos_token_id = tokenizer_config.get("eos_token_id", 1)
            
            io_config = config["io_system"]
            input_interface = InputInterface(
                vocab_size=io_config["input_interface"]["vocab_size"],
                grid_height=input_h,
                grid_width=input_w,
                embedding_dim=io_config["input_interface"].get("embedding_dim", 512),
                max_seq_len=io_config["input_interface"].get("max_seq_len", 128),
                num_heads=io_config["input_interface"].get("num_heads", 8),
                use_positional_encoding=io_config["input_interface"].get(
                    "use_positional_encoding", 
                    io_config["input_interface"].get("positional_encoding", True)
                ),
                t5_model_name=io_config["input_interface"].get("t5_model_name", "t5-base"),
                device=device
            )

            output_interface = OutputInterface(
                vocab_size=io_config["output_interface"]["vocab_size"],
                grid_height=output_h,
                grid_width=output_w,
                pad_token_id=pad_token_id,
                embedding_dim=io_config["output_interface"].get("embedding_dim", 256),
                max_output_len=io_config["output_interface"].get("max_output_len", 128),
                num_heads=io_config["output_interface"].get("num_heads", 4),
                num_decoder_layers=io_config["output_interface"].get("num_decoder_layers", 2),
                dim_feedforward=io_config["output_interface"].get("dim_feedforward", 1024),
                dropout=io_config["output_interface"].get("dropout", 0.1),
                use_positional_encoding=io_config["output_interface"].get(
                    "use_positional_encoding", 
                    io_config["output_interface"].get("positional_encoding", True)
                ),
                t5_model_name=io_config["output_interface"].get("t5_model_name", "t5-base"),
                device=device
            )
                        
            # --- 단계 4: TimingManager 객체 생성 ---
            timing_config = config.get("timing_manager", config.get("adaptive_output_timing", config.get("timing", {})))
            timing_manager = TimingManager(
                sync_ema_alpha=timing_config.get("sync_ema_alpha", 0.1),
                sync_threshold_start=timing_config.get("sync_threshold_start", 0.6),
                sync_threshold_end=timing_config.get("sync_threshold_end", 0.2),
                min_processing_clk=timing_config.get("min_processing_clk", 50),
                max_processing_clk=timing_config.get("max_processing_clk", 500),
                min_output_length=timing_config.get("min_output_length", 5),
                fixed_len=timing_config.get("fixed_len", -1),
                fixed_delay=timing_config.get("fixed_delay", -1)
            )
            
            # --- 단계 5: 최종 SCS 시스템 조립 ---
            scs_system = SCSSystem(
                nodes=nodes,
                local_connections=local_connections,
                axonal_connections=axonal_connections,
                input_interface=input_interface,
                output_interface=output_interface,
                timing_manager=timing_manager,
                input_node=input_node_name,
                output_node=output_node_name,
                acc_node=config['system_roles'].get('acc_node', 'ACC'),
                eos_token_id=eos_token_id,
                device=device
            )
            
            return scs_system.to(device)
            
        except KeyError as e:
            raise KeyError(f"Config 파일에서 필수 키를 찾을 수 없습니다: {e}") from e
        except Exception as e:
            raise RuntimeError(f"SCS 모델 생성 중 오류 발생: {e}") from e
    
    @staticmethod
    def validate_config_structure(config: Dict[str, Any]) -> List[str]:
        """
        설정 파일의 구조 유효성 검사
        
        Args:
            config: 설정 딕셔너리
            
        Returns:
            오류 메시지 리스트 (빈 리스트면 유효함)
        """
        errors = []
        
        # 필수 최상위 섹션 확인
        required_sections = [
            "system_roles", "brain_regions", "axonal_connections", 
            "spike_dynamics", "connectivity", "io_system"
        ]
        
        for section in required_sections:
            if section not in config:
                errors.append(f"필수 섹션 '{section}'이 누락되었습니다.")
        
        # system_roles 구조 확인
        if "system_roles" in config:
            if "input_node" not in config["system_roles"]:
                errors.append("system_roles에 'input_node'가 필요합니다.")
            if "output_node" not in config["system_roles"]:
                errors.append("system_roles에 'output_node'가 필요합니다.")
        
        # brain_regions 구조 확인
        if "brain_regions" in config:
            for region_name, region_config in config["brain_regions"].items():
                if "grid_size" not in region_config:
                    errors.append(f"'{region_name}' 영역에 'grid_size' [H, W]가 필요합니다.")
                elif not isinstance(region_config["grid_size"], list) or len(region_config["grid_size"]) != 2:
                    errors.append(f"'{region_name}' 영역의 'grid_size'는 [height, width] 형태의 배열이어야 합니다.")
                
                if "decay_rate" not in region_config:
                    errors.append(f"'{region_name}' 영역에 'decay_rate'가 필요합니다.")
                if "distance_tau" not in region_config:
                    errors.append(f"'{region_name}' 영역에 'distance_tau'가 필요합니다.")
        
        # axonal_connections 구조 확인
        if "axonal_connections" in config:
            if "connections" not in config["axonal_connections"]:
                errors.append("axonal_connections에 'connections' 리스트가 필요합니다.")
            else:
                for i, conn in enumerate(config["axonal_connections"]["connections"]):
                    required_conn_fields = ["source", "target", "kernel_size", "weight_scale"]
                    for field in required_conn_fields:
                        if field not in conn:
                            errors.append(f"연결 {i}에 '{field}' 필드가 필요합니다.")
        
        # 노드 참조 무결성 확인
        if "system_roles" in config and "brain_regions" in config:
            available_nodes = set(config["brain_regions"].keys())
            
            input_node = config["system_roles"].get("input_node")
            output_node = config["system_roles"].get("output_node")
            
            if input_node and input_node not in available_nodes:
                errors.append(f"input_node '{input_node}'이 brain_regions에 정의되지 않았습니다.")
            if output_node and output_node not in available_nodes:
                errors.append(f"output_node '{output_node}'이 brain_regions에 정의되지 않았습니다.")
            
            # 축삭 연결의 노드 참조 확인
            if "axonal_connections" in config:
                for i, conn in enumerate(config["axonal_connections"]["connections"]):
                    source = conn.get("source")
                    target = conn.get("target")
                    
                    if source and source not in available_nodes:
                        errors.append(f"연결 {i}의 source '{source}'가 brain_regions에 정의되지 않았습니다.")
                    if target and target not in available_nodes:
                        errors.append(f"연결 {i}의 target '{target}'가 brain_regions에 정의되지 않았습니다.")
        
         # 타이밍 섹션 검증 추가
        has_timing = "timing" in config
        has_adaptive_timing = "adaptive_output_timing" in config
        has_timing_manager = "timing_manager" in config
        if not (has_timing or has_adaptive_timing or has_timing_manager):
            errors.append("'timing', 'adaptive_output_timing', 또는 'timing_manager' 섹션이 필요합니다.")
        
        # IO System 상세 검증 추가
        if "io_system" in config:
            io_config = config["io_system"]
            
            # input_interface 필수 필드 검증
            if "input_interface" in io_config:
                input_required = ["embedding_dim", "max_seq_len", "num_heads"]
                for field in input_required:
                    if field not in io_config["input_interface"]:
                        errors.append(f"io_system.input_interface에 '{field}' 필드가 필요합니다.")
            else:
                errors.append("io_system에 'input_interface' 섹션이 필요합니다.")
            
            # output_interface 필수 필드 검증  
            if "output_interface" in io_config:
                output_required = ["embedding_dim", "max_output_len", "num_heads", "num_decoder_layers"]
                for field in output_required:
                    if field not in io_config["output_interface"]:
                        errors.append(f"io_system.output_interface에 '{field}' 필드가 필요합니다.")
            else:
                errors.append("io_system에 'output_interface' 섹션이 필요합니다.")
        
        # 학습 설정 검증 추가
        learning_config = config.get("learning", config.get("training", {}))
        if learning_config:
            # 필수 학습 파라미터 확인
            required_learning = ["epochs", "max_clk_training"]
            learning_rate_variants = ["learning_rate", "base_learning_rate"]
            
            for field in required_learning:
                if field not in learning_config:
                    errors.append(f"학습 설정에 '{field}' 필드가 필요합니다.")
            
            # learning_rate 또는 base_learning_rate 중 하나는 있어야 함
            if not any(variant in learning_config for variant in learning_rate_variants):
                errors.append("학습 설정에 'learning_rate' 또는 'base_learning_rate' 필드가 필요합니다.")
        
        return errors