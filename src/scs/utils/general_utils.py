# utils/general_utils.py
"""일반 유틸리티"""

import random
import numpy as np
from typing import Union


def set_random_seed(seed: int) -> None:
    """재현 가능한 결과를 위한 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(device: str = "auto") -> str:
    """사용할 연산 장치 결정"""
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    else:
        return device


def format_time(seconds: float) -> str:
    """초를 읽기 쉬운 시간 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}분 {secs:.1f}초"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}시간 {int(minutes)}분"


class ModelBuilder:
    """SCS 모델 생성을 위한 빌더 클래스"""
    
    @staticmethod
    def build_scs_from_config(config: dict, device: str = "cpu"):
        """설정 파일로부터 SCS 시스템을 생성합니다"""
        try:
            # 필요한 모듈들을 동적으로 import
            from ..architecture import (
                SpikeNode, LocalConnectivity, AxonalConnections, 
                MultiScaleGrid, InputInterface, OutputInterface, 
                SCSSystem, AdaptiveOutputTiming
            )
            import torch.nn as nn
            
            # 1. 노드 생성
            nodes = {}
            local_connections = {}
            node_grid_sizes = {}
            
            for region_name, region_config in config["brain_regions"].items():
                grid_height = region_config["grid_height"]
                grid_width = region_config["grid_width"]
                node_grid_sizes[region_name] = (grid_height, grid_width)
                
                # SpikeNode 생성
                nodes[region_name] = SpikeNode(
                    grid_height=grid_height,
                    grid_width=grid_width,
                    spike_threshold=config["spike_dynamics"]["threshold"],  # threshold -> spike_threshold
                    refractory_base=config["spike_dynamics"]["refractory_base"],
                    refractory_adaptive_factor=config["spike_dynamics"]["refractory_adaptive_factor"],
                    decay_rate=region_config["decay_rate"],
                    # distance_tau는 SpikeNode에 없음 - LocalConnectivity용으로 보임
                    surrogate_beta=config["spike_dynamics"]["surrogate_beta"],
                    ema_alpha=config["spike_dynamics"]["ema_alpha"],
                    device=device
                )
                
                # LocalConnectivity 생성
                local_connections[region_name] = LocalConnectivity(
                    grid_height=grid_height,
                    grid_width=grid_width,
                    distance_tau=region_config["distance_tau"],  # distance_tau는 LocalConnectivity에서 사용
                    max_distance=config["connectivity"]["local"]["max_distance"],
                    # connection_prob는 LocalConnectivity에서 사용하지 않음
                    device=device
                )
            
            # 2. 축삭 연결 생성
            connection_pairs = []
            for conn in config["connectivity"]["axonal"]["connections"]:
                connection_pairs.append((conn["source"], conn["target"], conn["weight_scale"]))
            
            axonal_connections = AxonalConnections(
                connection_pairs=connection_pairs,
                node_grid_sizes=node_grid_sizes,
                excitatory_ratio=config["connectivity"]["axonal"]["excitatory_ratio"],
                device=device
            )
            
            # 3. Multi-scale Grid 연결 생성
            multi_scale_config = config["connectivity"]["multi_scale_grid"]
            multi_scale_grid = MultiScaleGrid(
                node_list=list(node_grid_sizes.keys()),  # node_grid_sizes -> node_list
                fine_spacing=multi_scale_config["fine"]["spacing"],
                fine_weight=multi_scale_config["fine"]["weight"],
                medium_spacing=multi_scale_config["medium"]["spacing"],
                medium_weight=multi_scale_config["medium"]["weight"],
                coarse_spacing=multi_scale_config["coarse"]["spacing"],
                coarse_weight=multi_scale_config["coarse"]["weight"],
                device=device
            )
            
            # 4. 입출력 인터페이스 생성
            io_config = config["io_system"]
            
            # 입출력 노드 결정 (config에서 설정하거나 기본값 사용)
            input_node = config.get("input_node", "PFC")
            output_node = config.get("output_node", "PFC")
            
            if input_node not in node_grid_sizes:
                raise ValueError(f"Input node '{input_node}'이 brain_regions에 정의되지 않음")
            if output_node not in node_grid_sizes:
                raise ValueError(f"Output node '{output_node}'이 brain_regions에 정의되지 않음")
            
            input_interface = InputInterface(
                vocab_size=io_config["input_interface"]["vocab_size"],
                grid_height=node_grid_sizes[input_node][0],
                grid_width=node_grid_sizes[input_node][1],
                embedding_dim=io_config["input_interface"]["embedding_dim"],
                max_seq_len=io_config["input_interface"]["max_seq_len"],
                num_heads=io_config["input_interface"]["num_heads"],
                use_positional_encoding=io_config["input_interface"]["positional_encoding"],  # positional_encoding -> use_positional_encoding
                device=device
            )
            
            output_interface = OutputInterface(
                vocab_size=io_config["output_interface"]["vocab_size"],
                grid_height=node_grid_sizes[output_node][0],
                grid_width=node_grid_sizes[output_node][1],
                embedding_dim=io_config["output_interface"]["embedding_dim"],
                max_output_len=io_config["output_interface"]["max_output_len"],
                num_heads=io_config["output_interface"]["num_heads"],
                num_decoder_layers=io_config["output_interface"]["num_decoder_layers"],
                dim_feedforward=io_config["output_interface"]["dim_feedforward"],
                dropout=io_config["output_interface"]["dropout"],
                device=device
            )
            
            # 5. 적응적 타이밍 생성
            timing_config = config["timing"]
            output_timing = AdaptiveOutputTiming(
                min_processing_clk=timing_config["min_processing_clk"],
                max_processing_clk=timing_config["max_processing_clk"],
                convergence_threshold=timing_config["convergence_threshold"],
                confidence_threshold=timing_config["confidence_threshold"],
                stability_window=timing_config["stability_window"],
                start_output_threshold=timing_config["start_output_threshold"]
            )
            
            # 6. SCS 시스템 생성
            scs_system = SCSSystem(
                nodes=nodes,
                local_connections=local_connections,
                axonal_connections=axonal_connections,
                multi_scale_grid=multi_scale_grid,
                input_interface=input_interface,
                output_interface=output_interface,
                output_timing=output_timing,
                input_node=input_node,   # config에서 결정된 입력 노드
                output_node=output_node, # config에서 결정된 출력 노드
                device=device
            )
            
            return scs_system.to(device)
            
        except Exception as e:
            raise RuntimeError(f"SCS 모델 생성 중 오류 발생: {e}") from e