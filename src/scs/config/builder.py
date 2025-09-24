# src/scs/config/builder.py
"""
SCS 모델 빌더 - Pydantic 설정 기반 모델 생성

검증된 설정 객체를 받아 동적으로 SCS 시스템을 조립합니다.
"""

from typing import Any
import torch.nn as nn

from .schemas import AppConfig


class ModelBuilder:
    """SCS 모델 생성을 위한 빌더 클래스 - Pydantic 설정 기반"""
    
    @staticmethod
    def build_model(config: AppConfig, device: str = "cpu"):
        """
        검증된 AppConfig로부터 SCS 모델 생성
        
        Args:
            config: 검증된 AppConfig 객체
            device: 모델이 위치할 디바이스
            
        Returns:
            SCSSystem: 완전히 구성된 SCS 시스템
        """
        try:
            # 필요한 모듈들을 동적으로 import
            from ..architecture import (
                SpikeNode, LocalConnectivity, AxonalConnections,
                InputInterface, OutputInterface, SCSSystem
            )
            from ..architecture.timing import TimingManager
            
            # 노드 및 로컬 연결 생성
            nodes = {}
            local_connections = {}
            node_grid_sizes = {}
            
            for region_name, region_config in config.brain_regions.items():
                grid_height, grid_width = region_config.grid_size
                node_grid_sizes[region_name] = (grid_height, grid_width)
                
                nodes[region_name] = SpikeNode(
                    grid_height=grid_height,
                    grid_width=grid_width,
                    decay_rate=region_config.decay_rate,
                    refractory_damping_factor=config.spike_dynamics.refractory_damping_factor,
                    spike_threshold=config.spike_dynamics.spike_threshold,
                    refractory_base=config.spike_dynamics.refractory_base,
                    refractory_adaptive_factor=config.spike_dynamics.refractory_adaptive_factor,
                    surrogate_beta=config.spike_dynamics.surrogate_beta,
                    ema_alpha=config.spike_dynamics.ema_alpha,
                    device=device
                )
                
                local_connections[region_name] = LocalConnectivity(
                    grid_height=grid_height,
                    grid_width=grid_width,
                    local_distance=config.connectivity.local_distance,
                    tau_D=config.connectivity.tau_D,
                    tau_F=config.connectivity.tau_F,
                    U=config.connectivity.U,
                    excitatory_ratio=config.connectivity.excitatory_ratio,
                    connection_sigma=config.connectivity.connection_sigma,
                    weight_mean=config.connectivity.weight_mean,
                    weight_std=config.connectivity.weight_std,
                    device=device
                )
            
            # 축삭 연결 생성
            connection_list = []
            for conn_config in config.axonal_connections.connections:
                connection_list.append({
                    "source": conn_config.source,
                    "target": conn_config.target,
                    "patch_size": conn_config.patch_size,
                    "patch_weight_scale": conn_config.patch_weight_scale,
                    "inner_weight_scale": conn_config.inner_weight_scale
                })
            
            axonal_connections = AxonalConnections(
                connections=connection_list,
                node_grid_sizes=node_grid_sizes,
                gate_init_mean=config.axonal_connections.gate_init_mean,
                gate_init_std=config.axonal_connections.gate_init_std,
                bias_init_mean=config.axonal_connections.bias_init_mean,
                bias_init_std=config.axonal_connections.bias_init_std,
                axon_temperature=config.axonal_connections.axon_temperature,
                tau_pre=config.axonal_connections.tau_pre,
                tau_post=config.axonal_connections.tau_post,
                A_plus=config.axonal_connections.A_plus,
                A_minus=config.axonal_connections.A_minus,
                device=device
            )
            
            # I/O 인터페이스 생성
            input_node_name = config.system_roles.input_node
            output_node_name = config.system_roles.output_node
            
            input_h, input_w = node_grid_sizes[input_node_name]
            output_h, output_w = node_grid_sizes[output_node_name]
            
            # 토크나이저 설정에서 토큰 ID들 추출
            pad_token_id = config.data_loading.tokenizer.pad_token_id
            eos_token_id = config.data_loading.tokenizer.eos_token_id
            
            io_config = config.io_system
            
            # InputInterface 생성
            input_interface = InputInterface(
                vocab_size=io_config.input_interface.vocab_size or 32128,  # 기본값
                grid_height=input_h,
                grid_width=input_w,
                embedding_dim=io_config.input_interface.embedding_dim,
                window_size=io_config.input_interface.window_size,
                encoder_layers=io_config.input_interface.encoder_layers,
                encoder_heads=io_config.input_interface.encoder_heads,
                encoder_dropout=io_config.input_interface.encoder_dropout,
                dim_feedforward=io_config.input_interface.dim_feedforward,
                input_power=io_config.input_interface.input_power,
                softmax_temperature=io_config.input_interface.softmax_temperature,
                t5_model_name=io_config.input_interface.t5_model_name,
                device=device
            )

            # OutputInterface 생성
            output_interface = OutputInterface(
                vocab_size=io_config.output_interface.vocab_size or 32128,  # 기본값
                grid_height=output_h,
                grid_width=output_w,
                pad_token_id=pad_token_id,
                embedding_dim=io_config.output_interface.embedding_dim,
                window_size=io_config.output_interface.window_size,
                decoder_layers=io_config.output_interface.decoder_layers,
                decoder_heads=io_config.output_interface.decoder_heads,
                dim_feedforward=io_config.output_interface.dim_feedforward,
                dropout=io_config.output_interface.dropout,
                t5_model_name=io_config.output_interface.t5_model_name,
                transplant_cross_attention=io_config.output_interface.transplant_cross_attention,
                device=device
            )
                            
            # TimingManager 생성
            timing_config = config.timing_manager
            timing_manager = TimingManager(
                train_fixed_ref=timing_config.train_fixed_ref,
                train_fixed_offset=timing_config.train_fixed_offset,
                evaluate_fixed_ref=timing_config.evaluate_fixed_ref,
                evaluate_fixed_offset=timing_config.evaluate_fixed_offset,
                sync_ema_alpha=timing_config.sync_ema_alpha,
                sync_threshold_start=timing_config.sync_threshold_start,
                sync_threshold_end=timing_config.sync_threshold_end,
                min_processing_clk=timing_config.min_processing_clk,
                max_processing_clk=timing_config.max_processing_clk,
                min_output_length=timing_config.min_output_length
            )
            
            # SCSSystem 생성
            scs_system = SCSSystem(
                nodes=nodes,
                local_connections=local_connections,
                axonal_connections=axonal_connections,
                input_interface=input_interface,
                output_interface=output_interface,
                timing_manager=timing_manager,
                input_node=input_node_name,
                output_node=output_node_name,
                acc_node=config.system_roles.acc_node,
                eos_token_id=eos_token_id,
                device=device
            )
            
            return scs_system.to(device)
            
        except KeyError as e:
            raise KeyError(f"설정에서 필수 키를 찾을 수 없습니다: {e}") from e
        except Exception as e:
            raise RuntimeError(f"SCS 모델 생성 중 오류 발생: {e}") from e