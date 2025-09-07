# src/scs/config/schemas.py
"""
Pydantic 기반 설정 스키마 정의

모든 설정 구조를 타입 안전하게 정의하고 검증
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import warnings


class BrainRegionConfig(BaseModel):
    """뇌 영역 설정"""
    grid_size: List[int] = Field(description="[height, width] 형태의 그리드 크기")
    decay_rate: float = Field(default=0.8, description="막전위 감쇠율")
    distance_tau: float = Field(default=1.0, description="거리 기반 연결 강도")
    
    @field_validator('grid_size')
    @classmethod
    def validate_grid_size(cls, v):
        if len(v) != 2:
            raise ValueError("grid_size는 [height, width] 형태여야 합니다")
        if any(size <= 0 for size in v):
            raise ValueError("grid_size의 모든 값은 양수여야 합니다")
        return v


class SystemRolesConfig(BaseModel):
    """시스템 역할 정의"""
    input_node: str = Field(description="입력 노드 이름")
    output_node: str = Field(description="출력 노드 이름")
    acc_node: str = Field(default="ACC", description="ACC 노드 이름")


class SpikeDynamicsConfig(BaseModel):
    """스파이크 역학 설정"""
    refractory_damping_factor: float = Field(default=0.2)
    spike_threshold: float = Field(default=1.0)
    refractory_base: int = Field(default=1)
    refractory_adaptive_factor: float = Field(default=5.0)
    surrogate_beta: float = Field(default=3.0)
    ema_alpha: float = Field(default=0.1)
    influence_init_mean: float = Field(default=1.0)
    influence_init_std: float = Field(default=0.01)
    excitatory_ratio: float = Field(default=1.0)
    gate_init_mean: float = Field(default=1.0)
    gate_init_std: float = Field(default=0.03)
    transform_init_mean: float = Field(default=1.0)
    transform_init_std: float = Field(default=0.03)


class ConnectivityConfig(BaseModel):
    """연결성 설정"""
    local: Dict[str, int] = Field(default={"max_distance": 5})


class AxonalConnectionConfig(BaseModel):
    """개별 축삭 연결 설정"""
    source: str
    target: str
    patch_size: int = Field(default=4)
    patch_weight_scale: float = Field(default=1.0)
    inner_weight_scale: float = Field(default=1.0)


class AxonalConnectionsConfig(BaseModel):
    """축삭 연결들 설정"""
    connections: List[AxonalConnectionConfig]


class InputInterfaceConfig(BaseModel):
    """입력 인터페이스 설정"""
    vocab_size: Optional[int] = None  # 런타임에 설정됨
    embedding_dim: int = Field(default=512)
    window_size: int = Field(default=32)
    encoder_layers: int = Field(default=6)
    encoder_heads: int = Field(default=8)
    encoder_dropout: float = Field(default=0.1)
    dim_feedforward: int = Field(default=2048)
    input_power: float = Field(default=0.5)
    softmax_temperature: float = Field(default=1.0)
    t5_model_name: str = Field(default="t5-base")


class OutputInterfaceConfig(BaseModel):
    """출력 인터페이스 설정"""
    vocab_size: Optional[int] = None  # 런타임에 설정됨
    embedding_dim: int = Field(default=512)
    window_size: int = Field(default=32)
    decoder_layers: int = Field(default=6)
    decoder_heads: int = Field(default=8)
    dim_feedforward: int = Field(default=2048)
    dropout: float = Field(default=0.1)
    t5_model_name: str = Field(default="t5-base")
    transplant_cross_attention: bool = Field(default=False)


class IOSystemConfig(BaseModel):
    """IO 시스템 설정"""
    input_interface: InputInterfaceConfig
    output_interface: OutputInterfaceConfig


class TimingManagerConfig(BaseModel):
    """타이밍 매니저 설정"""
    train_fixed_ref: str = Field(default='end')        # 'start' or 'end'
    train_fixed_offset: int = Field(default=0)
    evaluate_fixed_ref: str = Field(default='adaptive')  # 'start' or 'end' or 'adaptive'
    evaluate_fixed_offset: int = Field(default=0)
    sync_ema_alpha: float = Field(default=0.1)
    sync_threshold_start: float = Field(default=0.6)
    sync_threshold_end: float = Field(default=0.2)
    min_processing_clk: int = Field(default=50)
    max_processing_clk: int = Field(default=500)
    min_output_length: int = Field(default=5)


class GradualUnfreezingConfig(BaseModel):
    """점진적 해제 설정"""
    enabled: bool = Field(default=False)
    initial_frozen_patterns: List[str] = Field(default_factory=list)
    unfreeze_schedule: Dict[int, List[str]] = Field(default_factory=dict)
    freeze_schedule: Dict[int, List[str]] = Field(default_factory=dict)


class LearningConfig(BaseModel):
    """학습 설정"""
    epochs: int = Field(default=15)
    learning_rate: float = Field(default=1e-3)
    weight_decay: float = Field(default=1e-4)
    gradient_clip_norm: float = Field(default=1.0)
    eval_every: int = Field(default=3)
    save_every: int = Field(default=10)
    early_stopping_patience: int = Field(default=20)
    max_clk_training: int = Field(default=250)
    optimizer: str = Field(default="adamw")
    device: str = Field(default="cuda")
    pad_token_id: int = Field(default=0)
    
    # Scheduled Sampling
    use_scheduled_sampling: bool = Field(default=False)
    ss_start_prob: float = Field(default=1.0)
    ss_end_prob: float = Field(default=0.05)
    ss_decay_epochs: int = Field(default=10)
    eta_min: float = Field(default=0.0)

    gate_pruning_weight: float = Field(default=0.0)
    gate_temperature: float = Field(default=0.1)
    inner_pruning_weight: float = Field(default=0.0)
    inner_temperature: float = Field(default=0.1)
    axon_strength_reg_weight: float = Field(default=0.0)

    # 커리큘럼 학습
    use_curriculum_learning: bool = Field(default=False)
    curriculum_schedule: Optional[Dict[int, int]] = Field(default=None)
    
    # 손실 관련
    length_penalty_weight: float = Field(default=0.0)
    orthogonal_reg_weight: float = Field(default=0.0)

    spike_reg_weight: float = Field(default=0.0)
    target_spike_rate: float = Field(default=0.0)
    
    # 시간적 가중치
    use_temporal_weighting: bool = Field(default=False)
    initial_temporal_weight: float = Field(default=2.0)
    final_temporal_weight: float = Field(default=1.0)

    guide_sep_token_id: int = Field(default=32141)
    guide_weight: float = Field(default=0.3)

    # 타이밍 손실
    timing_weight: float = Field(default=1.0)
    sync_target_start: float = Field(default=1.0)
    sync_target_end: float = Field(default=0.0)
    
    # 점진적 해제
    gradual_unfreezing: GradualUnfreezingConfig = Field(default_factory=GradualUnfreezingConfig)


class DataLoaderConfig(BaseModel):
    """DataLoader 설정"""
    batch_size: int = Field(default=8)
    num_workers: int = Field(default=0)
    pin_memory: bool = Field(default=True)
    shuffle: bool = Field(default=True)
    drop_last: bool = Field(default=False)


class CheckpointConfig(BaseModel):
    """체크포인트 저장 설정"""
    save_dir: str = Field(default="./checkpoints")
    save_best_only: bool = Field(default=True)
    save_last: bool = Field(default=True)
    monitor: str = Field(default="val_loss")
    mode: str = Field(default="min")
    compress: bool = Field(default=False)
    max_checkpoints: Optional[int] = Field(default=5)

class TensorBoardConfig(BaseModel):
    """TensorBoard 설정"""
    enabled: bool = Field(default=False, description="TensorBoard 로깅 활성화")
    log_dir: str = Field(default="tensorboard_logs", description="TensorBoard 로그 디렉토리")
    log_interval: Dict[str, int] = Field(default={
        "scalars": 1,       # 매 배치
        "histograms": 100,  # 100배치마다
        "images": 500,      # 500배치마다
        "axonal_heatmaps": 200,  # 200배치마다
        "weight_heatmaps": 200
    }, description="로깅 간격 설정")
    auto_launch: bool = Field(default=False, description="TensorBoard 서버 자동 시작")
    port: int = Field(default=6006, description="TensorBoard 서버 포트")
    max_images_per_batch: int = Field(default=4, description="배치당 최대 이미지 수")
    histogram_freq: int = Field(default=100, description="히스토그램 로깅 빈도")

class LoggingConfig(BaseModel):
    """로깅 설정"""
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")
    log_file: Optional[str] = Field(default=None)
    log_every_n_steps: int = Field(default=10)
    log_gradients: bool = Field(default=False)
    log_weights: bool = Field(default=False)
    use_tensorboard: bool = Field(default=False)
    use_wandb: bool = Field(default=False)
    wandb_project: Optional[str] = Field(default=None)
    tensorboard: TensorBoardConfig = Field(default_factory=TensorBoardConfig, description="TensorBoard 설정")

class TokenizerConfig(BaseModel):
    """토크나이저 설정"""
    name: str = Field(default="t5-base")
    max_length: int = Field(default=128)
    pad_token_id: int = Field(default=0)
    eos_token_id: int = Field(default=1)
    bos_token_id: int = Field(default=1)
    unk_token_id: int = Field(default=2)


class DataLoadingConfig(BaseModel):
    """데이터 로딩 설정"""
    batch_size: int = Field(default=8)
    num_workers: int = Field(default=0)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)


class TaskConfig(BaseModel):
    """작업 설정"""
    name: Optional[str] = None
    dataset_name: str = Field(default="datatune/LogiQA2.0")
    task_type: str = Field(default="auto")
    task_id: int = Field(default=1)
    learning_style: str = Field(default="generative")
    bert_config: Optional[Dict[str, Any]] = None


class DataConfig(BaseModel):
    """데이터 설정"""
    guide_sep_token: str = Field(default="<extra_id_42>")
    train_samples: int = Field(default=-1)
    val_samples: int = Field(default=-1) 
    test_samples: int = Field(default=-1)


class EvaluationConfig(BaseModel):
    """평가 설정"""
    save_examples: int = Field(default=10)


class ExperimentConfig(BaseModel):
    """실험 설정 (선택적)"""
    name: Optional[str] = None
    description: Optional[str] = None
    phase: Optional[int] = None
    expected_runtime: Optional[str] = None


class AppConfig(BaseModel):
    """전체 애플리케이션 설정"""
    # 필수 섹션들
    system_roles: SystemRolesConfig
    brain_regions: Dict[str, BrainRegionConfig]
    axonal_connections: AxonalConnectionsConfig
    spike_dynamics: SpikeDynamicsConfig
    connectivity: ConnectivityConfig
    io_system: IOSystemConfig
    
    # 타이밍 관련
    timing_manager: TimingManagerConfig = Field(default_factory=TimingManagerConfig)
    
    # 학습 관련
    learning: LearningConfig = Field(default_factory=LearningConfig)
    
    # 데이터 관련
    data_loading: DataLoadingConfig = Field(default_factory=DataLoadingConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    
    # 체크포인트 및 로깅
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # 선택적 섹션들
    experiment: Optional[ExperimentConfig] = None
    
    # 최상위 레벨 호환성 필드들
    dataset_name: Optional[str] = None  # task.dataset_name으로 이동
    
    @model_validator(mode='after')
    def validate_task_dataset_name(self):
        """최상위 dataset_name을 task.dataset_name으로 이동"""
        if self.dataset_name and not self.task.dataset_name:
            self.task.dataset_name = self.dataset_name
        return self
    
    def validate_node_references(self):
        """노드 참조 무결성 검사"""
        available_nodes = set(self.brain_regions.keys())
        errors = []
        
        # 시스템 역할 노드 검사
        if self.system_roles.input_node not in available_nodes:
            errors.append(f"input_node '{self.system_roles.input_node}'이 brain_regions에 정의되지 않았습니다")
        if self.system_roles.output_node not in available_nodes:
            errors.append(f"output_node '{self.system_roles.output_node}'이 brain_regions에 정의되지 않았습니다")
        if self.system_roles.acc_node not in available_nodes:
            errors.append(f"acc_node '{self.system_roles.acc_node}'이 brain_regions에 정의되지 않았습니다")
        
        # 축삭 연결 노드 검사
        for i, conn in enumerate(self.axonal_connections.connections):
            if conn.source not in available_nodes:
                errors.append(f"연결 {i}의 source '{conn.source}'가 brain_regions에 정의되지 않았습니다")
            if conn.target not in available_nodes:
                errors.append(f"연결 {i}의 target '{conn.target}'가 brain_regions에 정의되지 않았습니다")
        
        if errors:
            raise ValueError("노드 참조 오류:\n" + "\n".join(f"  - {error}" for error in errors))
        
        return True
    
    class Config:
        extra = "forbid"  # 정의되지 않은 필드 허용 안함
        validate_assignment = True  # 할당 시에도 검증
        use_enum_values = True