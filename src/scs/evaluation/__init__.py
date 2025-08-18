# src/scs/evaluation/__init__.py
"""
SCS 평가 및 분석 패키지

모델 성능 평가, 시각화, 내부 동작 분석 기능 제공
"""

from .metrics import (
    SCSMetrics,
    SCSOptimizer,
    KHopBackpropagation,
    AdaptiveLearningRateScheduler,
    OptimizerFactory
)
from .visualizer import generate_visualizations
from .analyzer import analyze_io_pipeline

__all__ = [
    # 메트릭 및 최적화
    "SCSMetrics",
    "SCSOptimizer", 
    "KHopBackpropagation",
    "AdaptiveLearningRateScheduler",
    "OptimizerFactory",
    
    # 시각화 및 분석
    "generate_visualizations",
    "analyze_io_pipeline"
]