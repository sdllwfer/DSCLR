"""
FollowIR 评测系统
模块化、可配置的稠密检索模型评测框架
"""

from .engine import FollowIREvaluatorEngine, EvaluationRunner, FollowIRDataLoader
from .data import DataLoader
from .models import ModelFactory, DenseRetriever
from .metrics import DataLoader as MetricsDataLoader, FollowIREvaluator, MetricsRegistry
from .output import OutputManager

__version__ = "1.0.0"

__all__ = [
    'FollowIREvaluatorEngine',
    'EvaluationRunner',
    'FollowIRDataLoader',
    'DataLoader',
    'ModelFactory',
    'DenseRetriever',
    'MetricsDataLoader',
    'FollowIREvaluator',
    'MetricsRegistry',
    'OutputManager'
]
