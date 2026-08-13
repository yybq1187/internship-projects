"""教学型水平联邦直方图 XGBoost 公共接口。"""

from horizontal_xgb.evaluator import evaluate_binary_classification, predict_proba
from horizontal_xgb.model_io import load_model, save_model
from horizontal_xgb.tree import HorizontalXGBModel

__all__ = [
    "HorizontalXGBModel",
    "evaluate_binary_classification",
    "load_model",
    "predict_proba",
    "save_model",
]
