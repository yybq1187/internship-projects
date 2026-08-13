"""实现仅在 Alice/Bob PYU 内执行的客户端本地计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from horizontal_xgb.binning import BinningModel, local_feature_extrema, transform_with_binning
from horizontal_xgb.histogram import build_histogram, build_leaf_statistics
from horizontal_xgb.objective import (
    binary_logistic_grad_hess,
    binary_logloss_from_raw_score,
)


@dataclass
class ClientState:
    """保存在单个客户端 PYU 内、不会发送给 Charlie 的逐样本状态。"""

    features: NDArray[np.float64]
    labels: NDArray[np.float64]
    binned_features: NDArray[np.int64]
    raw_score: NDArray[np.float64]
    gradients: NDArray[np.float64]
    hessians: NDArray[np.float64]
    node_assignments: NDArray[np.int64]


def compute_local_extrema(features: ArrayLike) -> NDArray[np.float64]:
    """在客户端本地计算允许向 Charlie 披露的逐特征 min/max。"""
    return local_feature_extrema(features)


def initialize_client_state(
    features: ArrayLike,
    labels: ArrayLike,
    binning_payload: Mapping[str, Any],
    feature_names: Sequence[str],
    base_score_raw: float,
) -> ClientState:
    """在客户端本地分桶，并初始化逐样本预测和节点归属。"""
    values = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or y.shape != (values.shape[0],):
        raise ValueError("客户端特征和标签形状不一致。")
    model = BinningModel.from_dict(dict(binning_payload))
    binned = transform_with_binning(values, model, feature_names)
    raw_score = np.full(values.shape[0], float(base_score_raw), dtype=np.float64)
    gradients, hessians = binary_logistic_grad_hess(raw_score, y)
    return ClientState(
        features=values,
        labels=y,
        binned_features=binned,
        raw_score=raw_score,
        gradients=gradients,
        hessians=hessians,
        node_assignments=np.zeros(values.shape[0], dtype=np.int64),
    )


def start_boosting_round(state: ClientState) -> ClientState:
    """根据当前 raw_score 重新计算本轮 g/h，并把所有样本放回根节点。"""
    gradients, hessians = binary_logistic_grad_hess(state.raw_score, state.labels)
    state.gradients = gradients
    state.hessians = hessians
    state.node_assignments = np.zeros(state.labels.shape[0], dtype=np.int64)
    return state


def build_local_histogram(
    state: ClientState,
    node_ids: Sequence[int],
    max_bin: int,
) -> NDArray[np.float64]:
    """构造固定形状的本地 count/G/H 直方图。"""
    return build_histogram(
        state.binned_features,
        state.gradients,
        state.hessians,
        state.node_assignments,
        node_ids,
        max_bin,
    )


def apply_public_splits(
    state: ClientState,
    split_payloads: Sequence[Mapping[str, Any]],
) -> ClientState:
    """根据 Charlie 公布的分裂结果在本地更新样本节点归属。"""
    original_assignments = state.node_assignments.copy()
    updated_assignments = state.node_assignments.copy()
    for payload in split_payloads:
        if bool(payload["is_leaf"]):
            continue
        node_id = int(payload["node_id"])
        feature_index = int(payload["feature_index"])
        split_bin = int(payload["split_bin"])
        left_id = int(payload["left_id"])
        right_id = int(payload["right_id"])
        mask = original_assignments == node_id
        go_left = state.binned_features[:, feature_index] <= split_bin
        updated_assignments[mask & go_left] = left_id
        updated_assignments[mask & ~go_left] = right_id
    state.node_assignments = updated_assignments
    return state


def build_local_leaf_statistics(
    state: ClientState,
    leaf_ids: Sequence[int],
) -> NDArray[np.float64]:
    """按公开叶子编号累加本地 count/G/H。"""
    return build_leaf_statistics(
        state.gradients,
        state.hessians,
        state.node_assignments,
        leaf_ids,
    )


def apply_leaf_weights(
    state: ClientState,
    leaf_ids: Sequence[int],
    weights: Sequence[float],
    learning_rate: float,
) -> ClientState:
    """在客户端本地执行 raw_score += learning_rate * leaf_weight。"""
    if len(leaf_ids) != len(weights):
        raise ValueError("叶子编号和权重数量不一致。")
    increments = np.zeros(state.labels.shape[0], dtype=np.float64)
    for leaf_id, weight in zip(leaf_ids, weights):
        increments[state.node_assignments == int(leaf_id)] = float(weight)
    state.raw_score = state.raw_score + float(learning_rate) * increments
    return state


def local_loss_statistics(state: ClientState) -> NDArray[np.float64]:
    """只返回可安全求和的损失总和与样本数，不返回逐样本损失。"""
    loss = binary_logloss_from_raw_score(state.raw_score, state.labels)
    return np.asarray([loss * state.labels.size, state.labels.size], dtype=np.float64)


def client_state_shape(state: ClientState) -> dict[str, int]:
    """返回不包含原始值的本地状态规模，供运行时审计记录。"""
    return {
        "sample_count": int(state.features.shape[0]),
        "feature_count": int(state.features.shape[1]),
    }


def predict_local_model(
    features: ArrayLike,
    feature_names: Sequence[str],
    model_payload: Mapping[str, Any],
) -> NDArray[np.float64]:
    """在客户端本地加载公开模型并返回最终评估概率。"""
    # 局部导入可以避免客户端训练状态模块与树模块形成循环依赖。
    from horizontal_xgb.tree import HorizontalXGBModel

    model = HorizontalXGBModel.from_dict(dict(model_payload))
    return model.predict_proba(features, feature_names)
