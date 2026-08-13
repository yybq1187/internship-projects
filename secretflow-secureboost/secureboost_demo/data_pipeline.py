"""纵向联邦 SecureBoost 的本地数据校验、切分和联邦对象构造。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from secretflow.data import FedNdarray, PartitionWay

if TYPE_CHECKING:
    from secureboost_demo.devices import SecureBoostDevices


@dataclass(frozen=True)
class VerticalDataPartition:
    """一个训练集或测试集中的本地纵向切分结果。

    ``sample_ids`` 仅用于本地契约校验，不会被上传为模型特征。标签字段
    仅保存在本对象的 ``labels`` 中，Bob 特征表不包含标签列。
    """

    sample_ids: pd.Index
    alice_features: pd.DataFrame
    bob_features: pd.DataFrame
    labels: pd.DataFrame

    def __post_init__(self) -> None:
        """在数据进入联邦设备前检查单个分区的行数和标签结构。"""
        sample_count = len(self.sample_ids)
        if sample_count == 0:
            raise ValueError("训练集或测试集不能为空。")
        if self.sample_ids.has_duplicates:
            raise ValueError("单个数据分区中的 sample_id 不能重复。")
        if len(self.alice_features) != sample_count:
            raise ValueError("Alice 特征行数与 sample_id 数量不一致。")
        if len(self.bob_features) != sample_count:
            raise ValueError("Bob 特征行数与 sample_id 数量不一致。")
        if len(self.labels) != sample_count:
            raise ValueError("标签行数与 sample_id 数量不一致。")
        if self.labels.shape[1] != 1:
            raise ValueError("标签数据必须恰好包含一列。")

    @property
    def sample_count(self) -> int:
        """返回该训练集或测试集分区中的样本数。"""
        return len(self.sample_ids)


@dataclass(frozen=True)
class LocalVerticalDataset:
    """完成本地校验后的训练集和测试集纵向数据契约。"""

    train: VerticalDataPartition
    test: VerticalDataPartition

    def __post_init__(self) -> None:
        """检查训练/测试隔离，以及两方特征列的一致性和不重叠性。"""
        train_ids = set(self.train.sample_ids)
        test_ids = set(self.test.sample_ids)
        if train_ids & test_ids:
            raise ValueError("训练集与测试集存在重复 sample_id，发生了样本泄漏。")
        if (
            set(self.train.alice_features.columns)
            != set(self.test.alice_features.columns)
        ):
            raise ValueError("Alice 的训练集和测试集特征列必须完全一致。")
        if set(self.train.bob_features.columns) != set(self.test.bob_features.columns):
            raise ValueError("Bob 的训练集和测试集特征列必须完全一致。")
        if (
            set(self.train.alice_features.columns)
            & set(self.train.bob_features.columns)
        ):
            raise ValueError("Alice 与 Bob 的特征列不能重叠。")


@dataclass(frozen=True)
class FederatedVerticalDataset:
    """上传到 SecretFlow 设备后的纵向联邦数据对象。"""

    train_features: FedNdarray
    test_features: FedNdarray
    train_labels: FedNdarray
    test_labels: FedNdarray
    train_sample_count: int
    test_sample_count: int


def prepare_vertical_dataset(
    alice_table: pd.DataFrame,
    bob_table: pd.DataFrame,
    *,
    alice_feature_columns: Sequence[str],
    bob_feature_columns: Sequence[str],
    label_column: str,
    sample_id_column: str = "sample_id",
    test_size: float = 0.33,
    random_state: int = 42,
) -> LocalVerticalDataset:
    """将两方本地表转换为经过校验的训练/测试纵向数据。

    处理顺序固定为：校验两方数据 -> 按 Alice 的 sample_id 对齐 Bob ->
    只执行一次分层训练/测试切分 -> 按特征列拆分。两方绝不能各自独立
    随机切分，否则相同行号会指向不同样本，无法进行纵向联邦训练。
    """
    _validate_schema_and_roles(
        alice_table=alice_table,
        bob_table=bob_table,
        alice_feature_columns=alice_feature_columns,
        bob_feature_columns=bob_feature_columns,
        label_column=label_column,
        sample_id_column=sample_id_column,
    )
    _validate_sample_ids(alice_table, bob_table, sample_id_column)

    # 以 Alice 的样本顺序为基准重排 Bob，随后两方每一行代表同一个样本。
    alice_ids = alice_table[sample_id_column].reset_index(drop=True)
    bob_aligned = (
        bob_table.set_index(sample_id_column)
        .loc[alice_ids]
        .reset_index()
    )

    alice_features = _select_and_validate_features(
        alice_table, alice_feature_columns, "Alice"
    )
    bob_features = _select_and_validate_features(
        bob_aligned, bob_feature_columns, "Bob"
    )
    labels = _select_and_validate_binary_labels(alice_table, label_column)
    _validate_split_request(labels, test_size)

    all_indices = np.arange(len(alice_ids))
    try:
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels.to_numpy(),
        )
    except ValueError as error:
        raise ValueError(
            "无法按当前 test_size 完成分层切分；请增加每个类别的样本数或调整切分比例。"
        ) from error

    # 排序仅用于稳定输出行序；样本归属仍完全由同一次随机切分决定。
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)
    return LocalVerticalDataset(
        train=_build_partition(
            alice_ids, alice_features, bob_features, labels, train_indices
        ),
        test=_build_partition(
            alice_ids, alice_features, bob_features, labels, test_indices
        ),
    )


def build_federated_dataset(
    local_dataset: LocalVerticalDataset,
    devices: "SecureBoostDevices",
) -> FederatedVerticalDataset:
    """将已校验的本地数据上传到各自 PYU，并构造纵向 ``FedNdarray``。

    该函数只把 Alice 特征与标签发送到 Alice 的 PYU，只把 Bob 特征发送到
    Bob 的 PYU。``sample_id`` 仅用于本地对齐，绝不会参与模型训练或上传。
    """
    train_alice_features = devices.alice(_frame_to_float32_array)(
        local_dataset.train.alice_features
    )
    train_bob_features = devices.bob(_frame_to_float32_array)(
        local_dataset.train.bob_features
    )
    test_alice_features = devices.alice(_frame_to_float32_array)(
        local_dataset.test.alice_features
    )
    test_bob_features = devices.bob(_frame_to_float32_array)(
        local_dataset.test.bob_features
    )

    # 标签仅上传到 Alice；Bob 的标签分区在任何时刻都不存在。
    train_labels = devices.alice(_frame_to_float32_array)(local_dataset.train.labels)
    test_labels = devices.alice(_frame_to_float32_array)(local_dataset.test.labels)

    return FederatedVerticalDataset(
        train_features=FedNdarray(
            {devices.alice: train_alice_features, devices.bob: train_bob_features},
            partition_way=PartitionWay.VERTICAL,
        ),
        test_features=FedNdarray(
            {devices.alice: test_alice_features, devices.bob: test_bob_features},
            partition_way=PartitionWay.VERTICAL,
        ),
        train_labels=FedNdarray(
            {devices.alice: train_labels},
            partition_way=PartitionWay.VERTICAL,
        ),
        test_labels=FedNdarray(
            {devices.alice: test_labels},
            partition_way=PartitionWay.VERTICAL,
        ),
        train_sample_count=local_dataset.train.sample_count,
        test_sample_count=local_dataset.test.sample_count,
    )


def _validate_schema_and_roles(
    *,
    alice_table: pd.DataFrame,
    bob_table: pd.DataFrame,
    alice_feature_columns: Sequence[str],
    bob_feature_columns: Sequence[str],
    label_column: str,
    sample_id_column: str,
) -> None:
    """校验列定义，防止标签泄漏、保留字段误作特征和跨方特征重叠。"""
    if sample_id_column == label_column:
        raise ValueError("sample_id 列和标签列不能使用同一个列名。")
    _ensure_columns_exist(alice_table, [sample_id_column, label_column], "Alice")
    _ensure_columns_exist(bob_table, [sample_id_column], "Bob")
    if label_column in bob_table.columns:
        raise ValueError("Bob 的本地表中不能包含标签列。")

    alice_columns = _normalize_feature_columns(alice_feature_columns, "Alice")
    bob_columns = _normalize_feature_columns(bob_feature_columns, "Bob")
    reserved_columns = {sample_id_column, label_column}
    if set(alice_columns) & reserved_columns or set(bob_columns) & reserved_columns:
        raise ValueError("sample_id 和标签列不能被指定为模型特征。")
    if set(alice_columns) & set(bob_columns):
        raise ValueError("Alice 与 Bob 的特征列不能重叠。")
    _ensure_columns_exist(alice_table, alice_columns, "Alice")
    _ensure_columns_exist(bob_table, bob_columns, "Bob")


def _validate_sample_ids(
    alice_table: pd.DataFrame,
    bob_table: pd.DataFrame,
    sample_id_column: str,
) -> None:
    """校验 sample_id 非空、无重复且两方拥有完全相同的样本集合。"""
    alice_ids = alice_table[sample_id_column]
    bob_ids = bob_table[sample_id_column]
    if alice_ids.isna().any() or bob_ids.isna().any():
        raise ValueError("两方的 sample_id 均不能包含缺失值。")
    if alice_ids.duplicated().any() or bob_ids.duplicated().any():
        raise ValueError("两方的 sample_id 均不能重复。")
    if len(alice_ids) != len(bob_ids):
        raise ValueError("Alice 与 Bob 的样本数量不一致。")
    if set(alice_ids) != set(bob_ids):
        raise ValueError("Alice 与 Bob 的 sample_id 集合不一致。")


def _select_and_validate_features(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    party_name: str,
) -> pd.DataFrame:
    """提取指定方特征，并拒绝非数值、缺失或无穷的输入。"""
    features = table.loc[:, list(feature_columns)].reset_index(drop=True).copy()
    try:
        numeric_features = features.astype(np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{party_name} 的特征必须全部可以转换为数值类型。") from error
    if not np.isfinite(numeric_features.to_numpy()).all():
        raise ValueError(f"{party_name} 的特征不能包含 NaN 或无穷值。")
    return numeric_features


def _select_and_validate_binary_labels(
    alice_table: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """从 Alice 表中提取二分类标签，并检查其取值和数值有效性。"""
    try:
        labels = pd.to_numeric(
            alice_table[label_column], errors="raise"
        ).astype(np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("标签必须可以转换为数值类型。") from error
    if not np.isfinite(labels.to_numpy()).all():
        raise ValueError("标签不能包含 NaN 或无穷值。")
    if not set(labels.unique()).issubset({0.0, 1.0}):
        raise ValueError("当前项目仅支持取值为 0 或 1 的二分类标签。")
    return labels.reset_index(drop=True).to_frame(name=label_column)


def _validate_split_request(labels: pd.DataFrame, test_size: float) -> None:
    """检查分层训练/测试切分所需的类别数量和比例范围。"""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 必须位于 0 与 1 之间。")
    label_series = labels.iloc[:, 0]
    class_counts = label_series.value_counts()
    if len(class_counts) != 2:
        raise ValueError("训练数据必须同时包含两个二分类标签类别。")
    if (class_counts < 2).any():
        raise ValueError("每个标签类别至少需要两个样本，才能完成分层切分。")


def _build_partition(
    sample_ids: pd.Series,
    alice_features: pd.DataFrame,
    bob_features: pd.DataFrame,
    labels: pd.DataFrame,
    indices: np.ndarray,
) -> VerticalDataPartition:
    """依据同一组样本下标构造 Alice、Bob 与标签严格对齐的单个分区。"""
    return VerticalDataPartition(
        sample_ids=pd.Index(sample_ids.iloc[indices].to_numpy(), name=sample_ids.name),
        alice_features=alice_features.iloc[indices].reset_index(drop=True),
        bob_features=bob_features.iloc[indices].reset_index(drop=True),
        labels=labels.iloc[indices].reset_index(drop=True),
    )


def _frame_to_float32_array(frame: pd.DataFrame) -> np.ndarray:
    """在目标 PYU 内将已校验表转换为 SecretFlow 训练需要的二维数组。"""
    return frame.to_numpy(dtype=np.float32, copy=True)


def _ensure_columns_exist(
    table: pd.DataFrame, columns: Sequence[str], party_name: str
) -> None:
    """确认指定方的原始表包含全部必需列。"""
    missing_columns = sorted(set(columns) - set(table.columns))
    if missing_columns:
        raise ValueError(f"{party_name} 的本地表缺少必需列：{missing_columns}。")


def _normalize_feature_columns(
    columns: Sequence[str],
    party_name: str,
) -> tuple[str, ...]:
    """把特征列转换为稳定元组，并拒绝空列表和重复列名。"""
    normalized_columns = tuple(columns)
    if not normalized_columns:
        raise ValueError(f"{party_name} 至少需要提供一个特征列。")
    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError(f"{party_name} 的特征列名不能重复。")
    return normalized_columns
