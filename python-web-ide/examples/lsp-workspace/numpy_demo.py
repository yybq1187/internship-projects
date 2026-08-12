"""Pyright 与容器内 NumPy 环境的最小分析样例。"""

import numpy as np


# 后续用于验证 NumPy 模块成员与 ndarray 成员的类型推断。
data = np.array([1, 2, 3])
mean_value = data.mean()

print(mean_value)
