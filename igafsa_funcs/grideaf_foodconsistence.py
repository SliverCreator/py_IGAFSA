import numpy as np
from .arry2orxy import arry2orxy


def GrideAF_foodconsistence(n, X, goal):
    """
    计算人工鱼群的食物浓度。

    参数:
    n (int): 矩阵维数。
    X (list or np.ndarray): 人工鱼群当前位置。
    goal (int): 目标位置。

    返回:
    list: 当前鱼群位置的食物浓度。
    """
    Xf = []  # 记录当前鱼群的横坐标
    Yf = []  # 记录当前鱼群的纵坐标
    H = np.array([], dtype=np.int64)  # 记录当前鱼群的食物浓度

    # 转换目标位置为坐标
    goal_row, goal_col = np.unravel_index(goal, (n, n))
    Xg, Yg = arry2orxy(n, goal_row, goal_col)

    # 转换鱼群位置为坐标
    X = np.array(X, dtype=np.int64)
    X = X.flatten()
    # for x in X:
    row, col = np.unravel_index(np.int64(X), (n, n))
    xf, yf = arry2orxy(n, row, col)
    Xf.append(xf)
    Yf.append(yf)

    # 计算食物浓度的欧氏距离
    for xf, yf in zip(Xf, Yf):
        H = np.append(H, np.sqrt((xf - Xg) ** 2 + (yf - Yg) ** 2))

    return H
