import numpy as np
from .arry2orxy import arry2orxy


def eachAF_dist(n, N, ppValue, position):
    """
    计算当前人工鱼与其他人工鱼的位置距离，包括自身在内。

    参数:
    n (int): 矩阵维数。
    N (int): 人工鱼数量。
    ppValue (int): 当前人工鱼的位置。
    position (list or np.ndarray): 所有人工鱼的位置。

    返回:
    np.ndarray: 当前人工鱼与其他人工鱼的位置距离数组。
    """
    D = np.zeros(N)  # 初始化距离数组

    # 将当前人工鱼位置转换为坐标
    pp_row, pp_col = np.unravel_index(ppValue, (n, n))  # 栅格中的数值转化成数组行列值
    pp_array_x, pp_array_y = arry2orxy(n, pp_row, pp_col)  # 将矩阵下标转换为坐标轴xy形式

    # for i in range(N):
    position_row, position_col = np.unravel_index(position, (n, n))  # 栅格中的数值转化成数组行列值
    position_array_x, position_array_y = arry2orxy(n, position_row, position_col)  # 将矩阵下标转换为坐标轴xy形式

    # 计算欧式距离
    # D[i] = np.linalg.norm([pp_array_x - position_array_x, pp_array_y - position_array_y])
    dx = pp_array_x - position_array_x  # (1, 30)
    dy = pp_array_y - position_array_y  # (1, 30)

    D = np.sqrt(dx ** 2 + dy ** 2)  # (1, 30)

    return D
