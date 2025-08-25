import numpy as np


def arry2orxy(n, row, col):
    """
    将矩阵下标转换为坐标轴坐标。

    参数:
    n (int): 矩阵维数。
    row (list or np.ndarray): 行索引。
    col (list or np.ndarray): 列索引。

    返回:
    tuple: 横坐标 array_x 和纵坐标 array_y。
    """

    # k = len(row)  # 获取行索引的长度
    row = np.atleast_2d(row)
    col = np.atleast_2d(col)
    k = row.shape[1]
    array_x = np.zeros(k, dtype=int)  # 初始化横坐标数组
    array_y = np.zeros(k, dtype=int)  # 初始化纵坐标数组

    array_x = col  # 将列索引赋值给横坐标
    array_y = row  # 将行索引赋值给纵坐标

    return array_x, array_y
