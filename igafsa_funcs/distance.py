import numpy as np
from .arry2orxy import arry2orxy


def distance(n, ppValue, randomValue):
    """
    计算两个位置之间的欧式距离。

    参数:
    n (int): 矩阵的维数。
    ppValue (int): 当前位置索引。
    randomValue (int): 任意位置索引。

    返回:
    float: 两个位置之间的距离。
    """
    # 将当前位置和任意位置的索引转换为矩阵的行列值
    pp_row, pp_col = np.unravel_index(ppValue, (n, n))
    randomValue = np.array([randomValue])
    rand_row, rand_col = np.unravel_index(randomValue, (n, n))

    # 将矩阵下标转换为坐标轴上的坐标
    pp_array_x, pp_array_y = arry2orxy(n, pp_row, pp_col)
    rand_array_x, rand_array_y = arry2orxy(n, rand_row, rand_col)

    # 计算欧式距离
    dist = np.sqrt((pp_array_x - rand_array_x) ** 2 + (pp_array_y - rand_array_y) ** 2)

    return dist
