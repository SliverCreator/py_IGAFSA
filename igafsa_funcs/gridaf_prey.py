import numpy as np
from .eachaf_dist import eachAF_dist
from .distance import distance
from .grideaf_foodconsistence import GrideAF_foodconsistence


def GridAF_prey(n, N, ppValue, X, ii, try_number, lastH, Barrier, goal, k, MAXGEN, rightInf, BX, a):
    """
    人工鱼觅食行为函数。

    参数:
    n (int): 矩阵维数。
    N (int): 人工鱼数量。
    ppValue (int): 人工鱼当前在栅格中的值（单个）。
    X (list or np.ndarray): 当前鱼群的位置。
    ii (int): 当前人工鱼序号。
    try_number (int): 最大尝试次数。
    lastH (list or np.ndarray): 上次人工鱼食物浓度。
    Barrier (list or np.ndarray): 障碍矩阵。
    goal (int): 人工鱼目标位置。
    k (int): 当前迭代次数。
    MAXGEN (int): 最大迭代次数。
    rightInf (float): 视野范围。
    BX (int): 最优鱼的位置。
    a (list or np.ndarray): 障碍环境。

    返回:
    tuple: 下一时刻的人工鱼栅格矩阵值和该位置的食物浓度。
    """
    nextPosition = None
    allow_area = np.array([], dtype=np.int64)  # 记录可行域
    j = 0  # 记录单个鱼可行域的个数
    present_H = lastH[ii]  # 当前位置时刻的食物浓度值
    Xi = ppValue

    # 自适应步长
    alpha = 1
    D = eachAF_dist(n, N, Xi, X)  # 计算当前人工鱼与其他所有鱼群的欧式距离
    # mask = (D > 0)
    if np.all(D > 0):
        visual = np.mean(D)  # 自适应加权视野
        rightInf = alpha * visual

    # 人工鱼的可行域，计算出当前位置周边能走的点，距离在根号2以内
    A = np.arange(0, n ** 2 - 1)
    allow = np.setdiff1d(A, Barrier)
    for i in allow:
        if 0 < distance(n, ppValue, i) <= rightInf:
            allow_area = np.append(allow_area, i)
            j += 1

    # 加入方向因子，直接选择allow_area区域中的最小值
    m = np.random.choice([0, 1], p=[0.2, 0.8])
    if m == 0:
        # 正常选择，直接随机走一步
        for _ in range(try_number):
            Xj = np.random.choice(allow_area)
            Hj = GrideAF_foodconsistence(n, Xj, goal)
            if present_H > Hj:  # 说明下一步的值距离goal更近，保留
                nextPosition = Xj
                break
    else:
        H_min = present_H
        for Xi in allow_area:
            Hi = GrideAF_foodconsistence(n, Xi, goal)
            if Hi < H_min:
                H_min = Hi
                nextPosition = Xi

    if nextPosition is None:
        nextPosition = np.random.choice(allow_area)

    nextPositionH = GrideAF_foodconsistence(n, nextPosition, goal)
    return nextPosition, nextPositionH
