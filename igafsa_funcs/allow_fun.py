import numpy as np


def allow_fun(n, ppValue, allow_area):
    # 计算当前人工鱼的可行域
    # 输入参数：
    # n ---- 矩阵维数
    # ppValue ---- 当前人工鱼的位置的线性索引
    # allow_area ---- 包含所有允许区域线性索引的数组
    # 输出参数：
    # allow_area ---- 当前人工鱼的可行域

    n = 20
    # 将线性索引转换为行列坐标
    Xi = np.unravel_index(ppValue, (n, n))
    Xallow = np.array([np.unravel_index(idx, (n, n)) for idx in allow_area]).T

    # 初始化一个空列表来存储合理的斜向坐标
    Check = []

    # 计算四个斜向的坐标
    offsets = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])  # 对应左上、右上、左下、右下的行列偏移量

    # i的1234对应左上、右上、左下、右下
    for i in range(offsets.shape[0]):
        newRow = Xi[0] + offsets[i, 0]
        newCol = Xi[1] + offsets[i, 1]
        # 检查新坐标是否在矩阵的合理范围内
        if 1 <= newRow <= n and 1 <= newCol <= n:
            # 如果在范围内，添加到Check
            Check.append((newRow, newCol, i + 1))

    Check = np.array(Check)

    for i in range(Check.shape[0]):
        X = Check[i, :2]
        if any(np.all(X == row) for row in Xallow.T):
            j = Check[i, 2]
            if j == 1:
                X1 = [Xi[0] - 1, Xi[1]]
                X2 = [Xi[0], Xi[1] - 1]
            elif j == 2:
                X1 = [Xi[0] - 1, Xi[1]]
                X2 = [Xi[0], Xi[1] + 1]
            elif j == 3:
                X1 = [Xi[0] + 1, Xi[1]]
                X2 = [Xi[0], Xi[1] - 1]
            elif j == 4:
                X1 = [Xi[0] + 1, Xi[1]]
                X2 = [Xi[0], Xi[1] + 1]

            if not any(np.all(X1 == row) for row in Xallow.T) or not any(np.all(X2 == row) for row in Xallow.T):
                Xallow = Xallow[:, ~np.all(Xallow.T == X, axis=1)]

    allow_area = np.ravel_multi_index((Xallow[0], Xallow[1]), (n, n))
    return allow_area
