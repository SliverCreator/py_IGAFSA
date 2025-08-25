import numpy as np
from .myangle import myangle


def bezierSmooth(Path):
    """
    输入:
        Path: 形状为 (N, 2) 的路径点数组
    输出:
        smoothPathArr: 形状为 (M, 2) 的平滑后路径点数组
        distanceX: 路径总长度
    逻辑:
        1) 首先根据转折角度将原路径打断插值, 得到中间路径 newPathArr
        2) 再次检测转折角度, 在相邻转折点进行贝塞尔插值, 得到平滑后的路径 smoothPathArr
        3) 计算平滑后路径的总长度 distanceX
    """
    # ---------------------------
    # 第一次遍历, 找到原始 Path 中每一对相邻向量间的转折点
    # ---------------------------
    turn_idx_list = []
    for i in range(len(Path) - 2):
        x_n1 = Path[i + 1, 0] - Path[i, 0]
        y_n1 = Path[i + 1, 1] - Path[i, 1]
        x_n2 = Path[i + 2, 0] - Path[i + 1, 0]
        y_n2 = Path[i + 2, 1] - Path[i + 1, 1]
        angle = myangle(x_n1, y_n1, x_n2, y_n2)
        if angle > 0.01:
            turn_idx_list.append(i + 1)

    turn_idx = np.array(turn_idx_list, dtype=int)

    newPath_chunks = []
    newPath_chunks.append(Path[:turn_idx[0]])

    for i in range(len(turn_idx) - 1):
        s, e = turn_idx[i], turn_idx[i + 1]
        newPath_chunks.append(Path[s:e])
        if s + 1 == e:
            numPoints = 1
            t = 1 / (numPoints + 1)
            interpolatedPoint = (1 - t) * Path[s] + t * Path[e]
            newPath_chunks.append(interpolatedPoint[np.newaxis, :])

    newPath_chunks.append(Path[turn_idx[-1]:])
    newPathArr = np.concatenate(newPath_chunks, axis=0)

    # 第二次遍历
    turn_idx_list_2 = []
    for i in range(len(newPathArr) - 2):
        x_n1 = newPathArr[i + 1, 0] - newPathArr[i, 0]
        y_n1 = newPathArr[i + 1, 1] - newPathArr[i, 1]
        x_n2 = newPathArr[i + 2, 0] - newPathArr[i + 1, 0]
        y_n2 = newPathArr[i + 2, 1] - newPathArr[i + 1, 1]
        angle = myangle(x_n1, y_n1, x_n2, y_n2)
        if angle > 0.01:
            turn_idx_list_2.append(i + 1)

    turn_idx_2 = np.array(turn_idx_list_2, dtype=int)

    smoothPath_chunks = []
    smoothPath_chunks.append(newPathArr[:turn_idx_2[0]])

    for i in range(len(turn_idx_2)):
        idx_cur = turn_idx_2[i]

        ControlPoint_list = []
        numPoints = 2

        for j in range(1, numPoints + 1):
            t = j / (numPoints + 1)
            interpolated = (1 - t) * newPathArr[idx_cur - 1] + t * newPathArr[idx_cur]
            ControlPoint_list.append(interpolated)

        for j in range(1, numPoints + 1):
            t = j / (numPoints + 1)
            interpolated = (1 - t) * newPathArr[idx_cur] + t * newPathArr[idx_cur + 1]
            ControlPoint_list.append(interpolated)

        ControlPoint = np.array(ControlPoint_list)
        P0, P1, P2, P3 = ControlPoint[0], ControlPoint[1], ControlPoint[2], ControlPoint[3]

        t_vals = np.linspace(0, 1, 100)
        BezierPoints = np.zeros((len(t_vals), 2), dtype=float)
        for k, t_ in enumerate(t_vals):
            BezierPoints[k] = (1 - t_) ** 3 * P0 \
                              + 3 * (1 - t_) ** 2 * t_ * P1 \
                              + 3 * (1 - t_) * t_ ** 2 * P2 \
                              + t_ ** 3 * P3

        smoothPath_chunks.append(BezierPoints)

        if i < len(turn_idx_2) - 1:
            idx_next = turn_idx_2[i + 1]
            smoothPath_chunks.append(newPathArr[idx_cur + 1: idx_next])
        else:
            smoothPath_chunks.append(newPathArr[idx_cur + 1:])

    smoothPathArr = np.concatenate(smoothPath_chunks, axis=0)

    diff = smoothPathArr[1:] - smoothPathArr[:-1]
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    S = np.sum(distances)

    distanceX = S
    return smoothPathArr, distanceX
