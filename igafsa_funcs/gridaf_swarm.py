import numpy as np
from .eachaf_dist import eachAF_dist
from .distance import distance
from .grideaf_foodconsistence import GrideAF_foodconsistence
from .gridaf_prey import GridAF_prey
from .arry2orxy import arry2orxy


def GridAF_swarm(n, N, ppValue, ii, visual, delta, try_number, lastH, Barrier, goal, j, MAXGEN, rightInf, BX, a):
    """
    计算下一时刻人工鱼的位置和位置处的食物浓度。
    """
    sumx = 0  # 记录视野中人工鱼X轴数据之和
    sumy = 0  # 记录视野中人工鱼Y轴数据之和

    Xi = int(np.asarray(ppValue[:, ii]).ravel()[0])  # 当前人工鱼的位置，确保为标量
    BX = int(np.asarray(BX).ravel()[0])  # 确保为标量
    D = eachAF_dist(n, N, Xi, ppValue)  # 计算当前人工鱼与其他所有鱼群的欧式距离
    visual = np.mean(D)  # 自适应加权视野
    index = np.where((D > 0) & (D < visual))[1]  # 找到视野中的其他鱼群

    Nf = len(index)  # 确定视野之内的鱼群总数
    j = 1  # 记录单个鱼可行域的个数

    # 计算当前人工鱼的可行域
    A = np.arange(0, n ** 2 - 1)
    allow = np.setdiff1d(A, Barrier)
    allow_area = np.array([], dtype=np.int64)

    for i in allow:
        if 0 < distance(n, Xi, i) <= rightInf:
            allow_area = np.append(allow_area, i)
            j += 1

    if Nf > 0:  # Nf > 0说明视野之内有其他人工鱼群，则可以进行群聚行为
        for i in range(Nf):
            idx_i = int(np.asarray(ppValue[:, index[i]]).ravel()[0])
            row, col = np.unravel_index(idx_i, (n, n))
            array_x, array_y = arry2orxy(n, row, col)
            sumx += array_x
            sumy += array_y

        # 规整为整数处理, 逢小数进一
        avgx = int(np.ceil(sumx / Nf))
        avgy = int(np.ceil(sumy / Nf))

        # 将坐标反对应到矩阵值，再求食物浓度
        Xc = int(np.ravel_multi_index((avgy, avgx), (n, n)))  # 算出中心点位置
        Hc = GrideAF_foodconsistence(n, Xc, goal)  # 算出中心点和终点的距离
        Hi = float(lastH[ii])  # 当前人工鱼的食物浓度

        if Hc / Nf <= Hi * delta:  # 如果中心位置的食物浓度比当前位置高，并且不拥挤
            for _ in range(try_number):
                direction = (Xc - Xi) + (BX - Xi)
                norm_val = np.linalg.norm(direction)
                if norm_val == 0:
                    Xnext = Xi  # 无方向，保持不动或可选随机可行点
                else:
                    step = (direction / norm_val) * rightInf * float(np.random.rand())
                    Xnext = Xi + step

                # 将候选位置规整为标量整型并进行合法性判断
                ceil_cand = int(np.ceil(Xnext))
                floor_cand = int(np.floor(Xnext))
                if np.any(allow == ceil_cand):
                    Xnext = ceil_cand
                elif np.any(allow == floor_cand):
                    Xnext = floor_cand
                elif allow_area.size > 0:
                    Xnext = int(np.random.choice(allow_area))
                else:
                    Xnext = Xi  # 无可行域，原地

                if distance(n, Xnext, Xc) < distance(n, Xi, Xc):
                    nextPosition = int(Xnext)
                    nextPositionH = float(GrideAF_foodconsistence(n, nextPosition, goal))
                    break
                else:
                    nextPosition = int(Xi)
                    nextPositionH = float(GrideAF_foodconsistence(n, nextPosition, goal))
        else:  # 如果中心位置的食物浓度没有当前位置的食物浓度高，则进行觅食行为
            nextPosition, nextPositionH = GridAF_prey(n, N, ppValue[:, ii], ppValue, ii, try_number, lastH, Barrier,
                                                      goal, j, MAXGEN, rightInf, BX, a)
            nextPosition = int(np.asarray(nextPosition).ravel()[0])
            nextPositionH = float(np.asarray(nextPositionH).ravel()[0])
    else:  # 否则，Nf < 0说明视野范围内没有其他人工鱼群，那么就执行觅食行为
        nextPosition, nextPositionH = GridAF_prey(n, N, ppValue[:, ii], ppValue, ii, try_number, lastH, Barrier, goal, j,
                                                  MAXGEN, rightInf, BX, a)
        nextPosition = int(np.asarray(nextPosition).ravel()[0])
        nextPositionH = float(np.asarray(nextPositionH).ravel()[0])

    return nextPosition, nextPositionH
