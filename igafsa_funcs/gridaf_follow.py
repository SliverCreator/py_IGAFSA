import numpy as np
from .eachaf_dist import eachAF_dist
from .distance import distance
from .grideaf_foodconsistence import GrideAF_foodconsistence
from .gridaf_prey import GridAF_prey


def GridAF_follow(n, N, ppValue, ii, visual, delta, try_number, lastH, Barrier, goal, j, MAXGEN, rightInf, BX, a):
    """
    计算下一时刻人工鱼的位置和位置处的食物浓度。
    """
    Xi = int(np.asarray(ppValue[:, ii]).ravel()[0])  # 当前人工鱼的位置，标量
    BX = int(np.asarray(BX).ravel()[0])  # 标量
    D = eachAF_dist(n, N, Xi, ppValue)  # 计算当前人工鱼与其他所有鱼群的欧式距离
    visual = np.mean(D)  # 自适应加权视野
    index = np.where((D > 0) & (D < visual))[0]  # 找到视野中的其他鱼群
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
        Xvisual = ppValue[:, index]  # 取出与当前人工鱼邻近的鱼群的坐标
        Hvisual = lastH[index]  # 取出与当前人工鱼邻近的鱼群的食物浓度
        Hmin, minindex = np.min(Hvisual), np.argmin(Hvisual)  # 求出邻近鱼群中食物浓度最低的值
        Xmin = int(np.asarray(Xvisual[:, minindex]).ravel()[0])  # 得到食物浓度最低的鱼群位置

        Hi = float(lastH[ii])  # 当前人工鱼的食物浓度
        if Hmin / Nf <= Hi * delta:  # 如果中心位置的食物浓度比当前位置高，并且不拥挤
            for _ in range(try_number):
                direction = (Xmin - Xi) + (BX - Xi)
                norm_val = np.linalg.norm(direction)
                if norm_val == 0:
                    Xnext = Xi
                else:
                    step = (direction / norm_val) * rightInf * float(np.random.rand())
                    Xnext = Xi + step

                ceil_cand = int(np.ceil(Xnext))
                floor_cand = int(np.floor(Xnext))
                if np.any(allow == ceil_cand):
                    Xnext = ceil_cand
                elif np.any(allow == floor_cand):
                    Xnext = floor_cand
                elif allow_area.size > 0:
                    Xnext = int(np.random.choice(allow_area))
                else:
                    Xnext = Xi

                if distance(n, Xnext, Xmin) < distance(n, Xi, Xmin):
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
