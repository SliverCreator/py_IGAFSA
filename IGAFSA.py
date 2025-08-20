import time

import matplotlib.pyplot as plt
import numpy as np


def IAFSA():
    start_time = time.time()

    # 加载障碍环境
    a = np.loadtxt('object3_2.txt')

    # 统计矩阵中0和1的数量
    count_zeros = np.sum(a == 0)
    count_ones = np.sum(a == 1)

    n = a.shape[0]
    b = np.copy(a)
    # b = np.pad(b, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    plt.figure()
    plt.imshow(b, cmap='gray', origin='lower', extent=[0, n, 0, n])
    plt.xticks(np.arange(0, n - 1, 10))
    plt.yticks(np.arange(0, n - 1, 10))
    plt.text(-3, -1, 'START', color='red', fontsize=10)
    plt.text(n, n, 'GOAL', color='red', fontsize=10)
    # plt.show()

    # 障碍物矩阵
    Barrier = np.where(a.ravel() == 0)[0]

    # 人工鱼群参数
    N = 50
    try_number = 8
    MAXGEN = 100
    visual = 10
    delta = 0.618
    start = np.array([0])
    DistMin = np.sqrt((1 - n) ** 2 + (n - 1) ** 2)
    goal = np.array([399])
    shift = 1
    shiftFreq = 4
    rightInf = np.sqrt(2)
    arrayValue = [20]
    ppValue = np.full((1, N), start)
    # ppValue = [start] * N
    position = np.tile(ppValue, (MAXGEN, 1))
    H = GrideAF_foodconsistence(n, ppValue, goal)
    BH, BHindex = np.min(H), np.argmin(H)
    BX = ppValue[:,BHindex]
    count = 1
    runDist = 0
    runDist_part = 0
    BestH = np.zeros((1,MAXGEN))
    index = []

    for j in range(MAXGEN):
        if shift == 1:
            for i in range(N):
                if j == 0:
                    nextPosition, nextPositionH = GridAF_prey(n, N, ppValue[:,i], position[j:j+1], i, try_number, H, Barrier,
                                                              goal, j, MAXGEN, rightInf, BX, a)
                else:
                    nextPosition, nextPositionH = GridAF_prey(n, N, ppValue[:,i], position[j - 1:j], i, try_number, H,
                                                              Barrier, goal, j, MAXGEN, rightInf, BX, a)
                position[j][i] = nextPosition
                H[i] = nextPositionH
        else:
            for i in range(N):
                nextPosition_S, nextPositionH_S = GridAF_swarm(n, N, position[j - 1:j], i, visual, delta, try_number, H,
                                                               Barrier, goal, j, MAXGEN, rightInf, BX, a)
                nextPosition_F, nextPositionH_F = GridAF_follow(n, N, position[j - 1:j], i, visual, delta, try_number, H,
                                                                Barrier, goal, j, MAXGEN, rightInf, BX, a)
                if nextPositionH_F < nextPositionH_S:
                    nextPosition = nextPosition_F
                    nextPositionH = nextPositionH_F
                else:
                    nextPosition = nextPosition_S
                    nextPositionH = nextPositionH_S
                position[j][i] = nextPosition
                H[i] = nextPositionH

        count += 1
        shift = 2 if count % shiftFreq == 0 else 1
        ppValue = position[j:j+1]
        BH, BHindex = np.min(H), np.argmin(H)
        BX = ppValue[:,BHindex]
        index = np.where(position[j,:] == (goal[0]-1))[0]
        if index.size > 0:
            break

    if j >= MAXGEN-1:
        print('没有路径可以到达目标!!!')
    else:
        transimit = np.array([])
        for i in index:
            arrayValue = position[:,i]
            arrayValue = np.trim_zeros(arrayValue, 'b')
            arrayValue = np.concatenate((start, arrayValue,goal))  # 先拼到一起，得到一维数组
            arrayValue = arrayValue.reshape(1, -1) # 变形为一行
            for j in range(arrayValue.shape[1] - 1):
                d = distance(n, arrayValue[0,j], arrayValue[0,j + 1])
                runDist_part += d
            transimit = np.append(transimit,runDist_part)
            runDist_part = 0

        runDist, runMin_index = np.min(transimit), np.argmin(transimit)
        arrayValue = position[:, index[runMin_index]]
        arrayValue = np.trim_zeros(arrayValue, 'b')
        arrayValue = np.concatenate((start, arrayValue,goal))
        arrayValue = arrayValue.reshape(1, -1)
        print(f'IGAFSA行走长度为: {runDist}')

        for i in range(arrayValue.shape[1]):
            BestH[0,i] = goal - arrayValue[0,i]

        row, col = np.unravel_index(arrayValue, (n, n))
        array_x, array_y = arry2orxy(n, row, col)
        h1 = plt.plot(array_x[0,:]+0.5, array_y[0,:]+0.5,'r',linewidth=2, marker = 'o')
        # plt.show(block=False)

        Optimal_path = np.column_stack((array_x.ravel(), array_y.ravel()))
        _, idx = np.unique(Optimal_path, axis=0, return_index=True)
        idx_sorted = np.sort(idx)
        Optimal_path = Optimal_path[idx_sorted]
        IAFSA_path = Optimal_path

        # plt.plot(IAFSA_path[:, 0] + 0.5, IAFSA_path[:, 1] + 0.5, 'r', linewidth=2)
        Optimal_path = np.flipud(Optimal_path)
        Optimal_path = np.hstack([Optimal_path, np.zeros((Optimal_path.shape[0], 2))])
        Optimal_path[:-1, 2] = Optimal_path[1:, 0]  # 复制 x
        Optimal_path[:-1, 3] = Optimal_path[1:, 1]  # 复制 y
        Optimal_path = np.delete(Optimal_path, -1, axis=0)
        # h1 = plt.plot(IAFSA_path[:, 0] + 0.5, IAFSA_path[:, 1] + 0.5, 'b', linewidth=2,marker = 'o')
        # plt.show()


        Path, distanceX = Smooth(a, Optimal_path, 0.75)
        # h2 = plt.plot(Path[:, 0] + 0.5, Path[:, 1] + 0.5, 'b', linewidth=2,marker = 'o')
        # plt.show()
        print(f'IGAFSA+Smooth行走长度为: {distanceX}')

        newPath, distanceB = bezierSmooth(Path)
        h3 = plt.plot(newPath[:, 0] + 0.5, newPath[:, 1] + 0.5, 'm', linewidth=2)
        plt.show()
        print(f'IGAFSA+Smooth+bezier行走长度为: {distanceB}')
        # plt.legend([h1, h2, h3], ['IGAFSA', 'IGAFSA1', 'IGAFSA2'], loc='northwest')

    end_time = time.time()
    print(f'代码运行时间： {end_time - start_time:.2f} 秒')


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
    allow_area = np.array([],dtype=np.int64)  # 记录可行域
    j = 0  # 记录单个鱼可行域的个数
    present_H = lastH[ii]  # 当前位置时刻的食物浓度值
    Xi = ppValue

    # 自适应步长
    alpha = 1
    D = eachAF_dist(n, N, Xi, X)  # 计算当前人工鱼与其他所有鱼群的欧式距离
    # mask = (D > 0)
    if np.all(D>0):
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


def GridAF_swarm(n, N, ppValue, ii, visual, delta, try_number, lastH, Barrier, goal, j, MAXGEN, rightInf, BX, a):
    """
    计算下一时刻人工鱼的位置和位置处的食物浓度。

    参数:
    n (int): 矩阵维数。
    N (int): 人工鱼总数。
    ppValue (list or np.ndarray): 所有人工鱼群的位置。
    ii (int): 当前人工鱼群的编号。
    visual (float): 感知范围。
    delta (float): 拥挤因子。
    try_number (int): 尝试次数。
    lastH (list or np.ndarray): 当前人工鱼上一次的食物浓度。
    Barrier (list or np.ndarray): 障碍物矩阵。
    goal (int): 人工鱼目标位置。
    j, MAXGEN: 用于传递 prey 函数的参数，与本函数无关。
    rightInf (float): 右侧边界信息。
    BX (float): 边界X坐标。
    a (float): 参数a。

    返回:
    tuple: 下一时刻该条人工鱼的位置和位置处的食物浓度。
    """
    sumx = 0  # 记录视野中人工鱼X轴数据之和
    sumy = 0  # 记录视野中人工鱼Y轴数据之和

    Xi = ppValue[:,ii]  # 当前人工鱼的位置
    D = eachAF_dist(n, N, Xi, ppValue)  # 计算当前人工鱼与其他所有鱼群的欧式距离
    visual = np.mean(D)  # 自适应加权视野
    index = np.where((D > 0) & (D < visual))[1]  # 找到视野中的其他鱼群

    Nf = len(index)  # 确定视野之内的鱼群总数
    j = 1  # 记录单个鱼可行域的个数

    # 计算当前人工鱼的可行域
    A = np.arange(0, n ** 2 - 1)
    allow = np.setdiff1d(A, Barrier)
    allow_area = np.array([],dtype=np.int64)

    for i in allow:
        if 0 < distance(n, Xi, i) <= rightInf:
            allow_area = np.append(allow_area,i)
            j += 1

    if Nf > 0:  # Nf > 0说明视野之内有其他人工鱼群，则可以进行群聚行为
        for i in range(Nf):
            row, col = np.unravel_index(ppValue[:,index[i]], (n, n))
            array_x, array_y = arry2orxy(n, row, col)
            sumx += array_x
            sumy += array_y

        # 规整为整数处理, 逢小数进一
        avgx = np.ceil(sumx / Nf).astype(int)
        avgy = np.ceil(sumy / Nf).astype(int)

        # 将坐标反对应到矩阵值，再求食物浓度
        Xc = np.ravel_multi_index((avgy, avgx), (n, n))  # 算出中心点位置
        Hc = GrideAF_foodconsistence(n, Xc, goal)  # 算出中心点和终点的距离
        Hi = lastH[ii]  # 当前人工鱼的食物浓度

        if Hc / Nf <= Hi * delta:  # 如果中心位置的食物浓度比当前位置高，并且不拥挤
            for i in range(try_number):
                Xnext = Xi + ((Xc - Xi) + (BX - Xi)) / np.linalg.norm(
                    (Xc - Xi) + (BX - Xi)) * rightInf * np.random.rand()
                if np.ceil(Xnext).astype(np.int64) in allow:
                    Xnext = np.ceil(Xnext).astype(np.int64)
                elif np.floor(Xnext).astype(np.int64) in allow:
                    Xnext = np.floor(Xnext).astype(np.int64)
                else:
                    Xnext = np.random.choice(allow_area).astype(np.int64)

                # Xnext = np.random.choice(allow_area)
                if distance(n, Xnext, Xc) < distance(n, Xi, Xc):
                    nextPosition = Xnext
                    nextPositionH = GrideAF_foodconsistence(n, nextPosition, goal)
                    break
                else:
                    nextPosition = Xi
                    nextPositionH = GrideAF_foodconsistence(n, nextPosition, goal)
        else:  # 如果中心位置的食物浓度没有当前位置的食物浓度高，则进行觅食行为
            nextPosition, nextPositionH = GridAF_prey(n, N, ppValue[:,ii], ppValue, ii, try_number, lastH, Barrier, goal,
                                                      j, MAXGEN, rightInf, BX, a)
    else:  # 否则，Nf < 0说明视野范围内没有其他人工鱼，那么就执行觅食行为
        nextPosition, nextPositionH = GridAF_prey(n, N, ppValue[:,ii], ppValue, ii, try_number, lastH, Barrier, goal, j,
                                                  MAXGEN, rightInf, BX, a)

    return nextPosition, nextPositionH


def GridAF_follow(n, N, ppValue, ii, visual, delta, try_number, lastH, Barrier, goal, j, MAXGEN, rightInf, BX, a):
    """
    计算下一时刻人工鱼的位置和位置处的食物浓度。

    参数:
    n (int): 矩阵维数。
    N (int): 人工鱼总数。
    ppValue (list or np.ndarray): 所有人工鱼群的位置。
    ii (int): 当前人工鱼群的编号。
    visual (float): 感知范围。
    delta (float): 拥挤因子。
    try_number (int): 尝试次数。
    lastH (list or np.ndarray): 当前人工鱼上一次的食物浓度。
    Barrier (list or np.ndarray): 障碍物矩阵。
    goal (int): 人工鱼目标位置。
    j, MAXGEN: 用于传递 prey 函数的参数，与本函数无关。
    rightInf (float): 右侧边界信息。
    BX (float): 边界X坐标。
    a (float): 参数a。

    返回:
    tuple: 下一时刻该条人工鱼的位置和位置处的食物浓度。
    """
    Xi = ppValue[:,ii]  # 当前人工鱼的位置
    D = eachAF_dist(n, N, Xi, ppValue)  # 计算当前人工鱼与其他所有鱼群的欧式距离
    visual = np.mean(D)  # 自适应加权视野
    index = np.where((D > 0) & (D < visual))[0]  # 找到视野中的其他鱼群
    Nf = len(index)  # 确定视野之内的鱼群总数
    j = 1  # 记录单个鱼可行域的个数

    # 计算当前人工鱼的可行域
    A = np.arange(0, n ** 2 - 1)
    allow = np.setdiff1d(A, Barrier)
    allow_area = np.array([],dtype=np.int64)

    for i in allow:
        if 0 < distance(n, Xi, i) <= rightInf:
            allow_area = np.append(allow_area,i)
            j += 1

    if Nf > 0:  # Nf > 0说明视野之内有其他人工鱼群，则可以进行群聚行为
        Xvisual = ppValue[:,index]  # 取出与当前人工鱼邻近的鱼群的坐标
        Hvisual = lastH[index]  # 取出与当前人工鱼邻近的鱼群的食物浓度
        Hmin, minindex = np.min(Hvisual), np.argmin(Hvisual)  # 求出邻近鱼群中食物浓度最低的值
        Xmin = Xvisual[:,minindex]  # 得到食物浓度最低的鱼群位置

        Hi = lastH[ii]  # 当前人工鱼的食物浓度
        if Hmin / Nf <= Hi * delta:  # 如果中心位置的食物浓度比当前位置高，并且不拥挤
            for i in range(try_number):
                Xnext = Xi + ((Xmin - Xi) + (BX - Xi)) / np.linalg.norm(
                    (Xmin - Xi) + (BX - Xi)) * rightInf * np.random.rand()
                if np.ceil(Xnext).astype(np.int64) in allow:
                    Xnext = np.ceil(Xnext).astype(np.int64)
                elif np.floor(Xnext).astype(np.int64) in allow:
                    Xnext = np.floor(Xnext).astype(np.int64)
                else:
                    Xnext = np.random.choice(allow_area).astype(np.int64)

                # Xnext = np.random.choice(allow_area)

                if distance(n, Xnext, Xmin) < distance(n, Xi, Xmin):
                    nextPosition = Xnext
                    nextPositionH = GrideAF_foodconsistence(n, nextPosition, goal)
                    break
                else:
                    nextPosition = Xi
                    nextPositionH = GrideAF_foodconsistence(n, nextPosition, goal)
        else:  # 如果中心位置的食物浓度没有当前位置的食物浓度高，则进行觅食行为
            nextPosition, nextPositionH = GridAF_prey(n, N, ppValue[:,ii], ppValue, ii, try_number, lastH, Barrier, goal,
                                                      j, MAXGEN, rightInf, BX, a)
    else:  # 否则，Nf < 0说明视野范围内没有其他人工鱼，那么就执行觅食行为
        nextPosition, nextPositionH = GridAF_prey(n, N, ppValue[:,ii], ppValue, ii, try_number, lastH, Barrier, goal, j,
                                                  MAXGEN, rightInf, BX, a)

    return nextPosition, nextPositionH


def Smooth(a, Optimal_path, D):
    """
    平滑路径并计算路径长度。

    参数:
    a (float): 参数，用于调整障碍物矩阵。
    Optimal_path (np.ndarray): 初始的最优路径。
    D: 未使用的参数，可能用于距离计算。

    返回:
    tuple: 平滑后的路径和路径长度。
    """
    # 预备变量
    k = 0
    CLOSED = []
    MAX = 1 - a
    MAX_X = MAX.shape[1]  # 获取列数，即x轴长度
    MAX_Y = MAX.shape[0]  # 获取行数，即y轴长度

    # 找到所有障碍物位置
    for j in range(MAX_X):  # 列遍历
        for i in range(MAX_Y):  # 行遍历
            if MAX[i, j] == 1:
                CLOSED.append((i, j))
                k += 1

    Obs_Closed = np.array(CLOSED)

    xStart = Optimal_path[-1, 2]
    yStart = Optimal_path[-1, 3]
    xTarget = Optimal_path[0, 0]
    yTarget = Optimal_path[0, 1]

    # 路径优化
    Num_obs = Obs_Closed.shape[0]
    CLOSED = Obs_Closed
    Num_Opt = Optimal_path.shape[0]

    # 优化折线,第一次传入倒序n*4格式，得到正序n*4
    Optimal_path_one = Line_OPEN_ST(Optimal_path, CLOSED, Num_obs, Num_Opt, D)
    # 使用 unique 函数删除重复的行
    _, ia = np.unique(Optimal_path_one[:, :2], axis=0, return_index=True)
    Optimal_path_one = Optimal_path_one[np.sort(ia)]
    Optimal_path_one1 = Optimal_path_one[:, 0:2]
    Optimal_path_one1 = np.vstack((Optimal_path_one1, Optimal_path_one[-1, 2:4]))

    # 通过切片计算相邻点坐标差
    diff = Optimal_path_one1[1:] - Optimal_path_one1[:-1]  # 形状 (N-1, 2)
    # 计算每段距离，并求和
    distances = np.sqrt(np.sum(diff ** 2, axis=1))  # 形状 (N-1,)
    S = np.sum(distances)
    print(f'一次IGAFSA+Smooth行走长度为: {S}')
    plt.plot(Optimal_path_one1[:, 0]+0.5, Optimal_path_one1[:, 1]+0.5, 'b',marker = 'o')
    # plt.show()

    # 每隔2个取点，传入正序n*2,得到正序n*2
    L_obst = 2
    Obst_d_d_line = Line_obst(Optimal_path_one1, L_obst)
    Optimal_path_one1 = np.vstack([Obst_d_d_line,[xTarget,yTarget]])

    # 反过来排列路线节点
    # 1. 取后两列并行翻转（列索引 2 和 3），得到 (M, 2)
    part1 = Optimal_path_one1[1:][::-1]
    # 2. 将第一行 (0,1) 两列的值追加到末尾
    part2 = Optimal_path_one1[:-1][::-1]  # [None, :] 保证形状统一为 (1,2)
    # 3. 将两部分拼接获得最终的 Optimal_path (形状 (M+1, 2))
    Optimal_path_1 = np.hstack((part1, part2))

    # 二次优化折线，传入倒序n*4格式，得到正序n*4
    Num_Opt = Optimal_path_1.shape[0]
    Optimal_path_two = Line_OPEN_ST(Optimal_path_1, CLOSED, Num_obs, Num_Opt, D)
    # 使用 unique 函数删除重复的行
    _, ia = np.unique(Optimal_path_two[:, :2], axis=0, return_index=True)
    Optimal_path_two = Optimal_path_two[np.sort(ia)]
    Optimal_path_two2 = Optimal_path_two[:, 0:2]
    Optimal_path_two2 = np.vstack((Optimal_path_two2, Optimal_path_two[-1, 2:4]))

    # 通过切片计算相邻点坐标差
    diff = Optimal_path_two2[1:] - Optimal_path_two2[:-1]  # 形状 (N-1, 2)
    # 计算每段距离，并求和
    distances = np.sqrt(np.sum(diff ** 2, axis=1))  # 形状 (N-1,)
    S = np.sum(distances)
    print(f'二次IGAFSA+Smooth行走长度为: {S}')
    plt.plot(Optimal_path_two2[:, 0]+0.5, Optimal_path_two2[:, 1]+0.5, 'c',marker = 'o')
    # plt.show()

    # # 二次折线优化，传入倒序n*2格式
    # num_op = Optimal_path.shape[0] - 1
    # Optimal_path_two = Line_OPEN_STtwo(Optimal_path, CLOSED, Num_obs, num_op, D)
    # _, ia = np.unique(Optimal_path_two[:, :2], axis=0, return_index=True)
    # Optimal_path_two = Optimal_path_two[np.sort(ia)]
    # num_optwo = Optimal_path_two.shape[0]
    # Optimal_path_two = np.vstack([Optimal_path_two, [xStart, yStart]])

    # 三次折线，传入正序n*2
    num_optwo = Optimal_path_two2.shape[0]
    Optimal_path_three = Line_OPEN_STtwo(Optimal_path_two2, CLOSED, Num_obs, num_optwo, D)
    _, ia = np.unique(Optimal_path_three[:, :2], axis=0, return_index=True)
    Optimal_path_three = Optimal_path_three[np.sort(ia)]
    num_opthree = Optimal_path_three.shape[0]
    Optimal_path_three = np.vstack([Optimal_path_three, [xTarget, yTarget]])

    # 每隔2个取点，传入正序n*2
    L_obst = 2
    Obst_d_d_line = Line_obst(Optimal_path_three, L_obst)
    Obst_d_d_line = np.vstack([Obst_d_d_line,[xTarget,yTarget]])
    _, ia = np.unique(Obst_d_d_line[:, :2], axis=0, return_index=True)
    Obst_d_d_line = Obst_d_d_line[np.sort(ia)]
    num_opthree = Obst_d_d_line.shape[0]

    NewOptimal_path = Obst_d_d_line
    _, ia = np.unique(NewOptimal_path[:, :2], axis=0, return_index=True)
    NewOptimal_path = NewOptimal_path[np.sort(ia)]

    plt.plot(NewOptimal_path[:, 0] + 0.5, NewOptimal_path[:, 1] + 0.5, 'g', linewidth=2,marker = 'o')
    # plt.show()
    # 计算路径长度
    # 通过切片计算相邻点坐标差
    diff = NewOptimal_path[1:] - NewOptimal_path[:-1]  # 形状 (N-1, 2)
    # 计算每段距离，并求和
    distances = np.sqrt(np.sum(diff ** 2, axis=1))  # 形状 (N-1,)
    S = np.sum(distances)

    distanceX = S
    Path = NewOptimal_path

    return Path, distanceX


def Line_OPEN_ST(Optimal_path_ts, CLOSED, Num_obs, Num_Opt, D):
    """
    优化路径，通过检查路径上的节点是否在一条直线上并且没有障碍物来减少路径节点。

    参数:
    Optimal_path_ts (np.ndarray): 初始的路径节点。
    CLOSED (np.ndarray): 障碍物位置。
    Num_obs (int): 障碍物数量。
    Num_Opt (int): 路径节点数量。
    D (float): 判断障碍物距离的阈值。

    返回:
    np.ndarray: 优化后的路径。
    """
    n = Num_Opt
    v = 0
    Optimal_path_st = np.zeros((Num_Opt, 4))

    # 初始化路径
    for q in range(Num_Opt):
        Optimal_path_st[q, 0] = Optimal_path_ts[n - 1 - q, 2]
        Optimal_path_st[q, 1] = Optimal_path_ts[n - 1 - q, 3]
        Optimal_path_st[q, 2] = Optimal_path_ts[n - 1 - q, 0]
        Optimal_path_st[q, 3] = Optimal_path_ts[n - 1 - q, 1]

    Optimal_path = Optimal_path_st

    # ---------------------------------------------------
    # 第一步：只处理 angle > 0.01 的情况（检查障碍、调整父节点）
    # ---------------------------------------------------
    v_temp = v
    vp_temp = v
    while not np.array_equal(Optimal_path[vp_temp, 2:4], Optimal_path[-1, 2:4]):
        parent_node = Optimal_path[v_temp, :2]  # 当前节点
        parentP_node = Optimal_path[vp_temp, 2:4]  # 当前节点的子节点
        index = Optimal_index(Optimal_path, parentP_node[0], parentP_node[1])
        parentPP_node = Optimal_path[index, 2:4]  # 子节点的子节点

        # 计算向量
        x_n1 = parentP_node[0] - parent_node[0]
        y_n1 = parentP_node[1] - parent_node[1]
        x_n2 = parentPP_node[0] - parentP_node[0]
        y_n2 = parentPP_node[1] - parentP_node[1]

        # 判断两个向量的夹角
        angle = myangle(x_n1, y_n1, x_n2, y_n2)

        # 只处理 angle > 0.01 的逻辑
        if angle > 0.01:
            x1, y1 = parent_node
            x3, y3 = parentPP_node
            k = (y3 - y1) / (x3 - x1)
            b = y1 - k * x1
            x_min, x_max = min(x1, x3), max(x1, x3)
            y_min, y_max = min(y1, y3), max(y1, y3)

            f = 0
            # 筛选出行驶区域内的点
            x_obn = CLOSED[:,1]
            y_obn = CLOSED[:,0]
            mask_area = (
                            (x_obn >= x_min) & (x_obn <= x_max) &
                            (y_obn >= y_min) & (y_obn <= y_max)
                            )
            x_obn = x_obn[mask_area]
            y_obn = y_obn[mask_area]
            # 距离公式：d = |k*x - y + b| / sqrt(k^2 + 1)
            dist = np.abs(k * x_obn - y_obn + b) / np.sqrt(k * k + 1)
            # 判断是否有任意一个点满足 d <= D
            if dist.size > 0 and np.any(dist <= D):
                f = 1

            if f == 0:
                # 无障碍，修改父节点
                Optimal_path[v_temp, 2] = Optimal_path[index, 2]
                Optimal_path[v_temp, 3] = Optimal_path[index, 3]
                vp_temp = v_temp
            else:
                if v_temp + 1 < index:  # 逐步遍历第一个点
                    # 有障碍，更新 v_temp
                    v_temp += 1
                else:  # 第一个点遍历结束更改第一个点为拐点
                    v_temp += 1
                    vp_temp = v_temp

        else:
            # angle <= 0.01 在这里不做处理
            vp_temp = index

     # Step1: 将第一行放入新变量
    new_array = Optimal_path[0].reshape(1, 4)
    # Step2: 当新变量末行与Optimal_path末行后两列内容不同时，继续循环
    while not np.array_equal(new_array[-1, 2:4], Optimal_path[-1, 2:4]):
        # 获取新变量末行的后两列
        x, y = new_array[-1, 2], new_array[-1, 3]

        # 根据(x,y)寻找在Optimal_path中的下一行索引
        idx = Optimal_index(Optimal_path, x, y)
        if idx < 0:
            # 未找到对应行，结束循环
            break
        # 获取对应行并堆叠到新变量
        new_array = np.vstack((new_array,Optimal_path[idx].reshape(1, 4)))
    Optimal_path = new_array

    # ---------------------------------------------------
    # 第二步：针对处理后的路径再进行 angle < 0.01 的操作
    # ---------------------------------------------------
    # 同样重新初始化或直接在已更新的 Optimal_path 上继续
    n_final = Optimal_path.shape[0]
    v_final = 0
    while n_final > 1:
        n_final -= 1
        parent_node = Optimal_path[v_final, :2]
        parentP_node = Optimal_path[v_final, 2:4]
        index = Optimal_index(Optimal_path, parentP_node[0], parentP_node[1])
        parentPP_node = Optimal_path[index, 2:4]

        x_n1 = parentP_node[0] - parent_node[0]
        y_n1 = parentP_node[1] - parent_node[1]
        x_n2 = parentPP_node[0] - parentP_node[0]
        y_n2 = parentPP_node[1] - parentP_node[1]
        angle = myangle(x_n1, y_n1, x_n2, y_n2)

        # 此处只检查 angle < 0.01
        if angle < 0.01:
            Optimal_path[v_final, 2] = Optimal_path[index, 2]
            Optimal_path[v_final, 3] = Optimal_path[index, 3]
        else:
            v_final = index

     # Step1: 将第一行放入新变量
    new_array = Optimal_path[0].reshape(1, 4)
    # Step2: 当新变量末行与Optimal_path末行后两列内容不同时，继续循环
    while not np.array_equal(new_array[-1, 2:4], Optimal_path[-1, 2:4]):
        # 获取新变量末行的后两列
        x, y = new_array[-1, 2], new_array[-1, 3]

        # 根据(x,y)寻找在Optimal_path中的下一行索引
        idx = Optimal_index(Optimal_path, x, y)
        if idx < 0:
            # 未找到对应行，结束循环
            break
        # 获取对应行并堆叠到新变量
        new_array = np.vstack((new_array,Optimal_path[idx].reshape(1, 4)))

    return new_array

def Line_OPEN_STtwo(Optimal_path, CLOSED, Num_obs, num_op, D):
    i_t = 0
    v = 1  # Optimal_path 路径节点
    line_open_t = [Optimal_path[0]]  # line_open_t 新的路径节点

    while v < num_op - 1:
        p_node = line_open_t[i_t]  # 当前点的坐标
        p_node_1 = Optimal_path[v]  # 当前点子节点的坐标
        p_node_2 = Optimal_path[v + 1]  # 当前点的子节点的子节点坐标
        v += 1

        line_node = LINE(p_node_1, p_node_2)  # 把1,2两节点之间连线上的节点求出来

        num_ln = line_node.shape[0]
        p_x, p_y = p_node

        # 判断当前点和线段上的节点之间是否有障碍物
        fn = False
        f = True
        for j in range(num_ln - 1, -1, -1):
            l_x, l_y = line_node[j]
            x_min = min(p_x, l_x) - 0.8
            x_max = max(p_x, l_x) + 0.8
            y_min = min(p_y, l_y) - 0.8
            y_max = max(p_y, l_y) + 0.8
            k = (l_y - p_y) / (l_x - p_x)
            b = p_y - k * p_x  # 两点之间直线方程 y = k * x + b

            for a in range(Num_obs):
                x_obn, y_obn = CLOSED[a]  # 判断障碍点是否在行驶区域内
                if x_min <= x_obn <= x_max and y_min <= y_obn <= y_max:
                    yline = x_obn * k + b
                    d = abs(y_obn - yline) * np.cos(np.arctan(k))
                    if d <= D:
                        f = False  # 两点之间有障碍物

            if f:
                node_l = [l_x, l_y]
                fn = True  # 判断有可连接点

        # 将可连接的第一个点存储到新的路径中，并做当前点
        i_t += 1

        if fn:
            line_open_t.append(node_l)
            if node_l == line_node[0].tolist():
                v += 1
        else:
            line_open_t.append(p_node_1.tolist())

    return np.array(line_open_t)


def LINE(p_node_2, p_node_3):
    x2, y2 = p_node_2
    x3, y3 = p_node_3

    line_node = [p_node_3]  # 初始化line_node列表
    a = 1

    # 判断直线类型
    if x3 == x2 and y3 != y2:
        w = 1  # 垂直的直线
    elif x3 != x2 and y3 == y2:
        w = 2  # 水平直线
    else:
        w = 3  # 斜线

    if w == 1:  # 垂直线
        n = -a if y3 > y2 else a
        for i in np.arange(y3, y2 + n, n):
            line_node.append([x3, i])
    elif w == 2:  # 水平线
        n = -a if x3 > x2 else a
        for i in np.arange(x3, x2 + n, n):
            line_node.append([i, y3])
    elif w == 3:  # 斜线
        n = -a if x3 > x2 else a
        k = (y3 - y2) / (x3 - x2)
        b = y2 - k * x2
        for i in np.arange(x3, x2 + n, n):
            line_node.append([i, k * i + b])

    return np.array(line_node)


def Line_obst(Obst_d_path_X, L_obst):
    num = Obst_d_path_X.shape[0]
    Obst_d_d_line = np.empty((0, 2), dtype=float)

    for v in range(num - 1):
        p_node_1 = Obst_d_path_X[v]  # 当前点子节点的坐标
        p_node_2 = Obst_d_path_X[v + 1]  # 当前点的子节点的子节点坐标

        line_node = LINE_o2(p_node_1, p_node_2, L_obst)  # 把1,2两节点之间连线上的节点求出来
        Obst_d_d_line = np.vstack((Obst_d_d_line, line_node))


    # 将多个数组垂直堆叠成一个数组
    return Obst_d_d_line


def LINE_o2(p_node_2, p_node_3, L_obst):
    """
    从 p_node_2 到 p_node_3 间，如果距离能被 L_obst 整除，
    则每隔 L_obst 插入一个点；否则在首尾间均匀插入，返回一个 (N, 2) 的 ndarray。
    """

    x2, y2 = p_node_2
    x3, y3 = p_node_3
    # 起终点向量
    dx, dy = x3 - x2, y3 - y2
    dist = np.hypot(dx, dy)  # 两点间距离
    # 若两点几乎重合，直接返回单点
    if dist < 1e-9:
        return np.array([[x2, y2]], dtype=float)
    # 方向单位向量
    direction = np.array([dx, dy], dtype=float) / dist
    # 计算可插入的段数
    n = int(dist // L_obst)  # 能够放下 n 个完整 L_obst 步长

    # 判断是否能被整除
    if abs(n * L_obst - dist) < 1e-9:
        # 可整除：每隔 L_obst 插点
        # n 段意味着有 n+1 个点(包含起点和终点)
        params = np.arange(n + 1)[:-1] * L_obst
    else:
        # 不可整除：进行均匀插值
        # 在 [0, dist] 间分成 n+1 段，即有 n+2 个点
        params = np.linspace(0, dist, n + 2)[:-1]

    # 根据距离序列和方向向量生成插值
    x_vals = x2 + params * direction[0]
    y_vals = y2 + params * direction[1]

    line_node = np.column_stack((x_vals, y_vals))
    return line_node

def Optimal_index(Optimal, xval, yval):
    # 此函数返回节点在列表OPEN中的位置索引
    #
    # Copyright 2009-2010 The MathWorks, Inc.
    i = 0  # Python中的索引从0开始
    # 从OPEN第一组数据开始遍历，确定当前点在哪一行
    while Optimal[i][0] != xval or Optimal[i][1] != yval:
        i += 1
    n_index = i  # 将当前点的OPEN数组的行数赋值出去
    return n_index


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
    # 计算可供遍历的范围: 0 ~ len(Path)-3
    for i in range(len(Path) - 2):
        x_n1 = Path[i + 1, 0] - Path[i, 0]
        y_n1 = Path[i + 1, 1] - Path[i, 1]
        x_n2 = Path[i + 2, 0] - Path[i + 1, 0]
        y_n2 = Path[i + 2, 1] - Path[i + 1, 1]
        angle = myangle(x_n1, y_n1, x_n2, y_n2)  # 自定义的角度计算函数
        if angle > 0.01:
            turn_idx_list.append(i + 1)

    turn_idx = np.array(turn_idx_list, dtype=int)

    # 将原路径切分并做插值 (若两转折点仅相差1, 则插一点)
    newPath_chunks = []
    # 1) 先添加起点到第一个转折点之前的段
    newPath_chunks.append(Path[:turn_idx[0]])

    # 2) 添加中间若干段
    for i in range(len(turn_idx) - 1):
        s, e = turn_idx[i], turn_idx[i + 1]
        newPath_chunks.append(Path[s:e])
        if s + 1 == e:
            # 在 s 和 e 之间插一个点
            numPoints = 1
            t = 1 / (numPoints + 1)
            interpolatedPoint = (1 - t) * Path[s] + t * Path[e]
            newPath_chunks.append(interpolatedPoint[np.newaxis, :])  # 保证是 2D 形状

    # 3) 最后一个转折点到结尾
    newPath_chunks.append(Path[turn_idx[-1]:])

    # 将各段拼接起来
    newPathArr = np.concatenate(newPath_chunks, axis=0)

    # ---------------------------
    # 第二次遍历, 找到 newPathArr 中的转折点并进行贝塞尔插值
    # ---------------------------
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
    # 1) 首先把开头到第一个转折点的段装入
    smoothPath_chunks.append(newPathArr[:turn_idx_2[0]])

    # 2) 在转折点间进行贝塞尔插值
    for i in range(len(turn_idx_2)):
        idx_cur = turn_idx_2[i]

        # 控制点计算: 分两段各插入 numPoints 个
        ControlPoint_list = []
        numPoints = 2

        # 第一段: 在 (idx_cur - 1) -> (idx_cur) 间线性插值
        for j in range(1, numPoints + 1):
            t = j / (numPoints + 1)
            interpolated = (1 - t) * newPathArr[idx_cur - 1] + t * newPathArr[idx_cur]
            ControlPoint_list.append(interpolated)

        # 第二段: 在 (idx_cur) -> (idx_cur + 1) 间线性插值
        for j in range(1, numPoints + 1):
            t = j / (numPoints + 1)
            interpolated = (1 - t) * newPathArr[idx_cur] + t * newPathArr[idx_cur + 1]
            ControlPoint_list.append(interpolated)

        # P0, P1, P2, P3
        ControlPoint = np.array(ControlPoint_list)  # 形状 (4, 2)
        P0, P1, P2, P3 = ControlPoint[0], ControlPoint[1], ControlPoint[2], ControlPoint[3]

        # 生成贝塞尔曲线
        t_vals = np.linspace(0, 1, 100)
        BezierPoints = np.zeros((len(t_vals), 2), dtype=float)
        for k, t_ in enumerate(t_vals):
            # 三次贝塞尔插值
            BezierPoints[k] = (1 - t_) ** 3 * P0 \
                              + 3 * (1 - t_) ** 2 * t_ * P1 \
                              + 3 * (1 - t_) * t_ ** 2 * P2 \
                              + t_ ** 3 * P3

        # 先将贝塞尔插值结果拼入
        smoothPath_chunks.append(BezierPoints)

        # 如果还未到最后一个转折点, 在贝塞尔曲线之后拼接 (idx_cur+1) 到下一转折点之间的部分
        if i < len(turn_idx_2) - 1:
            idx_next = turn_idx_2[i + 1]
            smoothPath_chunks.append(newPathArr[idx_cur + 1: idx_next])
        else:
            # 否则拼接末段
            smoothPath_chunks.append(newPathArr[idx_cur + 1:])

    # 最终平滑后的路径
    smoothPathArr = np.concatenate(smoothPath_chunks, axis=0)
    # 计算路径长度
    # 通过切片计算相邻点坐标差
    diff = smoothPathArr[1:] - smoothPathArr[:-1]  # 形状 (N-1, 2)
    # 计算每段距离，并求和
    distances = np.sqrt(np.sum(diff ** 2, axis=1))  # 形状 (N-1,)
    S = np.sum(distances)

    distanceX = S
    return smoothPathArr, distanceX


def myangle(a1, a2, b1, b2):
    # 计算两个向量之间的角度（以度为单位）
    n = np.array([a1, a2])
    m = np.array([b1, b2])

    # 计算向量n和m的点积
    dot_product = np.dot(n, m)

    # 计算向量的范数（模长）
    norm_n = np.linalg.norm(n)
    norm_m = np.linalg.norm(m)

    # 计算夹角的余弦值
    cos_angle = dot_product / (norm_n * norm_m)

    # 使用arccos计算角度，并将结果从弧度转换为度
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle


# 运行函数
IAFSA()
