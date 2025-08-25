import time

import matplotlib.pyplot as plt
import numpy as np
from igafsa_funcs.grideaf_foodconsistence import GrideAF_foodconsistence
from igafsa_funcs.arry2orxy import arry2orxy
from igafsa_funcs.gridaf_prey import GridAF_prey
from igafsa_funcs.gridaf_swarm import GridAF_swarm
from igafsa_funcs.gridaf_follow import GridAF_follow
from igafsa_funcs.distance import distance
from igafsa_funcs.smooth import Smooth
from igafsa_funcs.bezier_smooth import bezierSmooth


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
                # 确保为标量再赋值，避免 DeprecationWarning
                position[j][i] = int(np.asarray(nextPosition).ravel()[0])
                H[i] = float(np.asarray(nextPositionH).ravel()[0])
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
                position[j][i] = int(np.asarray(nextPosition).ravel()[0])
                H[i] = float(np.asarray(nextPositionH).ravel()[0])

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
                d = distance(n, int(arrayValue[0,j]), int(arrayValue[0,j + 1]))
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
            BestH[0,i] = float(int(goal.item()) - int(arrayValue[0,i]))

        row, col = np.unravel_index(arrayValue.astype(int), (n, n))
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


# 运行函数
IAFSA()
