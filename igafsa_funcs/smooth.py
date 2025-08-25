import numpy as np
import matplotlib.pyplot as plt
from .line_open_st import Line_OPEN_ST
from .line_open_sttwo import Line_OPEN_STtwo
from .line_obst import Line_obst


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

    # 每隔2个取点，传入正序n*2,得到正序n*2
    L_obst = 2
    Obst_d_d_line = Line_obst(Optimal_path_one1, L_obst)
    Optimal_path_one1 = np.vstack([Obst_d_d_line,[xTarget,yTarget]])

    # 反过来排列路线节点
    part1 = Optimal_path_one1[1:][::-1]
    part2 = Optimal_path_one1[:-1][::-1]
    Optimal_path_1 = np.hstack((part1, part2))

    # 二次优化折线
    Num_Opt = Optimal_path_1.shape[0]
    Optimal_path_two = Line_OPEN_ST(Optimal_path_1, CLOSED, Num_obs, Num_Opt, D)
    _, ia = np.unique(Optimal_path_two[:, :2], axis=0, return_index=True)
    Optimal_path_two = Optimal_path_two[np.sort(ia)]
    Optimal_path_two2 = Optimal_path_two[:, 0:2]
    Optimal_path_two2 = np.vstack((Optimal_path_two2, Optimal_path_two[-1, 2:4]))

    diff = Optimal_path_two2[1:] - Optimal_path_two2[:-1]
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    S = np.sum(distances)
    print(f'二次IGAFSA+Smooth行走长度为: {S}')
    plt.plot(Optimal_path_two2[:, 0]+0.5, Optimal_path_two2[:, 1]+0.5, 'c',marker = 'o')

    # 三次折线
    num_optwo = Optimal_path_two2.shape[0]
    Optimal_path_three = Line_OPEN_STtwo(Optimal_path_two2, CLOSED, Num_obs, num_optwo, D)
    _, ia = np.unique(Optimal_path_three[:, :2], axis=0, return_index=True)
    Optimal_path_three = Optimal_path_three[np.sort(ia)]
    num_opthree = Optimal_path_three.shape[0]
    Optimal_path_three = np.vstack([Optimal_path_three, [xTarget, yTarget]])

    # 每隔2个取点
    L_obst = 2
    Obst_d_d_line = Line_obst(Optimal_path_three, L_obst)
    Obst_d_d_line = np.vstack([Obst_d_d_line,[xTarget,yTarget]])
    _, ia = np.unique(Obst_d_d_line[:, :2], axis=0, return_index=True)
    Obst_d_d_line = Obst_d_d_line[np.sort(ia)]

    NewOptimal_path = Obst_d_d_line
    _, ia = np.unique(NewOptimal_path[:, :2], axis=0, return_index=True)
    NewOptimal_path = NewOptimal_path[np.sort(ia)]

    plt.plot(NewOptimal_path[:, 0] + 0.5, NewOptimal_path[:, 1] + 0.5, 'g', linewidth=2,marker = 'o')

    diff = NewOptimal_path[1:] - NewOptimal_path[:-1]
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    S = np.sum(distances)

    distanceX = S
    Path = NewOptimal_path

    return Path, distanceX
