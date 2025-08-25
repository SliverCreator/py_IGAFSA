import numpy as np
from .line_o2 import LINE_o2


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
