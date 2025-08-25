import numpy as np


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
