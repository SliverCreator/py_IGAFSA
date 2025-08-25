import numpy as np


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
