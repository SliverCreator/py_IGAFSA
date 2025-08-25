import numpy as np
from .line_fn import LINE


def Line_OPEN_STtwo(Optimal_path, CLOSED, Num_obs, num_op, D):
    """
    基于几何开通性检验的三次折线优化：
    - 逐段在 p_node_1 -> p_node_2 的线段离散点中，从末端向前寻找
      与当前点 p_node 可直连且不过障的“最远候选点”；
    - 采用稳健的点到直线距离计算，显式处理竖直线，避免除零与 NaN；
    - 每个候选点独立评估，避免一次失败“污染”后续判断。
    """
    i_t = 0
    v = 1  # Optimal_path 路径节点游标
    line_open_t = [Optimal_path[0]]  # 新的路径节点列表（起点）

    CLOSED = np.asarray(CLOSED, dtype=float)  # (Num_obs, 2) 形如 [y, x] 或 [x, y] 需与调用处一致
    # 这里按原 smooth.py 的 CLOSED 组织方式使用：CLOSED[a] -> (i, j) 即 (y, x)
    # 因原实现中 x_obn, y_obn = CLOSED[a]，后续以 x_obn, y_obn 参与计算

    while v < num_op - 1:
        p_node = np.asarray(line_open_t[i_t], dtype=float)  # 当前点 [x, y]
        p_node_1 = np.asarray(Optimal_path[v], dtype=float)  # 子节点
        p_node_2 = np.asarray(Optimal_path[v + 1], dtype=float)  # 子节点的子节点
        v += 1

        # 取 p_node_1 -> p_node_2 连线上的离散点
        line_node = LINE(p_node_1, p_node_2)  # shape: (m, 2) 每行 [x, y]
        num_ln = line_node.shape[0]
        p_x, p_y = float(p_node[0]), float(p_node[1])

        # 在该段内，从末端往前寻找可直连的最远候选点
        found_connectable = False
        chosen_node = None

        for j in range(num_ln - 1, -1, -1):
            l_x, l_y = float(line_node[j, 0]), float(line_node[j, 1])

            # 每个候选点独立重置标志
            is_clear = True

            # 扩张的包围盒（与原代码一致）
            x_min = min(p_x, l_x) - 0.8
            x_max = max(p_x, l_x) + 0.8
            y_min = min(p_y, l_y) - 0.8
            y_max = max(p_y, l_y) + 0.8

            # 在包围盒中筛选可能阻挡的障碍点
            if Num_obs > 0:
                x_obn = CLOSED[:, 0]  # 注意：原实现中 x_obn, y_obn = CLOSED[a]
                y_obn = CLOSED[:, 1]
                mask = (x_obn >= x_min) & (x_obn <= x_max) & (y_obn >= y_min) & (y_obn <= y_max)
                x_cand = x_obn[mask]
                y_cand = y_obn[mask]

                if x_cand.size > 0:
                    # 直线 p_node -> (l_x, l_y):
                    dx = l_x - p_x
                    dy = l_y - p_y

                    if np.isclose(dx, 0.0):
                        # 竖直线：距离为 |x - p_x|
                        dist = np.abs(x_cand - p_x)
                        if np.any(dist <= D):
                            is_clear = False
                    else:
                        # 一般直线：y = kx + b
                        k = dy / dx
                        b = p_y - k * p_x
                        # 点到直线距离：|k*x - y + b| / sqrt(k^2 + 1)
                        dist = np.abs(k * x_cand - y_cand + b) / np.sqrt(k * k + 1)
                        if np.any(dist <= D):
                            is_clear = False

            if is_clear:
                chosen_node = [l_x, l_y]
                found_connectable = True
                break  # 选取最远可直连的候选点

        # 推进当前点
        i_t += 1

        if found_connectable:
            line_open_t.append(chosen_node)
            # 如果最远候选点恰好是 line_node 的第一点，则跨越一个段落
            if np.allclose(chosen_node, line_node[0].tolist()):
                v += 1
        else:
            # 无法直连任何候选点，退而求其次加入 p_node_1
            line_open_t.append(p_node_1.tolist())

    return np.array(line_open_t, dtype=float)
