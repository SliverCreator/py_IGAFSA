import numpy as np
from .myangle import myangle
from .optimal_index import Optimal_index


def Line_OPEN_ST(Optimal_path_ts, CLOSED, Num_obs, Num_Opt, D):
    """
    优化路径，通过检查路径上的节点是否在一条直线上并且没有障碍物来减少路径节点。
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
            # 竖直线分支，避免除零
            if x3 == x1:
                x_min, x_max = min(x1, x3), max(x1, x3)
                y_min, y_max = min(y1, y3), max(y1, y3)
                f = 0
                x_obn = CLOSED[:, 1]
                y_obn = CLOSED[:, 0]
                mask_area = (
                    (x_obn >= x_min) & (x_obn <= x_max) &
                    (y_obn >= y_min) & (y_obn <= y_max)
                )
                x_obn = x_obn[mask_area]
                # 与竖直线 x = x1 的距离为 |x - x1|
                if x_obn.size > 0 and np.any(np.abs(x_obn - x1) <= D):
                    f = 1
            else:
                k = (y3 - y1) / (x3 - x1)
                b = y1 - k * x1
                x_min, x_max = min(x1, x3), max(x1, x3)
                y_min, y_max = min(y1, y3), max(y1, y3)

                f = 0
                # 筛选出行驶区域内的点
                x_obn = CLOSED[:, 1]
                y_obn = CLOSED[:, 0]
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
        new_array = np.vstack((new_array, Optimal_path[idx].reshape(1, 4)))
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
        new_array = np.vstack((new_array, Optimal_path[idx].reshape(1, 4)))

    return new_array
