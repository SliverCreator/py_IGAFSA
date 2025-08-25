import numpy as np


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
