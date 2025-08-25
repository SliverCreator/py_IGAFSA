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
