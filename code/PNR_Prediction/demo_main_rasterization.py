import scipy.sparse as sp
import numpy as np
from utils import *
import math
import matplotlib.pyplot as plt
import seaborn as sns




if __name__=='__main__':
    '-----------------------------------------------Done----------------------------------------------------'
    '1、我都用了csr_matrix取存储矩阵，目的是减少内存消耗'
    '2、已经检验过XX_score_XX_list等四个的一一对应正确性和数字的正确性，没问题的' \
    '---------------------------------------------To be done------------------------------------------------'
    '1、归一化：是两个scores矩阵各自归一化还是两个scores矩阵统一归一化？？？'
    '2、若最终计算的PNR很多0，我觉得可以在分母也就是n_n = n_n + 1使得inf和nan都去除了，不影响evaluation但是可能好看'


    print('Starting rasterization......')
    # 读入scores数据，取上三角
    scores_matrix_one = sp.csr_matrix([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
                                   [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
                                   [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
                                   [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
                                   [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0],
                                   [6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0],
                                   [7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0],
                                   [8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0],
                                   [9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]])
    scores_matrix_one = sp.csr_matrix(np.triu(scores_matrix_one.A, k=1))

    scores_matrix_two = sp.csr_matrix(np.random.random(scores_matrix_one.shape))
    scores_matrix_two = sp.csr_matrix(np.triu(scores_matrix_two.A, k=1))

    # 读入train的binary数据
    train_binary = sp.csr_matrix([
                           [0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # 构建exist和nonexist的binary
    exist_binary = sp.csr_matrix(np.triu(train_binary.A, k=1)) # k=1表示不包括对角线
    nonexist_binary = sp.csr_matrix(np.triu(np.ones(exist_binary.shape), k=1) - exist_binary.A)

    # 分数归一化到[0.0, 1.0]
    scores_matrix_one_norm = normalize(csr_matrix1 = scores_matrix_one)
    scores_matrix_two_norm = normalize(csr_matrix1 = scores_matrix_two)

    # 划分bin
    binNum = 10
    val_max = 1.0
    val_min = 0.0
    bin_array = sorted(divide_bin(val_max = val_max, val_min = val_min, binNum = binNum))
    interval = float((val_max - val_min) / binNum)

    # 获取exist_binary和nonexist_binary的分数
    exist_scores_one_list =    (np.array(scores_matrix_one_norm[exist_binary    > 0]))[0]
    nonexist_scores_one_list = (np.array(scores_matrix_one_norm[nonexist_binary > 0]))[0]
    exist_scores_two_list =    (np.array(scores_matrix_two_norm[exist_binary    > 0]))[0]
    nonexist_scores_two_list = (np.array(scores_matrix_two_norm[nonexist_binary > 0]))[0]

    # 初始化两个大小为binNum* bnNum的二维栅格
    exist_raster_grids = np.zeros((binNum, binNum))
    nonexist_raster_grids = np.zeros((binNum, binNum))

    # 计算落在exist_raster_grids栅格的existing links的数量
    exist_links_num = len(exist_scores_one_list)
    exist_row_col_zero_num = 0 # 那些两个矩阵的分数都是0的不作统计
    for i in range(exist_links_num):
        # row_index和col_index的范围从0-->binNum-1
        if (exist_scores_one_list[i] == 0) & (exist_scores_two_list[i] == 0):
            exist_row_col_zero_num = exist_row_col_zero_num + 1
            continue
        row_index = get_row_col_index(score = exist_scores_one_list[i], interval = interval, binNum = binNum)
        col_index = get_row_col_index(score = exist_scores_two_list[i], interval = interval, binNum = binNum)
        exist_raster_grids[row_index, col_index] = exist_raster_grids[row_index, col_index] + 1

    print("exist_row_col_zero_num:" + str(exist_row_col_zero_num))
    print('sum  exist_raster_grids:' + str(np.sum(exist_raster_grids)))


    # 计算落在nonexist_raster_grids栅格的nonexisting links的数量
    nonexist_links_num = len(nonexist_scores_one_list)
    nonexist_row_col_zero_num = 0 # 那些两个矩阵的分数都是0的不作统计
    for i in range(nonexist_links_num):
        # row_index和col_index的范围从0-->binNum-1
        if (nonexist_scores_one_list[i] == 0) & (nonexist_scores_two_list[i] == 0):
            nonexist_row_col_zero_num = nonexist_row_col_zero_num + 1
            continue
        row_index = get_row_col_index(score = nonexist_scores_one_list[i], interval = interval, binNum = binNum)
        col_index = get_row_col_index(score = nonexist_scores_two_list[i], interval = interval, binNum = binNum)

        nonexist_raster_grids[row_index, col_index] = nonexist_raster_grids[row_index, col_index] + 1


    print("nonexist_row_col_zero_num:" + str(nonexist_row_col_zero_num))
    print('sum  nonexist_raster_grids:' + str(np.sum(nonexist_raster_grids)))

    # 计算PNR分数
    N = (np.max(list(train_binary.shape)))
    L_T = np.sum(train_binary.A)
    O = N * (N - 1) / 2
    coefficient = (O - L_T) / L_T
    PNR1 = coefficient * (exist_raster_grids / (nonexist_raster_grids + 1)) # 分母加1避免出现inf或nan，不影响evaluation但是可能好看
    PNR2 = (exist_raster_grids / nonexist_raster_grids )  # inf和nan置为0
    PNR2[np.isnan(PNR2)] = 0
    PNR2[np.isinf(PNR2)] = 0

    sns.heatmap(PNR1, cmap='Reds')
    plt.show()
    sns.heatmap(PNR2, cmap='Reds')
    plt.show()
    # plt.matshow(PNR1) # 好丑
    # plt.show()







    pass