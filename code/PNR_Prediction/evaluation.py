import numpy as np
import heapq
import gc
from scipy.sparse import csr_matrix
from utils import get_row_col_index
from sklearn.metrics import average_precision_score, auc, roc_curve, accuracy_score, recall_score, roc_auc_score




# 计算AP，AUC等指标（train_binary和test_binary均为csr_matrix）
def evaluators(train_binary=None,
               test_binary =None,
               scores_list =None,
               L_array=None):
    exist_binary    = csr_matrix(np.triu(train_binary.A, k=1))  # k=1表示不包括对角线
    nonexist_binary = csr_matrix(np.triu(np.ones(exist_binary.shape), k=1) - exist_binary.A)
    nonexist_test_binary_list = (np.array(test_binary[nonexist_binary > 0], dtype=float))[0]

    del nonexist_binary, exist_binary
    gc.collect()


    # 对于sklearn中的roc_auc_score和AP的计算，必须不能是sparse的，因此，这里对于大规模的数据集，的确没办法啦。
    # scores_list=csr_matrix(scores_list)
    # nonexist_test_binary_list=csr_matrix(nonexist_test_binary_list)

    ap = average_precision_score(y_score=scores_list, y_true=nonexist_test_binary_list)
    # fpr, tpr, thresholds = roc_curve(y_score=scores_list, y_true=nonexist_test_binary_list)
    # auc_this = auc(fpr, tpr)
    # roc_auc_score_this = roc_auc_score(y_score=scores_list, y_true=nonexist_test_binary_list)# roc_auc_score与auc函数计算出来的AUC是一模一样的
    roc_auc_score_this = 0.0 # 我们不评估AUC


    L_array=np.array(L_array, dtype=int)
    precision=np.zeros(shape=L_array.shape, dtype=float)
    recall = np.zeros(shape=L_array.shape, dtype=float)
    F1score = np.zeros(shape=L_array.shape, dtype=float)
    for i in range(len(L_array)):
        L_array_i=L_array[i]
        hits_num = hits(scores_list=scores_list,
                        nonexist_test_binary_list=nonexist_test_binary_list,
                        L=L_array_i)
        # 求precision、recall、F1_score
        precision_i=hits_num / L_array_i
        recall_i=hits_num / np.sum(test_binary)
        F1score_i= (2*precision_i*recall_i)/(precision_i+recall_i)

        precision[i]=precision_i
        recall[i]=recall_i
        F1score[i]=F1score_i
    pass


    return ap, roc_auc_score_this, precision, recall, F1score



# PNR修改分数，高效的方法（确认正确）
def transfer_scores_PNR(
                    scores_matrix_one_norm=None,
                    scores_matrix_two_norm=None,
                    train_binary=None,
                    PNR         =None,
                    interval    =None,
                    binNum      =None):

    exist_binary    = csr_matrix(np.triu(train_binary.A, k=1))  # k=1表示不包括对角线
    nonexist_binary = csr_matrix(np.triu(np.ones(exist_binary.shape), k=1) - exist_binary.A)

    nonexist_scores_one_list = (np.array(scores_matrix_one_norm[nonexist_binary > 0], dtype=float))[0]
    nonexist_scores_two_list = (np.array(scores_matrix_two_norm[nonexist_binary > 0], dtype=float))[0]

    del exist_binary, nonexist_binary
    gc.collect()


    nonexist_links_num = len(nonexist_scores_one_list)
    nonexist_scores_PNR_list = np.zeros(nonexist_links_num, dtype=float)
    for i in range(nonexist_links_num):
        # row_index和col_index的范围从0-->binNum-1
        if (nonexist_scores_one_list[i] <= 0.0) & (nonexist_scores_two_list[i] <= 0.0):
            continue
        row_index = int(get_row_col_index(score = nonexist_scores_one_list[i], interval = interval, binNum = binNum))
        col_index = int(get_row_col_index(score = nonexist_scores_two_list[i], interval = interval, binNum = binNum))
        PNR_score = PNR[row_index, col_index]
        nonexist_scores_PNR_list[i] = PNR_score

    return nonexist_scores_PNR_list



# # （这个太慢了，不用）计算two-dimensional PNR调整之后的分数；入参为稀疏矩阵
# def transfer_scores_PNR(
#                     scores_matrix_one_norm=None,
#                     scores_matrix_two_norm=None,
#                     train_binary=None,
#                     PNR=None,
#                     interval=None,
#                     binNum=None):
#     exist_binary    = csr_matrix(np.triu(train_binary.A, k=1))  # k=1表示不包括对角线
#     nonexist_binary = csr_matrix(np.triu(np.ones(exist_binary.shape), k=1) - exist_binary.A)
#
#     N = train_binary.shape[0]
#     scores_matrix_PNR = csr_matrix(np.zeros(shape=(N, N)))
#     for i in range(N):
#         for j in range(i+1, N):
#             if(nonexist_binary[i, j] > 0):
#                 row_index = get_row_col_index(score=scores_matrix_one_norm[i, j], interval=interval, binNum=binNum)
#                 col_index = get_row_col_index(score=scores_matrix_two_norm[i, j], interval=interval, binNum=binNum)
#                 PNR_score = PNR[row_index, col_index]
#                 scores_matrix_PNR[i, j] = PNR_score
#             pass
#         pass
#     pass
#
#     return scores_matrix_PNR




# 计算在预测长度为L下，hit中了多少边
def hits(scores_list=None, nonexist_test_binary_list=None, L=None):

    scores_list = (np.array(scores_list, dtype=float))
    nonexist_test_binary_list = (np.array(nonexist_test_binary_list, dtype=int))
    prediction_list = np.zeros(len(scores_list),dtype=int)


    # max_num_index_list = list(map(scores_list.index, heapq.nlargest(L, scores_list))) # 奇慢
    # prediction_list[max_num_index_list] = 1

    idx = np.argpartition(scores_list, -L) # 这个很快
    prediction_list[idx[-L:]]=1

    # prediction_list[scores_list.argsort()[-L:][::-1]] = 1 # 这个也很快


    hits_num = np.sum((prediction_list .dot (nonexist_test_binary_list)))


    return hits_num


    pass


