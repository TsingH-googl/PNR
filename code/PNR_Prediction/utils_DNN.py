from scipy.sparse import csr_matrix
import numpy as np
import gc
from utils import get_row_col_index



# 注意：默认是进行归一化后的csr_matrix1
def better_show_grids(csr_matrix1=None):

    score_list = (np.array(csr_matrix1[csr_matrix1 > 0], dtype=float))[0]
    indices_nonzero = np.nonzero(csr_matrix1 != 0)

    sorted_score_list = list(np.sort(score_list))

    for i in range (len(score_list)):
        score_index = sorted_score_list.index(score_list[i])
        new_score = 0.1 * (score_index + 1)
        score_list[i] = new_score
    pass


    N = csr_matrix1.A.shape[0]

    DNN_raster_grids = csr_matrix((score_list, (indices_nonzero[0], indices_nonzero[1])), shape=(N, N), dtype=float)

    return DNN_raster_grids


def rasterization_grids(binNum=None,
                        train_binary=None,
                        scores_matrix_DNN=None,
                        scores_matrix_one_norm=None,
                        scores_matrix_two_norm=None):


    # 获取non-existing links的特征分数和对应的DNN模型训练的分数
    train_binary = csr_matrix(np.triu(train_binary.A, k=1))  # k=1表示不包括对角线
    negative_binary = csr_matrix(np.triu(np.ones(train_binary.shape), k=1) - train_binary.A)
    nonexist_scores_one_list = (np.array(scores_matrix_one_norm[negative_binary > 0], dtype=float))[0]
    nonexist_scores_two_list = (np.array(scores_matrix_two_norm[negative_binary > 0], dtype=float))[0]
    nonexist_scores_DNN_list = (np.array(scores_matrix_DNN[negative_binary > 0], dtype=float))[0]


    # 计算rasterization grids
    DNN_raster_grids = np.zeros((binNum, binNum), dtype=float)
    val_max = 1.0
    val_min = 0.0
    interval = float((val_max - val_min) / binNum)
    for i in range(len(nonexist_scores_one_list)):
        if (nonexist_scores_one_list[i] <= 0.0) & (nonexist_scores_two_list[i] <= 0.0):
            continue
        row_index = int(get_row_col_index(score=nonexist_scores_one_list[i], interval=interval, binNum=binNum))
        col_index = int(get_row_col_index(score=nonexist_scores_two_list[i], interval=interval, binNum=binNum))
        DNN_raster_grids[row_index, col_index] = nonexist_scores_DNN_list[i]
    pass

    return DNN_raster_grids


# 统计DNN计算的分数，返回矩阵形式
def predicted_scores_DNN(model=None,
                         train_binary=None,
                         test_binary=None,
                         scores_matrix_one_norm=None,
                         scores_matrix_two_norm=None):

    # 获取non-existing links的特征分数
    train_binary = csr_matrix(np.triu(train_binary.A, k=1))  # k=1表示不包括对角线
    negative_binary = csr_matrix(np.triu(np.ones(train_binary.shape), k=1) - train_binary.A)
    nonexist_scores_one_list = (np.array(scores_matrix_one_norm[negative_binary > 0], dtype=float))[0]
    nonexist_scores_two_list = (np.array(scores_matrix_two_norm[negative_binary > 0], dtype=float))[0]

    # 组合features
    negative_samples_all = (np.array([nonexist_scores_one_list, nonexist_scores_two_list])).T
    negative_samples_all = np.array(negative_samples_all)


    # 进行预测
    preds_proba_distribution = model.predict_proba(negative_samples_all)
    preds_scores = preds_proba_distribution[:, 1]
    indices_nonzero = np.nonzero(negative_binary != 0)

    # 组合好分数
    N = train_binary.shape[0]
    scores_matrix_DNN = csr_matrix((preds_scores, (indices_nonzero[0], indices_nonzero[1])), shape=(N, N), dtype=float)

    return scores_matrix_DNN


# 去掉两个feature values都为0的行
def flush_negative_samples(negative_samples_all=None):
    negative_samples_all = csr_matrix(negative_samples_all)
    negative_samples_flush = negative_samples_all[negative_samples_all.getnnz(1) > 0]

    negative_samples_flush = negative_samples_flush.A

    return negative_samples_flush


# 获取负样本
def negative_samples(train_binary=None,
                     test_binary=None,
                     scores_matrix_one_norm=None,
                     scores_matrix_two_norm=None,
                     ratio=None):

    # 标记positive或negative样本
    positive_binary = train_binary + test_binary# 改这里，使得

    # 按顺序准备features
    positive_binary = csr_matrix(np.triu(positive_binary.A, k=1))  # k=1表示不包括对角线
    negative_binary = csr_matrix(np.triu(np.ones(positive_binary.shape), k=1) - positive_binary.A)
    nonexist_scores_one_list = (np.array(scores_matrix_one_norm[negative_binary > 0], dtype=float))[0]
    nonexist_scores_two_list = (np.array(scores_matrix_two_norm[negative_binary > 0], dtype=float))[0]


    del positive_binary, negative_binary, \
        scores_matrix_one_norm, scores_matrix_two_norm
    gc.collect()



    # 组合features
    negative_samples_all = (np.array([nonexist_scores_one_list, nonexist_scores_two_list])).T
    negative_samples_all = np.array(negative_samples_all)

    # 去除features都是0的样本
    negative_samples_flush = flush_negative_samples(negative_samples_all=negative_samples_all)


    # 随机挑选负样本
    num_negative_samples = train_binary.nnz * ratio
    if negative_samples_flush.shape[0] > num_negative_samples:
        # 打乱每一行的顺序
        rand_index = np.arange(len(negative_samples_flush))
        np.random.shuffle(rand_index)
        negative_samples = negative_samples_flush[rand_index[0: num_negative_samples]]
        return negative_samples
    else:
        negative_samples = negative_samples_flush
        return negative_samples


    pass