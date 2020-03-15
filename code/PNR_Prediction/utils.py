import scipy.sparse as sp
import numpy as np
import math
import networkx as nx
from scipy.sparse import csr_matrix
from scipy import io as sio
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 为了计算arope方法
import sys
import configparser
sys.path.append(r'D:\hybridrec\code\githubcode\AROPE-master\python')
sys.path.append(r'D:\hybridrec\code\githubcode\graph2gauss-master')
from main_arope import arope_main
from main_graph2gauss import graph2gauss_main



# 获取list的前L个最大值的阈值
def get_list_thresold(scores_list=None, L=None):
    scores_list = sorted(scores_list, reverse=True)
    return scores_list[L-1]

    pass


# 保存plus进行hybrid的分数
def save_plus_hybrid_scores(scores_matrix_plus=None,
                           method1=None, method2=None,
                           graph_results_dir=None,
                           dataset_name=None):
    save_mat_name = graph_results_dir + '//' + "plus_" + dataset_name + '_' + method1 + '_' + method2 + '_' + 'scores.mat' # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"scores": scores_matrix_plus}) # "scores"是matlab下load进工作空间的

# 保存plus的raster_grids
def save_plus_raster_scores(rastser_grids=None,
                           method1=None, method2=None,
                           graph_results_dir=None,
                           dataset_name=None,
                           plus_binNum=None):
    save_mat_name = graph_results_dir + '//' + 'plus' + "_" + dataset_name + \
                    '_' + method1 + \
                    '_' + method2 + \
                    '_' + str(plus_binNum) +\
                    '_count.mat' # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"count": rastser_grids}) # "scores"是matlab下load进工作空间的




# 保存multiply进行hybrid的分数
def save_multiply_hybrid_scores(scores_matrix_multiply=None,
                           method1=None, method2=None,
                           graph_results_dir=None,
                           dataset_name=None):
    save_mat_name = graph_results_dir + '//' + "multiply_" + dataset_name + '_' + method1 + '_' + method2 + '_' + 'scores.mat' # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"scores": scores_matrix_multiply}) # "scores"是matlab下load进工作空间的

# 保存multiply的raster_grids
def save_multiply_raster_scores(rastser_grids=None,
                           method1=None, method2=None,
                           graph_results_dir=None,
                           dataset_name=None,
                           multiply_binNum=None):
    save_mat_name = graph_results_dir + '//' + 'multiply' + "_" + dataset_name + \
                    '_' + method1 + \
                    '_' + method2 + \
                    '_' + str(multiply_binNum) +\
                    '_count.mat' # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"count": rastser_grids}) # "scores"是matlab下load进工作空间的


# 保存DNN的raster_grids
def save_DNN_raster_scores(rastser_grids=None,
                           method1=None, method2=None,
                           graph_results_dir=None,
                           dataset_name=None,
                           model_name=None,
                           DNN_binNum=None):
    save_mat_name = graph_results_dir + '//' + model_name + "_" + dataset_name + \
                    '_' + method1 + \
                    '_' + method2 + \
                    '_' + str(DNN_binNum) +\
                    '_count.mat' # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"count": rastser_grids}) # "scores"是matlab下load进工作空间的


# 保存DNN进行hybrid的分数
def save_DNN_hybrid_scores(scores_matrix_DNN=None,
                           method1=None, method2=None,
                           graph_results_dir=None,
                           dataset_name=None,
                           model_name=None):
    save_mat_name = graph_results_dir + '//' + model_name + "_" + dataset_name + '_' + method1 + '_' + method2 + '_' + 'scores.mat' # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"scores": scores_matrix_DNN}) # "scores"是matlab下load进工作空间的


# # 根据数据集类型获取embedding的大小（因为有一些数据集太小了，需要对某些baslines的hyperparameters做一些调整）
# def get_embedding_size(baseline_name=None, dataset_name=None, conf=None):
#     if dataset_name==

# 获取测试集的文件路径
def get_testset_path(base_dir=None, graph_name=None):
    'connected_pattern: str, undirected or directed'
    'from_zeros_one: str, 0 or 1'

    graph_test_name = 'test_' + graph_name + '_directed_0_giantCom.edgelist' # 注意，一定要是从0开始且是directed(triu)，我在matlab处理的时候已经保证test、train都是triu的啦！
    graph_test_path = base_dir + graph_name + '//' + graph_test_name

    return graph_test_path


# 在N*N的矩阵下，以稀疏矩阵的形式返回测试集
def get_test_matrix_binary(graph_test_path=None, N=None):

    test_binary_array = np.loadtxt(graph_test_path, delimiter=' ', dtype=int)

    test_matrix_binary = np.zeros(shape=(N, N), dtype=int)
    test_matrix_binary[test_binary_array[:, 0], test_binary_array[:, 1]] = 1

    return csr_matrix(test_matrix_binary)



# 展示matrix
def plot_matrix(matrix=None):
    # plt.matshow(matrix)
    sns.heatmap(np.array(matrix), cmap='Reds')
    plt.show()



# 计算heuristic 方法的分数
def is_heuristic_method(method_name=None):
     heuristic_names = ['cn', 'aa', 'ja', 'ra',
                        'cosine', 'pearson', 'as', 'degreeproduct',
                        'simrank', 'rootedpagerank',
                        'graphdistance', 'katz',
                        'community']
     # for i in range(len(heuristic_names)):
     #     if heuristic_names[i] == method_name:
     #         return True
     heuristic_names_set = set(heuristic_names)
     if method_name in heuristic_names_set:
         return True
     else:
         return False


     # return False


# compute graph2gauss method scores
def energy_kl_scores_graph2gauss(all_file_dir = None, graph_name=None, graph_results_dir = None):
    input = get_trainset_path(base_dir=all_file_dir,
                              graph_name=graph_name,
                              connected_pattern=get_connp('graph2gauss'),
                              from_zeros_one=get_from_zeros_one('graph2gauss'))
    output = graph_results_dir + '//' + graph_name + '_graph2gauss.mat'

    config_path = 'conf/graph2gauss.properties'
    config = configparser.ConfigParser()
    config.read(config_path)
    conf = dict(config.items("hyperparameters"))
    return graph2gauss_main(input=input, output=output,
                            L=int(conf['embedding_size']), K=int(conf['K']),
                            p_val=float(conf['p_val']), p_test=float(conf['p_test']),
                            n_hidden=eval(conf['n_hidden']), max_iter=int(conf['max_iter']))


# 特别地，计算arope的分数
def inner_product_scores_arope(all_file_dir = None, graph_name=None, graph_results_dir = None):
    input = get_trainset_path(base_dir=all_file_dir,
                              graph_name=graph_name,
                              connected_pattern=get_connp('arope'),
                              from_zeros_one=get_from_zeros_one('arope'))
    output_dir = graph_results_dir + '//'
    config_path = 'conf/arope.properties'
    config = configparser.ConfigParser()
    config.read(config_path)
    conf = dict(config.items("hyperparameters"))
    dimensions = int(conf["embedding_size"])
    order = eval(conf["order"])
    weights = eval(conf["weights"])

    scores=arope_main(input=input, output_dir=output_dir,
                      dimensions=dimensions, order=order,
                      weights=weights)
    return scores


# 保存exist_raster_grids和nonexist_raster_grids、PNR ndarray对象为.mat文件
def save_ndarray_to_mat(ndarray_data, prex, graph_results_dir, dataset_name, emb_method_name1, emb_method_name2, binNum):
    csr_ndarray = csr_matrix(ndarray_data)
    save_mat_name = graph_results_dir + '//' \
                    + prex + '_' \
                    + dataset_name \
                    + '_' + emb_method_name1 + '_'+ emb_method_name2 + '_' \
                    + str(binNum)+'_'\
                    + 'count.mat'  # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"count": csr_ndarray})  # "scores"是matlab下load进工作空间的


# prone方法计算两个结点之间的maximum dot product
def get_node_i_j_scores(nodei, nodej, json_string_transver, emb):
    scores_list = []
    emb_group_nodei_ids = json_string_transver[nodei]
    emb_group_nodej_ids = json_string_transver[nodej]

    # 获取结点nodei和nodej的所有embs
    emb_group_nodei=[]
    emb_group_nodej=[]
    for k in range(0, len(emb_group_nodei_ids)):
        emb_group_nodei.append(emb[int(emb_group_nodei_ids[k])])
    for k in range(0, len(emb_group_nodej_ids)):
        emb_group_nodej.append(emb[int(emb_group_nodej_ids[k])])

    for p in range(0, len(emb_group_nodei_ids)):
        for q in range(0, len(emb_group_nodej_ids)):
            temp_emb1 = emb_group_nodei[p]
            temp_emb2 = emb_group_nodej[q]
            temp_score = temp_emb1 .dot (temp_emb2)
            scores_list.append(temp_score)


    return np.max(scores_list)


# 获取prone的分数稀疏矩阵
def get_scores(emb=None, input_json=None):
    with open(input_json, 'r') as jsonfile:
        json_string = json.load(jsonfile)

    json_string_transver=defaultdict(list)
    for k, v in json_string.items():
        json_string_transver[v].append(k)

    N=len(json_string_transver)
    scores = np.zeros((N, N), dtype=float)

    for nodei in range(0, N):
        for nodej in range(nodei+1, N):
            temp_max_score = get_node_i_j_scores(nodei=nodei, nodej=nodej, json_string_transver = json_string_transver, emb=emb)
            scores[nodei][nodej]=temp_max_score

    return csr_matrix(scores)


# 由于Splitter的每一个结点都有1或多个embs，这里特殊处理计算inner product
def inner_product_scores_splitter(graph_results_dir = None,
                                 dataset_name = None, emb_method_name = None,
                                 col_start = None, col_end = None, skiprows = None, delimiter=','):

    emb_input = graph_results_dir + '/' + dataset_name + '_' + emb_method_name + '.emb'
    json_input = graph_results_dir + '/' + dataset_name + '_' + emb_method_name +'.json'
    emb = np.loadtxt(emb_input,  delimiter=delimiter, usecols= np.arange(col_start, col_end), skiprows= skiprows, dtype=float) # cols从col_strat~col_end-1，下标从0开始

    # 根据nodeid排序
    emb = emb[emb[:, 0].argsort()]

    # 去掉第一列的nodeid
    emb = np.delete(emb, [0], axis=1)  # [0]和axis=1：删除第“0”“列”

    # 获取分数，稀疏矩阵
    inner_product_scores = get_scores(emb=emb, input_json=json_input)

    # 把csr_matrix的scores分数保存为.mat文件; save_mat_name例子：blogcatalog_node2vec_scores.mat
    save_mat_name = graph_results_dir + '//' + dataset_name + '_' + emb_method_name + '_' + 'scores.mat' # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"scores": inner_product_scores}) # "scores"是matlab下load进工作空间的

    return inner_product_scores


# 根据不同的方法的数据集输入要求，返回输入数据集的连边模式
def get_connp(graph_name=None):
    dict = {'deepwalk': 'directed',# deepwalk directed或undirected都可以
            'node2vec': 'directed',
            'splitter': 'directed',
            'prone': 'directed', # prone未确定
            'attentionwalk': 'directed',# attentionwalk 为directed
            'grarep': 'directed',
            'sdne': 'directed',
            'struc2vec':'directed',
            'line':'directed',
            'drne': 'directed',
            'arope': 'directed',
            'prune':'undirected', # prune必须为undirected
            'graph2gauss':'directed'} # 未确定
    return dict[graph_name]


def get_from_zeros_one(graph_name=None):
    dict = {'deepwalk': '0', # deepwalk 0或1都可以
            'node2vec': '0', # node2vec 0或1都可以
            'splitter': '0',
            'prone':    '0', # prone只能为0
            'attentionwalk':'0',# attentionwalk只能为0
            'grarep': '0',
            'sdne':'0',
            'struc2vec':'0',
            'line':'0',
            'drne':'0',  # drne必须从0开始
            'arope':'0',
            'prune':'0',# prune必须从0开始
            'graph2gauss':'0'} # 未确定
    return dict[graph_name]





def get_trainset_path(base_dir=None, graph_name=None,
                      connected_pattern=None, from_zeros_one=None):
    'connected_pattern: str, undirected or directed'
    'from_zeros_one: str, 0 or 1'

    graph_train_name = 'train_' + graph_name + '_' + connected_pattern + '_' + from_zeros_one + '_giantCom.edgelist'
    graph_train_path = base_dir + graph_name + '//' + graph_train_name

    return graph_train_path


# 对整个矩阵进行normalize，注意：分数矩阵左下三角矩阵都是0
# def normalize_matrix(csr_matrix1):
#     array_matrix = csr_matrix1.A
#     amin, amax = array_matrix.min(), array_matrix.max()# 求最大最小值
#     array_matrix = (array_matrix-amin)/(amax-amin)# (矩阵元素-最小值)/(最大值-最小值)
#
#     return sp.csr_matrix(array_matrix)

# ego-twitter可以用这个试试看是否会改变内存不够的问题，注意，用了这个，facebook_combined我测试了，在AP和precision上有一点点的不同
# def normalize_matrix(csr_matrix1):
#     all_binary = csr_matrix(np.triu(np.ones(csr_matrix1.shape), k=1))
#     all_scores = csr_matrix((np.array(csr_matrix1[all_binary > 0], dtype=float))[0])
#
#     array_matrix = csr_matrix1.A # 不能是稀疏矩阵，否则出错
#     # amin, amax = array_matrix.min(), array_matrix.max()# 求最大最小值
#     amin = all_scores.data.min()
#     amax = all_scores.data.max()
#     array_matrix = (array_matrix-amin)/(amax-amin)# (矩阵元素-最小值)/(最大值-最小值)
#
#     return csr_matrix(array_matrix)

def normalize_matrix_full(csr_matrix1):
    all_binary = csr_matrix((np.ones(csr_matrix1.shape)))
    all_scores = (np.array(csr_matrix1[all_binary > 0], dtype=float))[0]

    array_matrix = csr_matrix1.A
    # amin, amax = array_matrix.min(), array_matrix.max()# 求最大最小值
    amin = min(all_scores)
    amax = max(all_scores)
    array_matrix = (array_matrix-amin)/(amax-amin)# (矩阵元素-最小值)/(最大值-最小值)

    return csr_matrix(array_matrix)


def normalize_matrix(csr_matrix1):
    all_binary = csr_matrix(np.triu(np.ones(csr_matrix1.shape), k=1))
    all_scores = (np.array(csr_matrix1[all_binary > 0], dtype=float))[0]

    array_matrix = csr_matrix1.A
    # amin, amax = array_matrix.min(), array_matrix.max()# 求最大最小值
    amin = min(all_scores)
    amax = max(all_scores)
    array_matrix = (array_matrix-amin)/(amax-amin)# (矩阵元素-最小值)/(最大值-最小值)

    return csr_matrix(array_matrix)

# # 对整个矩阵进行normalize(跟上面不同的是，上面的amin肯定为0)，注意：分数矩阵左下三角矩阵都是0
# def normalize_matrix2(csr_matrix1, N):
#     array_matrix = csr_matrix1.A
#     amax = array_matrix.max()# 求最大
#     ones_matrix = np.triu(np.ones(N), k=1)
#     data_list = np.array(array_matrix[ones_matrix > 0])
#     amin = data_list.min()
#     array_matrix = (array_matrix-amin)/(amax-amin)# (矩阵元素-最小值)/(最大值-最小值)
#
#     return sp.csr_matrix(array_matrix)

# def divide_bin(val_max, val_min, binNum):
#     interval = (val_max - val_min) / binNum
#     bin_array = []
#     bin_array.append(val_min)
#     for i in range(binNum-1):
#         bin_array.append(val_min + (i+1)*interval)
#     bin_array.append(val_max)
#
#     return bin_array

def get_row_col_index(score, interval, binNum):
    if (score <= 1.0) & (score >= (interval * (binNum-1))):
        return binNum - 1
    pass
    if (score >= 0.0) & (score < interval): # 这里score >= 0必须有==0因为可能会row_score等于0而col_score不等于0
        return 0

    return int(math.floor(score / interval))


# 应该是从1开始
def read_graph(weighted=0, input=None, directed=0):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),),
                             create_using=nx.DiGraph(), delimiter=' ')
    else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph(), delimiter=' ')
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G


# 计算inner product的分数
def inner_product_scores(graph_results_dir = None,
                                 dataset_name = None, emb_method_name = None,
                                 col_start = None, col_end = None, skiprows = None, delimiter=' '):

    emb_input = graph_results_dir + '/' + dataset_name + '_' + emb_method_name + '.emb'
    emb = np.loadtxt(emb_input, usecols= np.arange(col_start, col_end), skiprows= skiprows, delimiter=delimiter, dtype=float) # cols从col_strat~col_end-1，下标从0开始

    # 根据nodeid排序
    emb = emb[emb[:, 0].argsort()]

    if not emb_method_name == 'drne':
        # 去掉第一列的nodeid
        emb = np.delete(emb, [0], axis=1)  # [0]和axis=1：删除第“0”“列”

    emb_transpose = np.transpose(emb)
    inner_product_scores = (emb).dot (emb_transpose)

    # 把csr_matrix的scores分数保存为.mat文件; save_mat_name例子：blogcatalog_node2vec_scores.mat
    save_mat_name = graph_results_dir + '//' + dataset_name + '_' + emb_method_name + '_' + 'scores.mat' # save_mat_name是文件夹下显示的名；
    sio.savemat(save_mat_name, {"scores": inner_product_scores}) # "scores"是matlab下load进工作空间的

    return csr_matrix(inner_product_scores)






