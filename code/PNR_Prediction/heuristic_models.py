import  numpy as np
import pandas as pd
from scipy import io  as sio
import networkx as nx
import linkpred
import os
from scipy.sparse import csr_matrix
from utils import get_trainset_path, plot_matrix

# 为了优化heuristic method的计算效率，如果存在该分数，则直接load进来，不要再计算一次
def is_heuristic_scores_exist(save_mat_path):

    return os.path.isfile(save_mat_path)


# 读取网络
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


# there are 13 heuristic methods I implemented
def heuristic_scores(all_file_dir=None,
                     graph_name=None,
                     graph_results_dir=None,
                     heuristic_method=None):

    save_mat_path = graph_results_dir + '//' + graph_name + '_' + heuristic_method + '_' + 'scores.mat'  # save_mat_name是文件夹下显示的名；
    if is_heuristic_scores_exist(save_mat_path):
        score_matrix_dict = sio.loadmat(save_mat_path)
        score_matrix_triu = csr_matrix(score_matrix_dict['scores'])
        return score_matrix_triu



    input=get_trainset_path(base_dir=all_file_dir,
                              graph_name=graph_name,
                              connected_pattern='directed',
                              from_zeros_one='0')
    G=read_graph(input=input) # 必须传入undirected graph！！！


    # neighbor_based (8)
    if heuristic_method=='cn':
        model = linkpred.predictors.CommonNeighbours(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method=='ra':
        model = linkpred.predictors.ResourceAllocation(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method == 'aa':
        model = linkpred.predictors.AdamicAdar(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method == 'ja':
        model = linkpred.predictors.Jaccard(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method == 'cosine':
        model = linkpred.predictors.Cosine(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method == 'pearson':
        model = linkpred.predictors.Pearson(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method == 'as':
        model = linkpred.predictors.AssociationStrength(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method == 'degreeproduct':
        model = linkpred.predictors.DegreeProduct(G)  # , excluded=G.edges()
        results = model.predict()


    # rank based
    elif heuristic_method == 'simrank':
        model = linkpred.predictors.SimRank(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method == 'rootedpagerank':
        model = linkpred.predictors.RootedPageRank(G)  # , excluded=G.edges()
        results = model.predict()


    # path based
    elif heuristic_method == 'graphdistance':
        model = linkpred.predictors.GraphDistance(G)  # , excluded=G.edges()
        results = model.predict()
    elif heuristic_method == 'katz':
        model = linkpred.predictors.Katz(G)  # , excluded=G.edges()
        results = model.predict()


    # community based（貌似没有这个了）
    elif heuristic_method == 'community':
        model = linkpred.predictors.Community(G)  # , excluded=G.edges()
        results = model.predict()
    else:
        print('error baseline name')
        pass


    # 格式化相似性分数并且保存
    score_list  = []
    nodeid_list = []
    for authors, score in results.items():
        temp = authors.elements
        #if float(score) > 0.0: # 谁告诉你分数就一定会小于0？？？
        nodeid_list.append(list(temp))
        score_list.append(score)
        pass
    pass

    score_array = np.array(score_list, dtype=float)
    nodeid_array = np.array(nodeid_list, dtype=int)

    # 得到score矩阵.
    N = len(G)
    score_matrix = np.zeros(shape=(N, N), dtype=float)
    score_matrix[nodeid_array[:, 0], nodeid_array[:, 1]] = score_array

    # 获取右上角的分数
    score_matrix_triu = csr_matrix(np.triu(score_matrix + score_matrix.T, k=1))
    # plot_matrix(score_matrix_triu.A)

    # save the scores
    sio.savemat(save_mat_path, {"scores": score_matrix_triu})  # "scores"是matlab下load进工作空间的


    return score_matrix_triu

