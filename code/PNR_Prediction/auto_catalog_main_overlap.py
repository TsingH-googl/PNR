from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import os
import gc
import numpy as np
import networkx as nx
import time
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# import evaluators, stacking, etc.
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
# Package for stacking models
from vecstack import stacking
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import cv2
import warnings
warnings.filterwarnings("ignore")


import scipy.misc


from utils import read_graph, normalize_matrix, normalize_matrix_full, get_trainset_path, get_testset_path, get_test_matrix_binary, \
    save_DNN_hybrid_scores, save_plus_raster_scores, save_plus_hybrid_scores, get_list_thresold
from utils_DNN import negative_samples, predicted_scores_DNN, rasterization_grids, better_show_grids
from evaluation import evaluators
from fileIO_utils import is_excel_file_exist, plus_write_to_excel
from utils_plot import plot_contourf, plot_heatmap, plot_contourf_overlap
from evaluation import transfer_scores_PNR, evaluators

all_heuristic_methods = ['cn', 'ja', 'aa', 'ra', 'cosine', 'pearson', 'degreeproduct', 'simrank']
all_embedding_methods = ['arope', 'drne', 'graph2gauss', 'prone', 'attentionwalk', 'deepwalk', 'node2vec','splitter', 'prune']



def auto_overlap(prex=None,
                  graph_name=None,
                  emb_method_name1=None,
                  emb_method_name2=None,
                  binNum=None):
    time_start = time.time()
    print('----------------------------------------------------------')
    print("dataset: " + graph_name + '\n' + "baselines:" + emb_method_name1 + "," + emb_method_name2)



    results_base_dir = 'D:\hybridrec//results//'
    all_file_dir = 'D:\hybridrec\dataset\split_train_test//' + prex
    results_dir = 'D:\hybridrec/results//' + prex
    graph_results_dir = results_dir + graph_name + '//'


    path_scores_method1 = results_base_dir + prex + graph_name + "//" + graph_name + "_" + emb_method_name1 + "_scores.mat"
    path_scores_method2 = results_base_dir + prex + graph_name + "//" + graph_name + "_" + emb_method_name2 + "_scores.mat"


    if not (os.path.exists(path_scores_method1) and os.path.exists(path_scores_method2)):
        print("dataset: " + graph_name + '----' + "baselines:" + emb_method_name1 + "," + emb_method_name2 + ': 分数未完全计算')


    if os.path.exists(path_scores_method1) and os.path.exists(path_scores_method2):
        # 获取归一化分数
        scores_matrix_one_dict = (loadmat(path_scores_method1))
        scores_matrix_two_dict = (loadmat(path_scores_method2))
        scores_matrix_one = scores_matrix_one_dict['scores']
        scores_matrix_two = scores_matrix_two_dict['scores']
        if emb_method_name1 not in all_embedding_methods:
            scores_matrix_one = csr_matrix(np.triu(scores_matrix_one.A, k=1))  # k=1表示不包括对角线
        if emb_method_name2 not in all_embedding_methods:
            scores_matrix_two = csr_matrix(np.triu(scores_matrix_two.A, k=1))
        scores_matrix_one_norm = normalize_matrix(csr_matrix1=csr_matrix(scores_matrix_one))# 去掉传参的csr_matrix()则会
        scores_matrix_two_norm = normalize_matrix(csr_matrix1=csr_matrix(scores_matrix_two))

        # 获取train_binary和test_binary
        graph_train_path = get_trainset_path(base_dir=all_file_dir,
                                             graph_name=graph_name,
                                             connected_pattern='undirected',
                                             from_zeros_one='0')
        graph_test_path = get_testset_path(base_dir=all_file_dir, graph_name=graph_name)
        G = read_graph(weighted=0, input=graph_train_path, directed=0)
        train_binary = csr_matrix(nx.convert_matrix.to_scipy_sparse_matrix(G))
        train_binary = csr_matrix(np.triu(train_binary.A, k=1))
        test_binary = get_test_matrix_binary(graph_test_path=graph_test_path, N=train_binary.shape[0])




        # 读取plus的原始分数（未归一化）
        plus_scores_name= 'plus_' + graph_name + '_' + emb_method_name1 + '_' + emb_method_name2 + '_scores.mat'
        plus_scores_path = graph_results_dir + plus_scores_name
        scores_matrix_plus_dict = (loadmat(plus_scores_path))
        scores_matrix_plus = scores_matrix_plus_dict['scores']

        # 读取multiply的原始分数（未归一化）
        multiply_scores_name= 'multiply_' + graph_name + '_' + emb_method_name1 + '_' + emb_method_name2 + '_scores.mat'
        multiply_scores_path = graph_results_dir + multiply_scores_name
        scores_matrix_multiply_dict = (loadmat(multiply_scores_path))
        scores_matrix_multiply = scores_matrix_multiply_dict['scores']

        # 读取MLP的原始分数（未归一化）
        mlp_scores_name= 'mlp_' + graph_name + '_' + emb_method_name1 + '_' + emb_method_name2 + '_scores.mat'
        mlp_scores_path = graph_results_dir + mlp_scores_name
        scores_matrix_mlp_dict = (loadmat(mlp_scores_path))
        scores_matrix_mlp = scores_matrix_mlp_dict['scores']



        # 归一化hybrid分数
        scores_matrix_plus_norm = normalize_matrix(csr_matrix1=scores_matrix_plus)
        scores_matrix_multiply_norm = normalize_matrix(csr_matrix1=scores_matrix_multiply)
        scores_matrix_mlp_norm = normalize_matrix(csr_matrix1=scores_matrix_mlp)

        # 计算plus、multiply、mlp、PNR的rasterization grids
        mlp_path = results_base_dir + prex + graph_name + "//" + "mlp_" + graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 + "_50_count.mat"
        mlp_dict = (loadmat(mlp_path))
        mlp_raster_grids = mlp_dict["count"]
        multiply_path = results_base_dir + prex + graph_name + "//" + "multiply_" + graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 + "_50_count.mat"
        multiply_dict = (loadmat(multiply_path))
        multiply_raster_grids = multiply_dict["count"]
        plus_path = results_base_dir + prex + graph_name + "//" + "plus_" + graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 + "_50_count.mat"
        plus_dict = (loadmat(plus_path))
        plus_raster_grids = plus_dict["count"]

        # plus_raster_grids = rasterization_grids(binNum=binNum,
        #                                        train_binary=train_binary,
        #                                        scores_matrix_DNN=scores_matrix_plus_norm,
        #                                        scores_matrix_one_norm=scores_matrix_one_norm,
        #                                        scores_matrix_two_norm=scores_matrix_two_norm)
        # multiply_raster_grids = rasterization_grids(binNum=binNum,
        #                                        train_binary=train_binary,
        #                                        scores_matrix_DNN=scores_matrix_multiply_norm,
        #                                        scores_matrix_one_norm=scores_matrix_one_norm,
        #                                        scores_matrix_two_norm=scores_matrix_two_norm)
        # mlp_raster_grids = rasterization_grids(binNum=binNum,
        #                                        train_binary=train_binary,
        #                                        scores_matrix_DNN=scores_matrix_mlp_norm,
        #                                        scores_matrix_one_norm=scores_matrix_one_norm,
        #                                        scores_matrix_two_norm=scores_matrix_two_norm)
        PNR_path = results_base_dir + prex + graph_name + "//" + "PNR2_" + graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 + "_50_count.mat"
        PNR_dict = (loadmat(PNR_path))
        PNR_raster_grids = PNR_dict["count"]




        exist_binary = csr_matrix(np.triu(train_binary.A, k=1))  # k=1表示不包括对角线
        nonexist_binary = csr_matrix(np.triu(np.ones(exist_binary.shape), k=1) - exist_binary.A)
        # 获取plus的nonexist_scores_list
        nonexist_scores_plus_list = transfer_scores_PNR(
            scores_matrix_one_norm=scores_matrix_one_norm,
            scores_matrix_two_norm=scores_matrix_two_norm,
            train_binary=train_binary,
            PNR=plus_raster_grids,
            interval=float((1.0 - 0.0) / binNum),
            binNum=binNum)
        # 获取multiply的nonexist_scores_list
        nonexist_scores_multiply_list = transfer_scores_PNR(
            scores_matrix_one_norm=scores_matrix_one_norm,
            scores_matrix_two_norm=scores_matrix_two_norm,
            train_binary=train_binary,
            PNR=multiply_raster_grids,
            interval=float((1.0 - 0.0) / binNum),
            binNum=binNum)
        # 获取mlp的nonexist_scores_list
        nonexist_scores_mlp_list = transfer_scores_PNR(
            scores_matrix_one_norm=scores_matrix_one_norm,
            scores_matrix_two_norm=scores_matrix_two_norm,
            train_binary=train_binary,
            PNR=mlp_raster_grids,
            interval=float((1.0 - 0.0) / binNum),
            binNum=binNum)
        # 获取PNR的nonexist_scores_list
        nonexist_scores_PNR_list = transfer_scores_PNR(
            scores_matrix_one_norm=scores_matrix_one_norm,
            scores_matrix_two_norm=scores_matrix_two_norm,
            train_binary=train_binary,
            PNR=PNR_raster_grids,
            interval=float((1.0 - 0.0) / binNum),
            binNum=binNum)


        # 获取阈值
        E_test = np.sum(test_binary.A)
        thresold_plus = get_list_thresold(nonexist_scores_plus_list, L=E_test)
        thresold_multiply = get_list_thresold(nonexist_scores_multiply_list, L=E_test)
        thresold_mlp = get_list_thresold(nonexist_scores_mlp_list, L=E_test)
        thresold_PNR = get_list_thresold(nonexist_scores_PNR_list, L=E_test)


        # 这里的trick, L=1/2 |E_test|!!!!!!!!!!!
        # thresold_plus = int(thresold_plus*0.5)
        # thresold_multiply = int(thresold_multiply * 0.5)
        # thresold_mlp = int(thresold_mlp * 0.5)
        # thresold_PNR = int(thresold_PNR * 0.5)

        # 修改grids
        plus_raster_grids=plus_raster_grids.A
        multiply_raster_grids= multiply_raster_grids.A
        mlp_raster_grids= mlp_raster_grids.A
        PNR_raster_grids= PNR_raster_grids.A
        # np.where(plus_raster_grids > thresold_plus, plus_raster_grids, 0)
        # np.where(multiply_raster_grids > thresold_multiply, multiply_raster_grids, 0)
        # np.where(mlp_raster_grids > thresold_mlp, mlp_raster_grids, 0)
        # np.where(PNR_raster_grids > thresold_PNR, PNR_raster_grids, 0)
        plus_raster_grids[plus_raster_grids <= thresold_plus]=0.0
        multiply_raster_grids[multiply_raster_grids <= thresold_multiply]=0.0
        mlp_raster_grids[mlp_raster_grids <= thresold_mlp]=0.0
        PNR_raster_grids[PNR_raster_grids <= thresold_PNR]=0.0

        plus_raster_grids[plus_raster_grids >= thresold_plus]=1.0
        multiply_raster_grids[multiply_raster_grids >= thresold_multiply]=1.0
        mlp_raster_grids[mlp_raster_grids >= thresold_mlp]=1.0
        PNR_raster_grids[PNR_raster_grids >= thresold_PNR]=1.0



        # 画图
        # colors = ['OrangeRed', 'darkseagreen', 'dodgerblue', 'blueviolet']
        colors=['Red', 'green', 'blue', 'purple']
        result = np.float32(PNR_raster_grids)
        result = cv2.GaussianBlur(result, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
        title = graph_name + '-PNR-' + emb_method_name1 + '-' + emb_method_name2
        plot_contourf_overlap(result=result, title=title, color=colors[0])

        result = np.float32(plus_raster_grids)
        result = cv2.GaussianBlur(result, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
        title = graph_name + '-plus-' + emb_method_name1 + '-' + emb_method_name2
        plot_contourf_overlap(result=result, title=title, color=colors[1])

        result = np.float32(multiply_raster_grids)
        result = cv2.GaussianBlur(result, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
        title = graph_name + '-multiply-' + emb_method_name1 + '-' + emb_method_name2
        plot_contourf_overlap(result=result, title=title, color=colors[2])

        result = np.float32(mlp_raster_grids)
        result = cv2.GaussianBlur(result, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
        title = graph_name + '-mlp-' + emb_method_name1 + '-' + emb_method_name2
        plot_contourf_overlap(result=result, title=title, color=colors[3])






        # # 计算plus的rasterization grids
        # plus_raster_grids = rasterization_grids(binNum=plus_binNum,
        #                                        train_binary=train_binary,
        #                                        scores_matrix_DNN=scores_matrix_plus_norm,
        #                                        scores_matrix_one_norm=scores_matrix_one_norm,
        #                                        scores_matrix_two_norm=scores_matrix_two_norm)
        # # plus_raster_grids = np.log10(plus_raster_grids) # 出现-inf而报错
        # plus_raster_grids = normalize_matrix_full(csr_matrix1=csr_matrix(plus_raster_grids))
        # plus_raster_grids = better_show_grids(csr_matrix1=plus_raster_grids)
        #
        # source = np.float32(plus_raster_grids.A)
        # result = cv2.GaussianBlur(source, (5, 5), 0)
        # title = graph_name + '-' + 'plus' +'-' + emb_method_name1 + '-' + emb_method_name2
        # plot_contourf(result=result, title=title, binNum=10)
        #



        time_end = time.time()
        print("It takes : " + str((time_end-time_start) / 60.0) + "  mins.")
        pass

if __name__ == '__main__':



    # ['cn', 'ja', 'aa', 'ra', 'cosine', 'pearson', 'degreeproduct', 'simrank']
    # ['arope', 'drne', 'graph2gauss', 'prone',
    #  'attentionwalk', 'deepwalk', 'node2vec','splitter', 'prune']
    methods_heuristic=['cn']
    methods_embedding=['node2vec']# 'graph2gauss', 'attentionwalk', 'splitter'
    binNum = 50  # DNN rasterization grids # 改这里


for method_i in range(len(methods_heuristic)):
  for method_j in range(len(methods_embedding)):
    prex='preprocessing_code//'  # 改这里

    # 10000以上、以下的数据集的定义
    graph_group_micro=[]
    graph_group_min=[]
    graph_group_mid =[]
    graph_group_max = []
    # 注释掉的数据集都是heuristic中表现很不好的数据集
    if prex=='preprocessing_code//':
        graph_group_micro =  []# 6个， 'jazz','celegans','pdzbase', 'moreno_blogs','moreno_propro','usair97',karate
        graph_group_min = [
           # 'email-eucore',
            'facebook_combined'] # 5个 # 'petster-friendships-hamster','wiki','petster-hamster','yeastprotein','ca-grqc', 'cora', 'citeseer','usairport', 'openflights'
        graph_group_mid = ['ca-hepph'
                           ]# 6个 # 'ppi','p2p-Gnutella08','p2p-Gnutella09','p2p-Gnutella05', 'p2p-Gnutella04','powergrid','wiki-vote', 'blogcatalog', 'dblp-cite'
        graph_group_max = [
                            'ego-gplus','epinions']# 2个， 'p2p-Gnutella25','ca-astroph', 'google', 'ca-condmat','ego-twitter'
    elif prex=='preprocessing_code2//':
        graph_group_micro = [] # 4个'brazil-airports','NS','PB','europe-airports'
        graph_group_min = [
                            'yeast']# 3个 # 'usa-airports','ppi2','europe-airports','citeseer2_mat','citeseer2',
        graph_group_mid = [] # 1个  #'wikipedia', 'flickr', 'dblp_mat', 'dblp'
        graph_group_max = ['pubmed']# 1个


    # 定义baselines
    emb_method_name1= methods_heuristic[method_i] # 改这里
    emb_method_name2 = methods_embedding[method_j] # 改这里
    ############################### 第一个baseline##########################
    # 第一级：splitter（WWW，2019）、DRNE(KDD，2018)、Arope(KDD，2018)、 graph2gauss(ICLR，2018)
    # 第二级：ProNE(IJCAI，2019，有两个emb可以作为优化空间，我看了原文，应该是用enhanced)、
    #        AttentionWalk(NIPS, 2018)
    #        struc2vec(KDD，2017)、SDNE(KDD，2016)、GraRep(CIKM，2015)、
    #        LINE(WWW，2015)、     deepwalk(KDD，2014)
    # 第三级：Prune(NIPS，2017)、node2vec(KDD，2016）
    ############################### 第二个baseline##########################
    # 第一级：cn, ja, aa, ra, cosine, pearson, degreeproduct（没有其他超参数，分数是定的）
    # 第二级：simrank, (暂时不跑)rootedpagerank（很慢，而且效果很差）
    # （效果很差，暂时不跑）第三级：katz（很慢），graphdistance
    # 第四级：community


    # # micro
    for i in range(len(graph_group_micro)):
        graph_name=graph_group_micro[i]
        auto_overlap(prex=prex, graph_name=graph_name,
                 emb_method_name1=emb_method_name1,
                 emb_method_name2=emb_method_name2,
                 binNum=binNum)
    pass


    # # min
    for i in range(len(graph_group_min)):
        graph_name=graph_group_min[i]
        auto_overlap(prex=prex, graph_name=graph_name,
                 emb_method_name1=emb_method_name1,
                 emb_method_name2=emb_method_name2,
                 binNum=binNum)
    pass
    #
    # # mid
    # for i in range(len(graph_group_mid)):
    #     graph_name=graph_group_mid[i]
    #     auto_multiply(prex=prex, graph_name=graph_name,
    #              emb_method_name1=emb_method_name1,
    #              emb_method_name2=emb_method_name2,
    #              multiply_binNum=multiply_binNum)
    # pass






