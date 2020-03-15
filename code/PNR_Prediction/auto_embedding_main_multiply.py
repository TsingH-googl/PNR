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
    save_DNN_hybrid_scores, save_multiply_raster_scores, save_multiply_hybrid_scores
from utils_DNN import negative_samples, predicted_scores_DNN, rasterization_grids, better_show_grids
from evaluation import evaluators
from fileIO_utils import is_excel_file_exist, multiply_write_to_excel
from utils_plot import plot_contourf, plot_heatmap

all_heuristic_methods = ['cn', 'ja', 'aa', 'ra', 'cosine', 'pearson', 'degreeproduct', 'simrank']
all_embedding_methods = ['arope', 'drne', 'graph2gauss', 'prone', 'attentionwalk', 'deepwalk', 'node2vec','splitter', 'prune']



def auto_multiply(prex=None,
                  graph_name=None,
                  emb_method_name1=None,
                  emb_method_name2=None,
                  multiply_binNum=None):
    print('----------------------------------------------------------')
    print("dataset: " + graph_name + '\n' + "baselines:" + emb_method_name1 + "," + emb_method_name2)



    results_base_dir = 'D:\hybridrec//results//'
    all_file_dir = 'D:\hybridrec\dataset\split_train_test//' + prex
    results_dir = 'D:\hybridrec/results//' + prex
    graph_results_dir = results_dir + graph_name + '//'
    # （facebook_combined的规律：ratio越小则正负样本的预测准确率越高，花的时间也越少）
    # ratio = 1  # 负样本的总数是正样 本的ratio倍  # 改这里


    path_scores_method1 = results_base_dir + prex + graph_name + "//" + graph_name + "_" + emb_method_name1 + "_scores.mat"
    path_scores_method2 = results_base_dir + prex + graph_name + "//" + graph_name + "_" + emb_method_name2 + "_scores.mat"


    # # Initialize the model，改这里
    #
    # # hidden_layer_sizes=(10, 20, 10)：三个隐藏层，分别10、20、10个神经元
    # if model_name == "mlp":
    #     model = MLPClassifier(hidden_layer_sizes=(10, 20), activation='relu', solver='adam', max_iter=200,
    #                           alpha=0.01, batch_size=256, learning_rate='constant', learning_rate_init=0.001,
    #                           shuffle=False, random_state=2020, early_stopping=True,
    #                           validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
    # pass
    #
    # if model_name == "svm":
    #    model = SVC(C=5, random_state=42) # 出问题了
    # pass
    #
    # if model_name == "lr":
    #    model = LogisticRegression(C=5, penalty='l1', tol=1e-6, random_state=42)  # penalty 有l1和l2
    # pass
    #
    # if model_name == "lgbm":
    #    model = LGBMClassifier(num_leaves=31, learning_rate=0.1,
    #                       n_estimators=64, random_state=42, n_jobs=-1)
    # pass
    #
    # if model_name == "xgb":
    #    model = XGBClassifier(max_depth=5, learning_rate=0.1, n_jobs=-1, nthread=-1,
    #                     gamma=0.06, min_child_weight=5,
    #                     subsample=1, colsample_bytree=0.9,
    #                     reg_alpha=0, reg_lambda=0.5,
    #                     random_state=42)
    # pass
    #
    # if model_name == "ld":
    #    model = LinearDiscriminantAnalysis(solver='lsqr')
    # pass
    #
    #
    # if model_name == "rf":
    #    model = RandomForestClassifier(n_estimators=50, max_depth=20,
    #                             min_samples_split=2, min_samples_leaf=5,
    #                             max_features="log2", random_state=12)
    # pass

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

        # del scores_matrix_one, scores_matrix_two
        # gc.collect()


        # # 获取正样本的分数
        # exist_binary = csr_matrix(np.triu(train_binary.A, k=1))  # k=1表示不包括对角线
        # exist_scores_one_list = (np.array(scores_matrix_one_norm[exist_binary > 0], dtype=float))[0]
        # exist_scores_two_list = (np.array(scores_matrix_two_norm[exist_binary > 0], dtype=float))[0]


        # # 构建测试样本（正样本+负样本）
        # X_train_1 = (np.array([exist_scores_one_list, exist_scores_two_list])).T
        # X_train_0 = negative_samples(train_binary=train_binary,
        #                              test_binary=test_binary,
        #                              scores_matrix_one_norm=scores_matrix_one_norm,
        #                              scores_matrix_two_norm=scores_matrix_two_norm,
        #                              ratio=ratio)
        # Y_train_1 = np.random.randint(1, 2, X_train_1.shape[0])
        # Y_train_0 = np.random.randint(0, 1, X_train_0.shape[0])
        # X_train = np.vstack((np.array(X_train_1), np.array(X_train_0)))
        # Y_train = (np.hstack((np.array(Y_train_1), np.array(Y_train_0)))).T

        time_start = time.time()

        # # 模型训练
        # model.fit(X_train, Y_train)
        #
        # # 模型预测
        # preds_0 = model.predict(X_train_0)
        # preds_1 = model.predict(X_train_1)
        # print(np.sum(preds_0))
        # print(np.sum(preds_1))
        # preds_0_proba = model.predict_proba(X_train_0)
        # preds_1_proba = model.predict_proba(X_train_1)


        # 模型预测
        # scores_matrix_DNN = predicted_scores_DNN(model=model,
        #                                          train_binary=train_binary,
        #                                          test_binary=test_binary,
        #                                          scores_matrix_one_norm=scores_matrix_one_norm,
        #                                          scores_matrix_two_norm=scores_matrix_two_norm)
        scores_matrix_multiply = csr_matrix(np.multiply(csr_matrix(scores_matrix_one).A,
                                                        csr_matrix(scores_matrix_two).A))

        # 以下相加的这两种结果是不一样的！！！因为一般来说，分数大的归一化后他的影响力就小了！！
        # scores_matrix_multiply = 0.5*csr_matrix(scores_matrix_one) + 0.5*csr_matrix(scores_matrix_two)
        # scores_matrix_multiply = 0.5 * csr_matrix(scores_matrix_one_norm) + 0.5 * csr_matrix(scores_matrix_two_norm)

        # scores_matrix_multiply = csr_matrix(csr_matrix(scores_matrix_one).A * csr_matrix(scores_matrix_two).A)

        # scores_matrix_multiply = csr_matrix(np.multiply(scores_matrix_one_norm.A,
        #                                                 scores_matrix_two_norm.A)) # 小数乘小数就会变小了！然后就会在normalize_matrix那里报错
        scores_matrix_multiply_norm = normalize_matrix(csr_matrix1=scores_matrix_multiply)
        save_multiply_hybrid_scores(scores_matrix_multiply=scores_matrix_multiply,
                               method1=emb_method_name1, method2=emb_method_name2,
                               graph_results_dir=graph_results_dir,
                               dataset_name=graph_name)


        # 计算multiply的rasterization grids
        multiply_raster_grids = rasterization_grids(binNum=multiply_binNum,
                                               train_binary=train_binary,
                                               scores_matrix_DNN=scores_matrix_multiply_norm,
                                               scores_matrix_one_norm=scores_matrix_one_norm,
                                               scores_matrix_two_norm=scores_matrix_two_norm)
        # multiply_raster_grids = np.log10(multiply_raster_grids) # 出现-inf而报错
        multiply_raster_grids = normalize_matrix_full(csr_matrix1=csr_matrix(multiply_raster_grids))
        multiply_raster_grids = better_show_grids(csr_matrix1=multiply_raster_grids)
        save_multiply_raster_scores(rastser_grids=multiply_raster_grids,
                               method1=emb_method_name1, method2=emb_method_name2,
                               graph_results_dir=graph_results_dir,
                               dataset_name=graph_name,
                               multiply_binNum=multiply_binNum)
        source = np.float32(multiply_raster_grids.A)
        result = cv2.GaussianBlur(source, (5, 5), 0)
        title = graph_name + '-' + 'multiply' +'-' + emb_method_name1 + '-' + emb_method_name2
        plot_contourf(result=result, title=title, binNum=10)



        # 读取PNR grids
        PNR_path=results_base_dir + prex + graph_name + "//" + "PNR1_"+graph_name + "_" + emb_method_name1 + "_"+emb_method_name2 +"_50_count.mat"
        if is_excel_file_exist(PNR_path):
            PNR_dict = (loadmat(PNR_path))
            PNR_matrix = PNR_dict["count"]
            PNR_matrix = better_show_grids(csr_matrix1=PNR_matrix)
            source = np.float32(PNR_matrix.A)
            result = cv2.GaussianBlur(source, (5, 5), 0) #(5, 5)表示高斯矩阵的长与宽都是5，标准差取0
            title = graph_name + '-PNR-' + emb_method_name1 + '-' + emb_method_name2
            plot_contourf(result=result, title=title, binNum=10)


        # 评估multiply hybrid
        exist_binary = csr_matrix(np.triu(train_binary.A, k=1))  # k=1表示不包括对角线
        nonexist_binary = csr_matrix(np.triu(np.ones(exist_binary.shape), k=1) - exist_binary.A)
        nonexist_scores_multiply_list = (np.array(scores_matrix_multiply_norm[nonexist_binary > 0], dtype=float))[0]
        L_full = int(np.sum(test_binary))
        L_array = np.array([int(L_full / 20), int(L_full / 10),
                            int(L_full / 5), int(L_full / 2),
                            L_full])
        AP_multiply, AUC_multiply, Precision_multiply, Recall_multiply, F1score_multiply = \
            evaluators(train_binary=train_binary,
                       test_binary=test_binary,
                       scores_list=nonexist_scores_multiply_list,
                       L_array=L_array)
        # print('AP_DNN:  ' + str(AP_DNN))
        # print('\n')
        # print('AUC_DNN:  ' + str(AUC_DNN))
        # print('\n')
        # print('Precision_DNN:  ' + str(Precision_DNN))
        # print('\n')
        # print('Recall_DNN:  ' + str(Recall_DNN))
        # print('\n')
        # print('F1score_DNN:  ' + str(F1score_DNN))
        # print('\n')

        # 把precision、recall、F1score、AP写入excel文件
        multiply_write_to_excel(dataset_name=graph_name, method1=emb_method_name1, method2=emb_method_name2,
                           precision_multiply=Precision_multiply,
                           recall_multiply=Recall_multiply,
                           F1score_multiply=F1score_multiply,
                           AP_multiply=AP_multiply)


        time_end = time.time()
        print("It takes : " + str((time_end-time_start) / 60.0) + "  mins.")
        pass

if __name__ == '__main__':



    # ['cn', 'ja', 'aa', 'ra', 'cosine', 'pearson', 'degreeproduct', 'simrank']
    # ['arope', 'drne', 'graph2gauss', 'prone',
    #  'attentionwalk', 'deepwalk', 'node2vec','splitter', 'prune']
    methods_heuristic=['cn', 'ja', 'aa', 'cosine', 'pearson']
    methods_embedding=['drne', 'prone', 'deepwalk', 'node2vec']# 'graph2gauss', 'attentionwalk', 'splitter'
    multiply_binNum = 50  # DNN rasterization grids # 改这里


for method_i in range(len(methods_embedding)):
  for method_j in range(method_i+1, len(methods_embedding)):
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
    emb_method_name1= methods_embedding[method_i] # 改这里
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
        auto_multiply(prex=prex, graph_name=graph_name,
                 emb_method_name1=emb_method_name1,
                 emb_method_name2=emb_method_name2,
                 multiply_binNum=multiply_binNum)
    pass


    # # min
    for i in range(len(graph_group_min)):
        graph_name=graph_group_min[i]
        auto_multiply(prex=prex, graph_name=graph_name,
                 emb_method_name1=emb_method_name1,
                 emb_method_name2=emb_method_name2,
                 multiply_binNum=multiply_binNum)
    pass

    # # mid
    # for i in range(len(graph_group_mid)):
    #     graph_name=graph_group_mid[i]
    #     auto_multiply(prex=prex, graph_name=graph_name,
    #              emb_method_name1=emb_method_name1,
    #              emb_method_name2=emb_method_name2,
    #              multiply_binNum=multiply_binNum)
    # pass

    #
    # # max
    # for i in range(len(graph_group_max)):
    #     graph_name=graph_group_max[i]
    # auto_multiply(prex=prex, graph_name=graph_name,
    #               emb_method_name1=emb_method_name1,
    #               emb_method_name2=emb_method_name2,
    #               multiply_binNum=multiply_binNum)
    # pass





