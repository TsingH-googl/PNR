import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import time
import gc
import configparser
np.random.seed(2020)
import numpy as np


from utils import get_trainset_path, get_testset_path, get_test_matrix_binary
from utils import normalize_matrix, get_row_col_index, read_graph, is_heuristic_method, plot_matrix
from utils import get_connp, get_from_zeros_one, save_ndarray_to_mat
from utils import inner_product_scores_arope, inner_product_scores, inner_product_scores_splitter, energy_kl_scores_graph2gauss
from embedding_models import run_emb_method
from heuristic_models import heuristic_scores
from evaluation import transfer_scores_PNR, evaluators
from fileIO_utils import write_to_excel, is_excel_file_exist, get_excel_save_path




def auto_PNR(prex=None, graph_name=None,
             emb_method_name1=None, emb_method_name2=None):

    print('----------------------------------------------------------')
    time_start = time.time()
    # 初始化训练集和测试集的路径
    # prex = 'preprocessing_code2//'  # 改这里
    all_file_dir = 'D:\hybridrec\dataset\split_train_test//' + prex



    binNum = 50 # 改这里



    emb_method_name1 = emb_method_name1.lower() # 改这里
    emb_method_name2 = emb_method_name2.lower() # 改这里
    print("dataset: " + graph_name + '\n' + "baselines:" + emb_method_name1 + "," + emb_method_name2)
    # config_path_method1 = 'conf/' + emb_method_name1 + '.properties'
    # config_method1 = configparser.ConfigParser()
    # config_method1.read(config_path_method1)
    # conf_method1 = dict(config_method1.items("hyperparameters"))
    # config_path_method2 = 'conf/' + emb_method_name2 + '.properties'
    # config_method2 = configparser.ConfigParser()
    # config_method2.read(config_path_method2)
    # conf_method2 = dict(config_method2.items("hyperparameters"))



    # 初始化embedding和scores的路径
    results_dir = 'D:\hybridrec/results//' + prex
    graph_results_dir = results_dir + graph_name + '//'




    # 计算emb method 1
    if not ((emb_method_name1 == 'arope') or (emb_method_name1 == 'graph2gauss') or (is_heuristic_method(emb_method_name1)==True)):
        graph_train_path=get_trainset_path(base_dir=all_file_dir,
                                           graph_name=graph_name,
                                           connected_pattern=get_connp(emb_method_name1),
                                           from_zeros_one=get_from_zeros_one(emb_method_name1))
        graph_results_path = graph_results_dir + graph_name+'_'+emb_method_name1+'.emb'
        run_emb_method(input=graph_train_path,
                       output=graph_results_path,
                       emb_method_name=emb_method_name1)



    # 计算emb method 2
    if not ((emb_method_name2 == 'arope') or (emb_method_name2 == 'graph2gauss') or (is_heuristic_method(emb_method_name2)==True)):
        graph_train_path=get_trainset_path(base_dir=all_file_dir,
                                           graph_name=graph_name,
                                           connected_pattern=get_connp(emb_method_name2),
                                           from_zeros_one=get_from_zeros_one(emb_method_name2))
        graph_results_path = graph_results_dir + graph_name+'_'+emb_method_name2+'.emb'
        run_emb_method(input=graph_train_path,
                       output=graph_results_path,
                       emb_method_name=emb_method_name2)




    # 计算scores1
    embedding_size_method1 = 0
    if emb_method_name1 == 'splitter':
        scores_matrix_one = inner_product_scores_splitter(graph_results_dir=graph_results_dir,
                                                          dataset_name=graph_name, emb_method_name=emb_method_name1,
                                                          col_start=0, col_end=embedding_size_method1 + 1, skiprows=1,
                                                          delimiter=',')
    elif (emb_method_name1 == 'attentionwalk') or (emb_method_name1 == 'grarep'):
        scores_matrix_one = inner_product_scores(graph_results_dir=graph_results_dir,
                                                 dataset_name=graph_name, emb_method_name=emb_method_name1,
                                                 col_start=0, col_end=embedding_size_method1 + 1, skiprows=1,
                                                 delimiter=',')
    elif (emb_method_name1 == 'drne') or (emb_method_name1 == 'prune'):
        scores_matrix_one = inner_product_scores(graph_results_dir=graph_results_dir,
                                                 dataset_name=graph_name, emb_method_name=emb_method_name1,
                                                 col_start=0, col_end=embedding_size_method1, skiprows=0,
                                                 delimiter=' ')  # embedding_size_method有一些是要+1有一些不需要的
    elif (emb_method_name1 == 'arope'):
        scores_matrix_one = inner_product_scores_arope(all_file_dir=all_file_dir,
                                                       graph_name=graph_name,
                                                       graph_results_dir=graph_results_dir)
    elif (emb_method_name1 == 'graph2gauss'):
        scores_matrix_one = energy_kl_scores_graph2gauss(all_file_dir=all_file_dir,
                                                         graph_name=graph_name,
                                                         graph_results_dir=graph_results_dir)
    elif is_heuristic_method(emb_method_name1):
        scores_matrix_one = heuristic_scores(all_file_dir=all_file_dir,
                                             graph_name=graph_name,
                                             graph_results_dir=graph_results_dir,
                                             heuristic_method=emb_method_name1)
    else:
        scores_matrix_one = inner_product_scores(graph_results_dir=graph_results_dir,
                                                 dataset_name=graph_name, emb_method_name=emb_method_name1,
                                                 col_start=0, col_end=embedding_size_method1 + 1, skiprows=1,
                                                 delimiter=' ')

    # 计算scores2
    embedding_size_method2 = 0
    if emb_method_name2 == 'splitter':
        scores_matrix_two = inner_product_scores_splitter(graph_results_dir=graph_results_dir,
                                                          dataset_name=graph_name, emb_method_name=emb_method_name2,
                                                          col_start=0, col_end=embedding_size_method2 + 1, skiprows=1,
                                                          delimiter=',')
    elif (emb_method_name2 == 'attentionwalk') or (emb_method_name2 == 'grarep'):
        scores_matrix_two = inner_product_scores(graph_results_dir=graph_results_dir,
                                                 dataset_name=graph_name, emb_method_name=emb_method_name2,
                                                 col_start=0, col_end=embedding_size_method2 + 1, skiprows=1,
                                                 delimiter=',')
    elif (emb_method_name2 == 'drne') or (emb_method_name2 == 'prune'):
        scores_matrix_two = inner_product_scores(graph_results_dir=graph_results_dir,
                                                 dataset_name=graph_name, emb_method_name=emb_method_name2,
                                                 col_start=0, col_end=embedding_size_method2, skiprows=0, delimiter=' ')
    elif (emb_method_name2 == 'arope'):
        scores_matrix_two = inner_product_scores_arope(all_file_dir=all_file_dir,
                                                       graph_name=graph_name,
                                                       graph_results_dir=graph_results_dir)
    elif (emb_method_name2 == 'graph2gauss'):
        scores_matrix_two = energy_kl_scores_graph2gauss(all_file_dir=all_file_dir,
                                                         graph_name=graph_name,
                                                         graph_results_dir=graph_results_dir)
    elif is_heuristic_method(emb_method_name2):
        scores_matrix_two = heuristic_scores(all_file_dir=all_file_dir,
                                             graph_name=graph_name,
                                             graph_results_dir=graph_results_dir,
                                             heuristic_method=emb_method_name2)
    else:
        scores_matrix_two = inner_product_scores(graph_results_dir=graph_results_dir,
                                                 dataset_name=graph_name, emb_method_name=emb_method_name2,
                                                 col_start=0, col_end=embedding_size_method2 + 1, skiprows=1,
                                                 delimiter=' ')



    # scores取上三角（注意:1、前面需要保证所有的分数在右上角或占满整个矩阵。2、前面有些是右上角，有些是占满整个矩阵）
    # scores_matrix_one_full = scores_matrix_one.A
    # scores_matrix_two_full = scores_matrix_two.A
    # plot_matrix(matrix = scores_matrix_one_full)
    # plot_matrix(matrix = scores_matrix_two_full)
    scores_matrix_one = sp.csr_matrix(np.triu(scores_matrix_one.A, k=1)) # k=1表示不包括对角线
    scores_matrix_two = sp.csr_matrix(np.triu(scores_matrix_two.A, k=1))




    # 读入train的binary数据
    graph_train_path = get_trainset_path(base_dir=all_file_dir,
                                         graph_name=graph_name,
                                         connected_pattern='undirected',
                                         from_zeros_one='0')
    G = read_graph(weighted=0, input=graph_train_path, directed=0)
    train_binary = sp.csr_matrix(nx.convert_matrix.to_scipy_sparse_matrix(G))
    train_binary = sp.csr_matrix(np.triu(train_binary.A, k=1))
    # train_binary_full = train_binary.A
    # 或 train_binary = sp.csr_matrix(np.array(nx.to_numpy_matrix(G)))



    # 构建exist和nonexist的binary
    exist_binary = sp.csr_matrix(np.triu(train_binary.A, k=1)) # k=1表示不包括对角线
    nonexist_binary = sp.csr_matrix(np.triu(np.ones(exist_binary.shape), k=1) - exist_binary.A)

    # 分数归一化到[0.0, 1.0]
    scores_matrix_one_norm = normalize_matrix(csr_matrix1 = scores_matrix_one)
    scores_matrix_two_norm = normalize_matrix(csr_matrix1 = scores_matrix_two)
    # plot_matrix(scores_matrix_one_norm.A)
    # plot_matrix(scores_matrix_two_norm.A)

    del scores_matrix_one, scores_matrix_two
    gc.collect()



    # 划分bin
    val_max = 1.0
    val_min = 0.0
    # bin_array = sorted(divide_bin(val_max = val_max, val_min = val_min, binNum = binNum))
    interval = float((val_max - val_min) / binNum)

    # 获取exist_binary和nonexist_binary的分数
    exist_scores_one_list =    (np.array(scores_matrix_one_norm[exist_binary    > 0], dtype=float))[0]
    nonexist_scores_one_list = (np.array(scores_matrix_one_norm[nonexist_binary > 0], dtype=float))[0]
    exist_scores_two_list =    (np.array(scores_matrix_two_norm[exist_binary    > 0], dtype=float))[0]
    nonexist_scores_two_list = (np.array(scores_matrix_two_norm[nonexist_binary > 0], dtype=float))[0]
    # # 变为稀疏矩阵
    # exist_scores_one_list_csr = sp.csr_matrix(exist_scores_one_list)
    # nonexist_scores_one_list_csr = sp.csr_matrix(nonexist_scores_one_list)
    # exist_scores_two_list_csr = sp.csr_matrix(exist_scores_two_list)
    # nonexist_scores_two_list_csr = sp.csr_matrix(nonexist_scores_two_list)

    # temp = scores_matrix_one_norm[exist_binary > 0][0] # 我怕在把分数变为list的时候出问题



    # 初始化两个大小为binNum* bnNum的二维栅格
    exist_raster_grids = np.zeros((binNum, binNum))
    nonexist_raster_grids = np.zeros((binNum, binNum))




    # 计算落在exist_raster_grids栅格的existing links的数量
    exist_links_num = len(exist_scores_one_list)
    exist_row_col_zero_num = 0 # 那些两个矩阵的分数都是0的不作统计
    for i in range(exist_links_num):
        # row_index和col_index的范围从0-->binNum-1
        if (exist_scores_one_list[i] == 0.0) & (exist_scores_two_list[i] == 0.0):
            exist_row_col_zero_num = exist_row_col_zero_num + 1
            continue
        row_index = int(get_row_col_index(score = exist_scores_one_list[i], interval = interval, binNum = binNum))
        col_index = int(get_row_col_index(score = exist_scores_two_list[i], interval = interval, binNum = binNum))
        exist_raster_grids[row_index, col_index] = exist_raster_grids[row_index, col_index] + 1

    print("exist_row_col_zero_num:" + str(exist_row_col_zero_num))
    print('sum  exist_raster_grids:' + str(np.sum(exist_raster_grids)))




    # 计算落在nonexist_raster_grids栅格的nonexisting links的数量
    nonexist_links_num = len(nonexist_scores_one_list)
    nonexist_row_col_zero_num = 0 # 那些两个矩阵的分数都是0的不作统计
    for i in range(nonexist_links_num):
        # row_index和col_index的范围从0-->binNum-1
        if (nonexist_scores_one_list[i] <= 0.0) & (nonexist_scores_two_list[i] <= 0.0):
            nonexist_row_col_zero_num = nonexist_row_col_zero_num + 1
            continue
        row_index = int(get_row_col_index(score = nonexist_scores_one_list[i], interval = interval, binNum = binNum))
        col_index = int(get_row_col_index(score = nonexist_scores_two_list[i], interval = interval, binNum = binNum))

        nonexist_raster_grids[row_index, col_index] = nonexist_raster_grids[row_index, col_index] + 1

    print("nonexist_row_col_zero_num:" + str(nonexist_row_col_zero_num))
    print('sum  nonexist_raster_grids:' + str(np.sum(nonexist_raster_grids)))





    # 计算PNR分数
    N = train_binary.shape[0]
    print("Graph size：" + str(N) + '\n')
    L_T = np.sum(train_binary.A)
    O = N * (N - 1) / 2
    coefficient = (O - L_T) / L_T
    PNR1 = coefficient * (exist_raster_grids / (nonexist_raster_grids + 1)) # 分母加1避免出现inf或nan，不影响evaluation但是可能好看
    PNR2 = (exist_raster_grids / nonexist_raster_grids )  # inf和nan置为0
    PNR2[np.isnan(PNR2)] = 0
    PNR2[np.isinf(PNR2)] = 0
    PNR2 = coefficient * PNR2


    # 画图（注意：图的横纵坐标是从左上角开始的而不是想象中的左上角）
    # sns.heatmap(PNR1, cmap='Reds')
    # plt.savefig(graph_results_dir + emb_method_name1 +'_'+ emb_method_name2 + '_' +'bin_' + str(binNum) + "_PNR1.jpg")
    # plt.show()
    # sns.heatmap(PNR2, cmap='Reds')
    # plt.savefig(graph_results_dir + emb_method_name1 +'_'+ emb_method_name2 + '_'+ 'bin_' + str(binNum) + "_PNR2.jpg")
    # plt.show()
    # plt.matshow(PNR1) # 好丑
    # plt.show()


    # 保存（exist_raster_grids、nonexist_raster_grids、PNR1、PNR2）
    save_ndarray_to_mat(exist_raster_grids,     'exist_raster_grids', graph_results_dir, graph_name, emb_method_name1,
                        emb_method_name2, binNum)
    save_ndarray_to_mat(nonexist_raster_grids, 'nonexist_raster_grids', graph_results_dir, graph_name, emb_method_name1,
                        emb_method_name2, binNum)
    save_ndarray_to_mat(PNR1, 'PNR1', graph_results_dir, graph_name, emb_method_name1,
                        emb_method_name2, binNum)
    save_ndarray_to_mat(PNR2, 'PNR2', graph_results_dir, graph_name, emb_method_name1,
                        emb_method_name2, binNum)





    # PNR调整分数(只调整non-existing link的部分)
    nonexist_scores_PNR_list = transfer_scores_PNR(
                                            scores_matrix_one_norm=scores_matrix_one_norm,
                                            scores_matrix_two_norm=scores_matrix_two_norm,
                                            train_binary=train_binary,
                                            PNR=PNR2,
                                            interval=interval,
                                            binNum=binNum)




    # weighted hybird方法的分数，0.5均权直接相加
    scores_matrix_hybrid_norm = 0.5 * scores_matrix_one_norm + 0.5 * scores_matrix_two_norm
    nonexist_scores_hybrid_list = (np.array(scores_matrix_hybrid_norm[nonexist_binary > 0], dtype=float))[0]



    # 评估evaluation
    graph_test_path=get_testset_path(base_dir=all_file_dir, graph_name=graph_name)
    test_binary=get_test_matrix_binary(graph_test_path=graph_test_path, N=N)
    L_full = int(np.sum(test_binary))
    L_array = np.array([int(L_full/20),int(L_full/10),
                        int(L_full/5), int(L_full/2),
                        L_full])



    del scores_matrix_one_norm, scores_matrix_two_norm, exist_scores_one_list, exist_scores_two_list,scores_matrix_hybrid_norm
    gc.collect()


    AP_PNR, AUC_PNR, Precision_PNR, Recall_PNR, F1score_PNR=\
        evaluators(train_binary=train_binary,
                   test_binary=test_binary,
                   scores_list=nonexist_scores_PNR_list,
                   L_array=L_array)
    AP_method1, AUC_method1, Precision_method1, Recall_method1, F1score_method1=\
        evaluators(train_binary=train_binary,
                   test_binary=test_binary,
                   scores_list=nonexist_scores_one_list,
                   L_array=L_array)
    AP_method2, AUC_method2, Precision_method2, Recall_method2, F1score_method2=\
        evaluators(train_binary=train_binary,
                   test_binary=test_binary,
                   scores_list=nonexist_scores_two_list,
                   L_array=L_array)
    AP_weighted, AUC_weighted, Precision_weighted, Recall_weighted, F1score_weighted=\
        evaluators(train_binary=train_binary,
                   test_binary=test_binary,
                   scores_list=nonexist_scores_hybrid_list,
                   L_array=L_array)

    print('AP_PNR:  ' +str(AP_PNR))
    print('AP_method1:  ' +str(AP_method1))
    print('AP_method2:  ' +str(AP_method2))
    print('AP_weighted:  ' + str(AP_weighted))
    print('\n')
    print('AUC_PNR:  ' +str(AUC_PNR))
    print('AUC_method1:  ' +str(AUC_method1))
    print('AUC_method2:  ' +str(AUC_method2))
    print('AUC_weighted:  ' + str(AUC_weighted))
    print('\n')
    print('Precision_PNR:  ' +str(Precision_PNR))
    print('Precision_method1:  ' +str(Precision_method1))
    print('Precision_method2:  ' +str(Precision_method2))
    print('Precision_weighted:  ' + str(Precision_weighted))
    print('\n')
    print('Recall_PNR:  ' +str(Recall_PNR))
    print('Recall_method1:  ' +str(Recall_method1))
    print('Recall_method2:  ' +str(Recall_method2))
    print('Recall_weighted:  ' + str(Recall_weighted))
    print('\n')
    print('F1score_PNR:  ' +str(F1score_PNR))
    print('F1score_method1:  ' +str(F1score_method1))
    print('F1score_method2:  ' +str(F1score_method2))
    print('F1score_weighted:  ' + str(F1score_weighted))
    print('\n')

    write_to_excel(graph_name, emb_method_name1, emb_method_name2,
                   Precision_PNR, Precision_method1, Precision_method2, Precision_weighted,
                   Recall_PNR, Recall_method1, Recall_method2, Recall_weighted,
                   F1score_PNR, F1score_method1, F1score_method2, F1score_weighted,
                   AP_PNR, AP_method1, AP_method2, AP_weighted,
                   AUC_PNR, AUC_method1, AUC_method2, AUC_weighted)






    time_end = time.time()
    print("time span:  " + str((time_end-time_start)/60.00) + "  mins")
    # facebook_combined：bin=5, 1.5分钟
    # facebook_combined：cn和pearson\aa和cn花了3.5分钟
    # facebook_combined：graphdistance和cn花了11分钟
    # facebook_combined: graphdistance和cn的PNE矩阵为全0
    # facebooke_combined: attentionwalk和prone花了7.5分钟
    # facebooke_combined: 有rootedpagerank的效果都很差;
    # arope比PNR好一点，SDNE和PRUE很差很差；drne和graph2gauss也是极差的但是PNR融合后表现极好；


    # blogcatalog:aa和ja花了3小时
    # （path based--katz和graphdistance都十分慢，neighbor based和rank based很快）

    # google 15000 nodes: 2.5小时
    print('--------------------------------------------------------------------------------')
    pass



if  __name__ =='__main__':
    # ['cn', 'ja', 'aa', 'ra', 'cosine', 'pearson', 'degreeproduct', 'simrank']
    # ['arope', 'drne', 'graph2gauss', 'prone',
    #  'attentionwalk', 'deepwalk', 'node2vec','splitter', 'prune']
    methods=['cn', 'ja', 'aa', 'cosine', 'pearson']# 'graph2gauss', 'attentionwalk', 'splitter'



for method_i in range(len(methods)-1):
  for method_j in range(method_i+1, len(methods)):
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
            'facebook_combined'] # 5个 #'wiki', 'petster-friendships-hamster','petster-hamster','yeastprotein','ca-grqc', 'cora', 'citeseer','usairport', 'openflights'
        graph_group_mid = ['ca-hepph'
                           ]# 6个 # 'ppi','p2p-Gnutella08','p2p-Gnutella09','p2p-Gnutella05', 'p2p-Gnutella04','powergrid','wiki-vote', 'blogcatalog', 'dblp-cite'
        graph_group_max = [
                            'ego-gplus','epinions']# 2个， 'p2p-Gnutella25','ca-astroph', 'google', 'ca-condmat','ego-twitter'
    elif prex=='preprocessing_code2//':
        graph_group_micro = [] # 4个'europe-airports''brazil-airports','NS','PB',
        graph_group_min = [
                            'yeast']# 3个 # 'usa-airports',,'ppi2','europe-airports','citeseer2_mat','citeseer2',
        graph_group_mid = [] # 1个  #'wikipedia', 'flickr', 'dblp_mat', 'dblp'
        graph_group_max = ['pubmed']# 1个


    # 定义baselines
    emb_method_name1= methods[method_i] # 改这里
    emb_method_name2 = methods[method_j] # 改这里
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

    #
    # # micro
    for i in range(len(graph_group_micro)):
        graph_name=graph_group_micro[i]
        if is_excel_file_exist(get_excel_save_path(dataset_name=graph_name, method1=emb_method_name1, method2=emb_method_name2))or \
                is_excel_file_exist(get_excel_save_path(dataset_name=graph_name, method1=emb_method_name2, method2=emb_method_name1)):# 有些xls文件的命名可能是两个method调换了
            print(graph_name+'-'+emb_method_name1+'-'+emb_method_name2+": existed...")
            continue
        auto_PNR(prex=prex, graph_name=graph_name,
                 emb_method_name1=emb_method_name1,
                 emb_method_name2=emb_method_name2)
    pass
    #
    #
    # # min
    for i in range(len(graph_group_min)):
        graph_name=graph_group_min[i]
        if is_excel_file_exist(get_excel_save_path(dataset_name=graph_name, method1=emb_method_name1, method2=emb_method_name2))or \
                is_excel_file_exist(get_excel_save_path(dataset_name=graph_name, method1=emb_method_name2, method2=emb_method_name1)):# 有些xls文件的命名可能是两个method调换了
            print(graph_name+'-'+emb_method_name1+'-'+emb_method_name2+": existed...")
            continue
        auto_PNR(prex=prex, graph_name=graph_name,
                 emb_method_name1=emb_method_name1,
                 emb_method_name2=emb_method_name2)
    pass

    # # # mid
    # for i in range(len(graph_group_mid)):
    #     graph_name=graph_group_mid[i]
    #     if is_excel_file_exist(get_excel_save_path(dataset_name=graph_name, method1=emb_method_name1, method2=emb_method_name2))or \
    #             is_excel_file_exist(get_excel_save_path(dataset_name=graph_name, method1=emb_method_name2, method2=emb_method_name1)):# 有些xls文件的命名可能是两个method调换了
    #         print(graph_name+'-'+emb_method_name1+'-'+emb_method_name2+": existed...")
    #         continue
    #     auto_PNR(prex=prex, graph_name=graph_name,
    #              emb_method_name1=emb_method_name1,
    #              emb_method_name2=emb_method_name2)
    # pass


    # # max
    # for i in range(len(graph_group_max)):
    #     graph_name=graph_group_max[i]
    #     if is_excel_file_exist(get_excel_save_path(dataset_name=graph_name, method1=emb_method_name1, method2=emb_method_name2))or \
    #             is_excel_file_exist(get_excel_save_path(dataset_name=graph_name, method1=emb_method_name2, method2=emb_method_name1)):# 有些xls文件的命名可能是两个method调换了
    #         print(graph_name+'-'+emb_method_name1+'-'+emb_method_name2+": existed...")
    #         continue
    #     auto_PNR(prex=prex, graph_name=graph_name,
    #              emb_method_name1=emb_method_name1,
    #              emb_method_name2=emb_method_name2)
    # pass


