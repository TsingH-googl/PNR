import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import time
import configparser
np.random.seed(2020)

from utils import get_trainset_path, get_testset_path, get_test_matrix_binary
from utils import normalize_matrix, get_row_col_index, read_graph, is_heuristic_method, plot_matrix
from utils import get_connp, get_from_zeros_one, save_ndarray_to_mat
from utils import inner_product_scores_arope, inner_product_scores, inner_product_scores_splitter, energy_kl_scores_graph2gauss
from embedding_models import run_emb_method
from heuristic_models import heuristic_scores
from evaluation import transfer_scores_PNR, evaluators
from fileIO_utils import write_to_excel
import gc









if __name__=='__main__':
    '-----------------------------------------------Done----------------------------------------------------'
    '1、我都用了csr_matrix取存储矩阵，目的是减少内存消耗'
    '2、已经检验过XX_score_XX_list等四个的一一对应正确性和数字的正确性，没问题的'
    '3、deepwalk虽然设置了seed=0但是每次的结果还是不同的'
    '4、对于一些basline例如prone、grarep等，节点数目木太少如karate，则emb的dimensions是不可以太大的'
    '5、splitter很慢！其他都挺快的'
    '6、grarep、SDNE、struc2vec、line没有用到官方代码：他们的结果好奇怪噢~~~低分值PNR更高'
    '7、drne：结果好奇怪，都是低分值的被预测了啊'
    '8、struc2vec有时会因为同时读取../temp_struc2vec中的文件，发生冲突而保存'
    '9、graph2gauss的结果好奇怪！！existing links的分数多数很高、nonexisting links的分数多数很低'
    '10、euroroad的aa、ja方法算出来的大于0的分数的位置是一样的！而且，最巧的是，存在边的分数恰好全都为0，导致PNR1/PNR2都为0'
    '11、注意，相似性分数也可能为负数啊！！！注意：要注意数据类型的指定dtype=float/int'
    '12、PNR的分布得太散，说明有些pn(s)也被置为0了，所以变差了'
    '13、micro、macro是在多类别分类中才有的，这里不适用'
    '14、MRR：sum(1/rank(i))_u / |U|:也就是推荐准确的rank的倒数相加的和再除以所有用户'
    '15、大规模的数据集，要单独跑，不然内存不够吃，实在不行可以考虑不计算AUC'
    '16、cn、aa的二维PNR都是斜直线的~~~~为啥子？'
    '17、cn-ra、cn-aa中我们没ra、aa好除了yeast/usairport\pester\p2p系列的: 因为PNR太零散不充满（特别的是，p2p系列PNR图分散）' \
    '18、pester系列、yeast、p2p、usairport一直很好啊;rootedpagerank和simrank都很慢, rootedpagerank比simrank慢；rootpageran很多都表现很差'
    '''
    有些xls文件的命名可能是两个baseline methods调换了的；不要调换auto_main_rasterazation.py注释中baseline的顺序，因为写xls结果不要乱，不过调换了也没事啦；
    katz在多数数据集下PNR表现得都是极差的！！！
    self-boost：cn-cn、ja-ja这些自己对自己进行boost的有很好的效果啊！！！！self-boost的PNR是一个对角线且颜色渐增。
    auto-XX.py和main_XX.py是一样的，auto-XX.py是批量处理而已；self-boost也就是一维的PNR了；
    
    20.1、对于heuristic models：
        最好：ca-hepph、 ppi（ppi2不行）、p2p-Gnutella04、05、08、09（特别是04）、dblp（dblp-cite不行）、facebook_combined、
             pester-friendship-hamster、yeast、email-eucore、wiki（wiki-vote不行）是很好的数据集；
        次好：petster-hamster、citeseer2（citeseer2_mat和citeceer不好）、usa-airports（usairports不好）
             pdzbase、celegans（pdzbase和celegans把binNum缩小一点试试，估计效果会更好：改成了25实际是更差了！！！）
        
        auto_XX.py中注释掉的数据集都是heuristic中表现很不好的数据集
        (不好的数据集基本都是因为：PNR的分布就几个点，然而，也有例外)
    20.2、对于embedding models（在第20.1点的heuristic进一步筛选）：
        最好：facebook_combined、email-eucore、petster-hamster、petster-friendships-hamster、
             yeast
         
        次好：usa-airports、wiki、ca-hepph、europe-airports、moreno_propro、celegans
        
        多数是arope为baselines就不好！！！！
        
        
    21、只跑pubmed（19700）的pearson、degreeproduct消耗了32Gb内存的98%。对于auto_XX.py中的XX_max网络，只跑cn、ja、aa、ra、cosine。
        pearson、degreeproduct和simrank分数满，比较慢。
    22、'graph2gauss'在以下数据集出问题：p2p-Gnutella08
        'attentionwalk'在一下数据集出问题：ppi
    
    23、Splitter非常耗CPU，到100%，且非常慢，p2p08数据集10000结点花了11小时
    24、'karate'在prone、arope等上会出现embedding size需要变小到16才不出错

    
    '''

    '''
    每个数据集的优势劣势：
    1、ca-hepPh（11204）
    heuristic_embedding: simrank、ra、drne
    heuristic: ra
    embedding: drne
    
    2、facebook_combined（4039）
    heuristic_embedding: simrank、ra
    heuristic: ra
    embedding: arope、drne
    
    3、Yeast（2357）
    heuristic_embedding: ra
    heuristic: simrank
    embedding: 无
    
    4、petster-friendships-hamster（1858）
    heuristic_embedding: simrank、ra、deepwalk、drne、attentionwalk
    heuristic: ra
    embedding: attentionwalk、arope
    
    5、email-eucore（986）
    heuristic_embedding: attentionwalk、ra、ja、pearson、prone
    heuristic: aa、simrank
    embedding: arope、attentionwalk
    
    6、wiki（2357）
    heuristic_embedding: ra、aa、prone
    heuristic: aa、cn
    embedding: arope、drne
    
    7、usa-airports（1186）
    heuristic_embedding: aa、ra、attentionwalk、prone
    heuristic: aa、ra、cn
    embedding: arope
    
    
    后续可能的数据集：
    enron（36000）、    ego-twitter（23000）、
    ego-gplus（23000）、epinions（26588）、  pubmed（19000）、
    google（15763）、   digg_reply（30360）、twitter_combined（81306）、
    youtube（1130000）、facebook-wosn-wall（45813）
    已经证实是有可能用的数据集：最优：pubmed（19000）、epinions（26588）;次优：ego-gplus（23000）、。。。
    已经证实是有可能用的数据集：ego-twitter（23000）、google（15763）。。。
    (!!!!ego-gplus由于处处花费大内存，因此先不用它)
    '''











    '---------------------------------------------To be done------------------------------------------------'
    '1、(ok)归一化：是两个scores矩阵各自归一化还是两个scores矩阵统一归一化？？？答案：当然是各自，不然有些分数会偏向一边的！！'
    '2、(ok)若最终计算的PNR很多0，我觉得可以在分母也就是n_n = n_n + 1使得inf和nan都去除了，不影响evaluation但是可能好看~~'
    '3  similar to item rec, predict a N list for each user u'
    '4、（ok）确定simrank、rootpagerank和katz是否在数据集划分确定的前提下scores是固定的（反正neighbor-based是固定的啦）:rank-based和katz也是固定的！！'
    '5、(ok)katz因为太慢、效果太差而没跑~~'
    '''
    6、（OK，老师说这是多余的）检测PNR grid的数目的比例，来决定是否用PNR进行调整，如果够，则用，否则则在validation数据集的指导下选最好的一个来进行prediction。
    7、（OK）计算self-boost看看结果：不用算了
    8、(OK) 优化保存的scores和list为sparse：不能啦，sklearn的ap计算函数只能接受非sparse得输入。我这里通过del和gc.collect()收集局部变量而提高空间效率。
    9、不同的train/test划分对实验结果也有影响，可以作为最后的一个补充实验。
    10、是否满足用PNR进行融合的条件：计算一维的PNR，若是递增的则不能融合；否则，可以融合。
    11、对于embedding的多个参数难以调优的问题：参考原文的参数，然后进行调优一下，如果实在是没有比传统方法好，那是因为embedding主要用于节点分类和其他的。
    12、测试同一ratio下heuristic结果是否会有差异。
    13、如果baseline不够，补充graphdistance和community以及../unused/SEAL/MATLAB文件夹中的ACT/HDI/HPI/LHN/PA试试。
    14、embedding得方法的分数一般不用缓存的，因为每次emb得数值都是不一样的，跟heuristic不同。
    15、对于小网络，auto_XX.py中的micro数据集，用binNum小一点试试：加了，好像没改善。
    16、prune有些数据集出问题或报错：目前未搞懂。
    '''


    time_start = time.time()
    # 初始化训练集和测试集的路径
    prex = 'preprocessing_code2//'  # 改这里
    all_file_dir = 'D:\hybridrec\dataset\split_train_test//' + prex



    # #######################preprocessing code#########################
    # ###10000以下#######（17个）
    # % -------karate: 34
    # % -------jazz: 198
    # % -------celegans: 297
    # % -------pdzbase: 212
    # % -------moreno_blogs: 1224
    # % -------moreno_propro: 1846
    # % -------usair97: 332
    # email-eucore(986), yeastprotein(1458),usairport（1574），
    # petster-friendships-hamster(1858), petster-hamster(2000), cora(2485), citeseer(2128)，
    # wiki(2357), openflights(3397),ppi(3480), facebook_combined(4039),
    # ca-grqc(4158), powergrid(4941),
    # p2p-Gnutella08(6299),
    # wiki-vote(7066, 在sklearn metrics中的AP计算中不够内存),
    # p2p-Gnutella09(8104),
    # p2p-Gnutella05(8842);
    # ####10000以上#######（11个）
    # blogcatalog(10312)，p2p-Gnutella04(10876), ca-hepph(11204）,dblp-cite（12495）,google(15763),
    # ca-astroph(17903, 在sklearn的roc_auc_score内存不够),
    # p2p-Gnutella25 (22687, 在sklearn的roc_auc_score内存不够),
    # ca-condmat(23133, 在sklearn的roc_auc_score内存不够),
    # ego-gplus(23628),
    # ego-twitter(23370, 太大不能normalized),
    # epinions(26588， nonexist_binary的第一次计算中不够内存)
    #
    # #######################preprocessing code2########################
    # % -------brazil - airports: 131
    # % -------NS: 1461
    # % -------PB: 1222
    # % -------europe - airports: 399
    # ###10000以下#######（8个）
    # europe-airports(399),usa-airports,1186, （citeseer2,2110, citeseer2_mat(2120),暂时用上面的citeseer）
    # yeast(2375),ppi2(3852),wikipedia(4777),flickr(7575);
    # ####10000以上#######（3个）
    # dblp(12231), dblp_mat(13326),pubmed(19717)
    graph_name = 'europe-airports' # 改这里
    # graph_train_name = 'train_' + graph_name + '_undirected_0_giantCom.edgelist' # 也可以从0开始
    # graph_test_name  = 'test_'  + graph_name + '_undirected_1_giantCom.edgelist'
    # graph_train_path = all_file_dir + graph_name + '//' + graph_train_name
    # graph_test_path  = all_file_dir + graph_name + '//' + graph_test_name



    binNum = 50 # 改这里



    ############################### 第一个baseline##########################
    # 第一级：splitter（WWW，2019）、DRNE(KDD，2018)、Arope(KDD，2018)、 graph2gauss(ICLR，2018)
    # 第二级：ProNE(IJCAI，2019，有两个emb可以作为优化空间)、AttentionWalk(NIPS, 2018)
    #        struc2vec(KDD，2017)、SDNE(KDD，2016)、GraRep(CIKM，2015)、
    #        LINE(WWW，2015)、     deepwalk(KDD，2014)
    # 第三级：Prune(NIPS，2017)、node2vec(KDD，2016）
    ############################### 第二个baseline##########################
    # 第一级：cn, ja, aa, ra, cosine, pearson, as, degreeproduct（没有其他超参数，分数是定的）
    # 第二级：simrank, rootedpagerank
    # 第三级：graphdistance, katz
    # 第四级：community
    emb_method_name1 = 'arope'.lower() # 改这里
    emb_method_name2 = 'drne'.lower() # 改这里
    print("dataset: " + graph_name + '\n' + "baselines:" + emb_method_name1 + "," + emb_method_name2)
    config_path_method1 = 'conf/' + emb_method_name1 + '.properties'
    config_method1 = configparser.ConfigParser()
    config_method1.read(config_path_method1)
    conf_method1 = dict(config_method1.items("hyperparameters"))
    config_path_method2 = 'conf/' + emb_method_name2 + '.properties'
    config_method2 = configparser.ConfigParser()
    config_method2.read(config_path_method2)
    conf_method2 = dict(config_method2.items("hyperparameters"))



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
    embedding_size_method1=int(conf_method1['embedding_size'])
    if emb_method_name1 == 'splitter':
        scores_matrix_one = inner_product_scores_splitter(graph_results_dir=graph_results_dir,
                                 dataset_name = graph_name, emb_method_name= emb_method_name1,
                                 col_start=0, col_end=embedding_size_method1+1, skiprows=1, delimiter=',')
    elif (emb_method_name1 == 'attentionwalk') or (emb_method_name1 == 'grarep'):
        scores_matrix_one = inner_product_scores(graph_results_dir=graph_results_dir,
                                 dataset_name = graph_name, emb_method_name= emb_method_name1,
                                 col_start=0, col_end=embedding_size_method1 +1 , skiprows=1, delimiter=',')
    elif (emb_method_name1 == 'drne') or (emb_method_name1 == 'prune'):
        scores_matrix_one = inner_product_scores(graph_results_dir=graph_results_dir,
                                 dataset_name=graph_name, emb_method_name=emb_method_name1,
                                 col_start=0, col_end=embedding_size_method1, skiprows=0, delimiter=' ') # embedding_size_method有一些是要+1有一些不需要的
    elif (emb_method_name1 == 'arope'):
        scores_matrix_one = inner_product_scores_arope(all_file_dir=all_file_dir,
                                                       graph_name=graph_name,
                                                       graph_results_dir = graph_results_dir)
    elif (emb_method_name1 == 'graph2gauss'):
        scores_matrix_one = energy_kl_scores_graph2gauss(all_file_dir=all_file_dir,
                                                         graph_name=graph_name,
                                                         graph_results_dir = graph_results_dir)
    elif is_heuristic_method(emb_method_name1):
        scores_matrix_one = heuristic_scores(all_file_dir=all_file_dir,
                                             graph_name=graph_name,
                                             graph_results_dir=graph_results_dir,
                                             heuristic_method=emb_method_name1)
    else:
        scores_matrix_one = inner_product_scores(graph_results_dir=graph_results_dir,
                                 dataset_name=graph_name, emb_method_name=emb_method_name1,
                                 col_start=0, col_end=embedding_size_method1+1, skiprows=1, delimiter=' ')


    # 计算scores2
    embedding_size_method2 = int(conf_method2['embedding_size'])
    if emb_method_name2 == 'splitter':
        scores_matrix_two = inner_product_scores_splitter(graph_results_dir=graph_results_dir,
                                 dataset_name = graph_name, emb_method_name= emb_method_name2,
                                 col_start=0, col_end=embedding_size_method2+1, skiprows=1, delimiter=',')
    elif (emb_method_name2 == 'attentionwalk') or (emb_method_name2 == 'grarep'):
        scores_matrix_two = inner_product_scores(graph_results_dir=graph_results_dir,
                                 dataset_name = graph_name, emb_method_name= emb_method_name2,
                                 col_start=0, col_end=embedding_size_method2+1, skiprows=1, delimiter=',')
    elif (emb_method_name2 == 'drne') or (emb_method_name2 == 'prune'):
        scores_matrix_two = inner_product_scores(graph_results_dir=graph_results_dir,
                                 dataset_name=graph_name, emb_method_name=emb_method_name2,
                                 col_start=0, col_end=embedding_size_method2, skiprows=0, delimiter=' ')
    elif (emb_method_name2 == 'arope'):
        scores_matrix_two = inner_product_scores_arope(all_file_dir=all_file_dir,
                                                       graph_name=graph_name,
                                                       graph_results_dir = graph_results_dir)
    elif (emb_method_name2 == 'graph2gauss'):
        scores_matrix_two = energy_kl_scores_graph2gauss(all_file_dir=all_file_dir,
                                                         graph_name=graph_name,
                                                         graph_results_dir = graph_results_dir)
    elif is_heuristic_method(emb_method_name2):
        scores_matrix_two = heuristic_scores(all_file_dir=all_file_dir,
                                             graph_name=graph_name,
                                             graph_results_dir = graph_results_dir,
                                             heuristic_method=emb_method_name2)
    else:
        scores_matrix_two = inner_product_scores(graph_results_dir=graph_results_dir,
                                 dataset_name = graph_name, emb_method_name= emb_method_name2,
                                 col_start=0, col_end=embedding_size_method2+1, skiprows=1, delimiter=' ')



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

    del scores_matrix_one_norm, scores_matrix_two_norm, exist_scores_one_list, exist_scores_two_list, scores_matrix_hybrid_norm
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
    print('----------------------------------------------------------------------------')
    pass