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
import matplotlib.colors as col
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier

# import evaluators, stacking, etc.
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
# Package for stacking models
# from vecstack import stacking
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import cv2
import warnings
warnings.filterwarnings("ignore")


import scipy.misc


# from utils import read_graph, normalize_matrix, normalize_matrix_full, get_trainset_path, get_testset_path, get_test_matrix_binary, \
#     save_DNN_hybrid_scores, save_plus_raster_scores, save_plus_hybrid_scores, get_list_thresold
# from utils_DNN import negative_samples, predicted_scores_DNN, rasterization_grids, better_show_grids
# from evaluation import evaluators
# from fileIO_utils import is_excel_file_exist, plus_write_to_excel
# from utils_plot import plot_contourf, plot_heatmap, plot_contourf_overlap
# from evaluation import transfer_scores_PNR, evaluators
def normalize_matrix_full(csr_matrix1):
    all_binary = csr_matrix((np.ones(csr_matrix1.shape)))
    all_scores = (np.array(csr_matrix1[all_binary > 0], dtype=float))[0]

    array_matrix = csr_matrix1.A
    # amin, amax = array_matrix.min(), array_matrix.max()# 求最大最小值
    amin = min(all_scores)
    amax = max(all_scores)
    array_matrix = (array_matrix-amin)/(amax-amin)# (矩阵元素-最小值)/(最大值-最小值)

    return csr_matrix(array_matrix)

def plot_contourf_overlap(result=None, title=None, color=None):
    result = normalize_matrix_full(csr_matrix1=csr_matrix(result))
    fig = plt.figure(figsize=(10, 10))  # 21, 29.7
    # 准备画图数据
    result = csr_matrix(result)
    x = np.linspace(0, 1, result.shape[0])
    y = np.linspace(0, 1, result.shape[0])
    [X, Y] = np.meshgrid(x, y)
    Z = result.A
    # 为等高线图填充颜色, 16指定将等高线分为几部分
    colorslist = ['white']
    colorslist.append(color)
    # 将颜色条命名为mylist，一共插值颜色条50个
    cmap = col.LinearSegmentedColormap.from_list('mylist', colorslist, N=len(colorslist) * 1)
    temp = plt.contourf(X, Y, Z, 1, alpha=1.0, cmap=cmap)  # 使用颜色映射来区分不同高度的区域
    plt.contour(X, Y, Z, [temp._A[-2]], linewidths=1.0, alpha=1.0, colors='black')  # 使用颜色映射来区分不同高度的区域
    # ax = plt.axes()
    # ax.set_title(title, fontsize=18, position=(0.5, 1.05))

    plt.xticks([])
    plt.yticks([])
    # plt.ylim(0.0, 1.0)
    # plt.xticks([0.0, 0.5, 1.0], fontsize=20)
    # plt.yticks([0.0, 0.5, 1.0], fontsize=20)

    foo_fig = plt.gcf()  # 'get current figure'
    foo_fig.savefig('C://Users\Rong-qin Xu\Desktop//figure-KDD2020//' + title + '.png', format='png',
                    bbox_inches='tight',
                    pad_inches=0
                    )
    plt.show()


base_dir = 'C://Users\Rong-qin Xu\Desktop\scores-KDD2020//'
graph_name= 'facebook_combined'
emb_method_name1='deepwalk'
emb_method_name2='node2vec'
PNR_ratio=0.98
mlp_ratio=0.7
plus_ratio=0.55
multiply_ratio=0.55


PNR_grids_path = base_dir + graph_name + "//" + 'PNR2_'+ graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 +"_50_count.mat"
mlp_grids_path = base_dir + graph_name + "//" + 'mlp_'+ graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 +"_50_count.mat"
plus_grids_path = base_dir + graph_name + "//" + 'plus_'+ graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 +"_50_count.mat"
multiply_grids_path = base_dir + graph_name + "//" + 'multiply_'+ graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 +"_50_count.mat"

# 读取pn背景
# pn_grids_path = base_dir + graph_name + "//" + 'nonexist_raster_grids_' +  graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 +"_50_count.mat"
# pn_grids = loadmat(pn_grids_path)
# pn_grids = pn_grids['count'].A
# pn_grids[pn_grids > 0.0]=1.0

scores_matrix_PNR_dict = (loadmat(PNR_grids_path))
scores_matrix_mlp_dict = (loadmat(mlp_grids_path))
scores_matrix_plus_dict = (loadmat(plus_grids_path))
scores_matrix_multiply_dict = (loadmat(multiply_grids_path))

PNR_grids = scores_matrix_PNR_dict['count'].A
mlp_grids = scores_matrix_mlp_dict['count'].A
plus_grids = scores_matrix_plus_dict['count'].A
multiply_grids = scores_matrix_multiply_dict['count'].A

PNR_list = PNR_grids[PNR_grids>0]
mlp_list = mlp_grids[mlp_grids>0]
plus_list = plus_grids[plus_grids>0]
multiply_list = multiply_grids[multiply_grids>0]
PNR_list=np.sort(PNR_list)
mlp_list=np.sort(mlp_list)
plus_list=np.sort(plus_list)
multiply_list=np.sort(multiply_list)

PNR_thresold = PNR_list[int(len(PNR_list) * (1.0-PNR_ratio))]
mlp_thresold = mlp_list[int(len(mlp_list) * (1.0-mlp_ratio))-1]
plus_thresold = plus_list[int(len(plus_list) * (1.0-plus_ratio))-1]
multiply_thresold = multiply_list[int(len(multiply_list) * (1.0-multiply_ratio))-1]


PNR_grids[PNR_grids <= PNR_thresold] = 0.0
mlp_grids[mlp_grids <= mlp_thresold] = 0.0
plus_grids[plus_grids <= plus_thresold] = 0.0
multiply_grids[multiply_grids <= multiply_thresold] = 0.0


PNR_grids[PNR_grids > PNR_thresold] = 1.0
mlp_grids[mlp_grids > mlp_thresold] = 1.0
plus_grids[plus_grids > plus_thresold] = 1.0
multiply_grids[multiply_grids > multiply_thresold] = 1.0

PNR_grids = np.float32(PNR_grids)
PNR_grids = cv2.GaussianBlur(PNR_grids, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
mlp_grids = np.float32(mlp_grids)
mlp_grids = cv2.GaussianBlur(mlp_grids, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
plus_grids = np.float32(plus_grids)
plus_grids = cv2.GaussianBlur(plus_grids, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
multiply_grids = np.float32(multiply_grids)
multiply_grids = cv2.GaussianBlur(multiply_grids, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
# pn_grids = np.float32(pn_grids)
# pn_grids = cv2.GaussianBlur(pn_grids, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0


plot_contourf_overlap(result=PNR_grids, title='PNR', color='firebrick')
plot_contourf_overlap(result=mlp_grids, title='mlp', color='darkorchid')
plot_contourf_overlap(result=plus_grids, title='plus', color='forestgreen')
plot_contourf_overlap(result=multiply_grids, title='multiply', color='royalblue')
# plot_contourf_overlap(result=pn_grids, title='pn', color='gray')



# 权重越大，透明度越低
pic1 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/' + graph_name + '_' + emb_method_name1 + '_' + emb_method_name2+'_bg.png')
pic2 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/PNR.png')
pic3 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/mlp.png')
pic4 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/plus.png')
pic5 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/multiply.png')
overlap_PNR = cv2.addWeighted(pic1, 0.5, pic2, 0.5, 0)
overlap_mlp = cv2.addWeighted(pic1, 0.5, pic3, 0.5, 0)
overlap_plus = cv2.addWeighted(pic1, 0.5, pic4, 0.5, 0)
overlap_multiply = cv2.addWeighted(pic1, 0.5, pic5, 0.5, 0)
# 保存叠加后的图片
cv2.imwrite('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/overlap_PNR_' + graph_name + '_' + emb_method_name1 + '_' + emb_method_name2
            + '.png', overlap_PNR)
cv2.imwrite('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/overlap_mlp_' + graph_name + '_' + emb_method_name1 + '_' + emb_method_name2
            + '.png', overlap_mlp)
cv2.imwrite('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/overlap_plus_' + graph_name + '_' + emb_method_name1 + '_' + emb_method_name2
            + '.png', overlap_plus)
cv2.imwrite('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/overlap_multiply_' + graph_name + '_' + emb_method_name1 + '_' + emb_method_name2
            + '.png', overlap_multiply)
# pic1 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/PNR.png')
# pic2 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/PNR.png')
# pic3 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/PNR.png')
# pic4 = cv2.imread('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/PNR.png')
# pic12 = cv2.addWeighted(pic1, 0.5, pic2, 0.5, 0)
# pic34 = cv2.addWeighted(pic4, 0.5, pic3, 0.5, 0)
# overlap_pic = cv2.addWeighted(pic34, 0.5, pic12, 0.5, 0)
# # 保存叠加后的图片
# cv2.imwrite('C://Users\Rong-qin Xu\Desktop//figure-KDD2020/overlap.png', overlap_pic)

