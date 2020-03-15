from __future__ import division
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import gc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
from utils import normalize_matrix_full
import warnings
warnings.filterwarnings("ignore")

def plot_contourf_overlap(result=None, title=None, color=None):
    result = normalize_matrix_full(csr_matrix1=csr_matrix(result))

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
    foo_fig = plt.gcf()  # 'get current figure'
    foo_fig.savefig('D:\第二个工作-实验数据\overlap//' + title + '.eps', format='eps')
    plt.show()


# prex = 'preprocessing_code//'  # 改这里
# results_base_dir = 'D:\hybridrec//results//'
# all_file_dir = 'D:\hybridrec\dataset\split_train_test//' + prex
# graph_name = 'facebook_combined'  # 改这里
# emb_method_name1 = 'aa'.lower()  # 改这里
# emb_method_name2 = 'ra'.lower()  # 改这里
# results_dir = 'D:\hybridrec/results//' + prex
# graph_results_dir = results_dir + graph_name + '//'
# # （facebook_combined的规律：ratio越小则正负样本的预测准确率越高，花的时间也越少）
# ratio = 1  # 负样本的总数是正样本的ratio倍  # 改这里
# binNum = 10  # DNN rasterization grids # 改这里
# model_name = "mlp".lower()  # 改这里  mlp  svm  lr  lgbm  xgb  ld  rf
#
#
# # 读取PNR grids
# PNR_path = results_base_dir + prex + graph_name + "//" + "PNR1_" + graph_name + "_" + emb_method_name1 + "_" + emb_method_name2 + "_50_count.mat"
# PNR_dict = (loadmat(PNR_path))
# PNR_matrix = PNR_dict["count"]
# PNR_matrix = better_show_grids(csr_matrix1=PNR_matrix)
# source = np.float32(PNR_matrix.A)
# result = cv2.GaussianBlur(source, (5, 5), 0)  # (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
def plot_contourf(result=None, title=None, binNum=10):
    result = normalize_matrix_full(csr_matrix1=csr_matrix(result))

    # 设置颜色映射cmap
    cmap = sns.diverging_palette(50, 20, sep=16, as_cmap=True, n=1)
    cmap = sns.light_palette((260, 75, 60), input="husl")  # input: {'rgb'，'hls'，'husl'，xkcd'}
    cmap = sns.dark_palette((260, 75, 60), input="husl", reverse=True)
    # startcolor = '#ffffff'   #红色，读者可以自行修改 #ff0000
    # midcolor = '#0000ff'     #绿色，读者可以自行修改  #00ff00
    # endcolor = '#ff0000'          #蓝色，读者可以自行修改  #0000ff
    # cmap = col.LinearSegmentedColormap.from_list('own2',[startcolor,midcolor,endcolor])

    # 准备画图数据
    result = csr_matrix(result)
    x = np.linspace(0, 1, result.shape[0])
    y = np.linspace(0, 1, result.shape[0])
    [X, Y] = np.meshgrid(x, y)
    Z = result.A
    # 为等高线图填充颜色, 16指定将等高线分为几部分
    # colorslist = ['GhostWhite', 'LightGray', 'LightBLue', 'SkyBlue', 'LightGoldenrodYellow', 'OrangeRed',  'DarkMagenta']
    colorslist = ['GhostWhite', 'LightGray', 'LightBLue', 'SkyBlue', 'LightGoldenrodYellow', 'OrangeRed']
    # 将颜色条命名为mylist，一共插值颜色条50个
    cmap = col.LinearSegmentedColormap.from_list('mylist', colorslist, N=len(colorslist) * 50)
    temp = plt.contourf(X, Y, Z, binNum, alpha=1.0, cmap=cmap)  # 使用颜色映射来区分不同高度的区域
    plt.colorbar()
    C = plt.contour(X, Y, Z, [temp._A[binNum - 2]], linewidths=1.0, alpha=1.0, colors='black')  # 使用颜色映射来区分不同高度的区域
    # plt.clabel(C, inline = True, fontsize = 10)
    ax = plt.axes()
    ax.set_title(title, fontsize=18, position=(0.5, 1.05))
    foo_fig = plt.gcf()  # 'get current figure'
    foo_fig.savefig('./figures//' + title + '.png', format='png', dpi=600)
    plt.show()

def plot_heatmap(result=None, title=None):
    sns.heatmap(result, cmap='Reds', linewidths=0.02)  # cmap='RdBu' 'Reds'
    ax = plt.axes()
    ax.set_title(title, fontsize=18, position=(0.5, 1.05))
    # foo_fig = plt.gcf()  # 'get current figure'
    # foo_fig.savefig(title + '.eps', format='eps', dpi=1000)
    plt.show()









