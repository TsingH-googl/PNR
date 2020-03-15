import xlwt
import xlrd
from xlutils.copy import copy
import os
from scipy.special import comb, perm
import matplotlib.pyplot as plt
import numpy as np


source_dir='D:\hybridrec//results//'

# 'ca-hepph', 'facebook_combined', 'petster-friendships-hamster', 'yeast', 'email-eucore'
dataset='ca-hepph' # 改这里
# 'degreeproduct'
# 'graph2gauss', 'attentionwalk', 'splitter'
# methods_heuristic = ['cn', 'ja', 'aa', 'cosine', 'pearson']
# methods_embedding = ['drne', 'prone', 'deepwalk', 'node2vec']
method_pairs1=['cn', 'prone', 'cn', 'ja'] # 改这里
method_pairs2=['ja', 'node2vec', 'prone', 'node2vec'] # 改这里

# plt.style.use("ggplot")
plt.style.use('seaborn-darkgrid')
# sns.set_style('whitegrid')
hatch1 = "\\\\"
hatch2 = '//'
rotation = 90 # 改这里
bar_width = 0.4
alpha = 1.0
ylabel = 'AP'
ticks_fontsize = 12
legend_fontsize = 12
x_y_label_fontsize = 12
title_fontsize = 14
hspace = 0.8
wspace = 0.41

fig = plt.figure(figsize=(13, 2.5))  # figsize=(17, 3)或1
x_labels=['(a)','(b)','(c)','(d)']

for method_pair_index in range(len(method_pairs1)):
    ax = fig.add_subplot(1, 4, method_pair_index + 1)

    # 获得excel的完整路径
    method1=method_pairs1[method_pair_index]
    method2=method_pairs2[method_pair_index]
    excel_name = dataset + '--' + method1 + '--' + method2
    excel_path = source_dir + excel_name + '.xls'

    # 获取文件的句柄
    read_f = xlrd.open_workbook(excel_path, formatting_info=True)
    table_read = read_f.sheet_by_index(0)

    # 计算每一个图的6个柱子的AP值，这里L=|E^{test}|
    method1_AP = table_read.cell(14, 6).value
    method2_AP = table_read.cell(15, 6).value
    plus_AP = table_read.cell(16, 6).value
    multiply_AP = table_read.cell(52, 6).value
    MLP_AP = table_read.cell(24, 6).value
    PNR_AP = table_read.cell(13, 6).value

    ######################### 画图 ##########################
    x_data = []
    y_data = []
    x_data.append(method1)
    x_data.append(method2)
    x_data.append(method1 + '+' + method2)  # method1+'\n + \n'+method2
    x_data.append(method1 + '×' + method2)  # method1+'\n × \n'+method2
    x_data.append('MLP')
    x_data.append('PNR')
    y_data.append(method1_AP)
    y_data.append(method2_AP)
    y_data.append(plus_AP)
    y_data.append(multiply_AP)
    y_data.append(MLP_AP)
    y_data.append(PNR_AP)

    plt.bar(np.arange(len(x_data)), y_data, width=bar_width,
            color='salmon', alpha=alpha, hatch=hatch1)
    # plt.xlabel('('+str(method_pair_index+1)+')',
    #            color='black', fontsize=x_y_label_fontsize)
    plt.ylabel(ylabel, color='black', fontsize=x_y_label_fontsize)
    plt.xticks(np.arange(len(x_data)), x_data,
               color='black', rotation=rotation,
               fontsize=ticks_fontsize)
    plt.yticks(color='black', fontsize=ticks_fontsize)
    plt.title(method1 + '&' + method2, fontsize=title_fontsize, color='black')

    fig.text(0.13 + 0.21 * method_pair_index, 0.80, x_labels[method_pair_index],
             fontsize=title_fontsize, fontweight='semibold', color='black')

pass

plt.subplots_adjust(wspace=wspace, hspace=hspace)  # 调整子图间距
plt.savefig('D:\hybridrec\临时文件夹//' + dataset+'_AP_demo.eps', format='eps',
            bbox_inches='tight')  # pad_inches使得四周的空白都去掉了
plt.savefig('D:\hybridrec\临时文件夹//' + dataset+'_AP_demo.png', format='png',
            bbox_inches='tight')  # pad_inches使得四周的空白都去掉了
plt.savefig('D:\hybridrec\临时文件夹//' + dataset+'_AP_demo.pdf', format='pdf',
            bbox_inches='tight')  # pad_inches使得四周的空白都去掉了
plt.show()


