import xlwt
import xlrd
from xlutils.copy import copy
import os
from scipy.special import comb, perm
import matplotlib.pyplot as plt
import numpy as np

source_dir = 'D:\第二个工作-实验数据\paraSen/'

# 'ca-hepph', 'facebook_combined', 'petster-friendships-hamster', 'yeast', 'email-eucore'
dataset = 'facebook_combined'  # 改这里
# 'degreeproduct'
# 'graph2gauss', 'attentionwalk', 'splitter'
# methods_heuristic = ['cn', 'ja', 'aa', 'cosine', 'pearson']
# methods_embedding = ['drne', 'prone', 'deepwalk', 'node2vec']
method_pairs1 = ['cn', 'prone', 'cn', 'ja']  # 改这里
method_pairs2 = ['ja', 'node2vec', 'prone', 'node2vec']  # 改这里
train_ratios = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

# plt.style.use("ggplot")
plt.style.use('seaborn-darkgrid')
# sns.set_style('whitegrid')
hatch1 = "\\\\"
hatch2 = '//'
rotation = 90  # 改这里
bar_width = 0.4
alpha = 1.0
xlabel='train ratios'
ylabel = 'Precision'
ticks_fontsize = 10
legend_fontsize = 8.5
x_y_label_fontsize = 12
title_fontsize = 14
markersize = 4
hspace = 0.8
wspace = 0.24
colors = ['steelblue', 'red', 'green', 'purple', 'hotpink', 'orange']
markers=['s', 'o', '^', '*', '>', '<']

fig = plt.figure(figsize=(14, 3.5))  # figsize=(17, 3)或1
x_labels = ['(a)', '(b)', '(c)', '(d)']

for method_pair_index in range(len(method_pairs1)):
    ax = fig.add_subplot(1, 4, method_pair_index + 1)

    # 获得excel的完整路径
    method1 = method_pairs1[method_pair_index]
    method2 = method_pairs2[method_pair_index]
    excel_name = dataset + '--' + method1 + '--' + method2

    # 初始化数据
    x_data=train_ratios
    y_data_method1=[]
    y_data_method2 = []
    y_data_plus = []
    y_data_multiply = []
    y_data_MLP = []
    y_data_PNR = []
    # print(method1 + method2) # debug用
    for train_ratio_index in range(len(train_ratios)):
         # print(train_ratios[train_ratio_index])  # debug用
        excel_path = source_dir + train_ratios[train_ratio_index] + '//save_all_results//' + excel_name + '.xls'

        # 获取文件的句柄
        read_f = xlrd.open_workbook(excel_path, formatting_info=True)
        table_read = read_f.sheet_by_index(0)

        # 计算每一个图的6个柱子的precision值，这里L=|E^{test}|
        method1_precision = table_read.cell(2, 6).value
        method2_precision = table_read.cell(3, 6).value
        plus_precision = table_read.cell(4, 6).value
        multiply_precision = table_read.cell(49, 6).value
        MLP_precision = table_read.cell(21, 6).value
        PNR_precision = table_read.cell(1, 6).value

        # 组合y_data
        y_data_method1.append(method1_precision)
        y_data_method2.append(method2_precision)
        y_data_plus.append(plus_precision)
        y_data_multiply.append(multiply_precision)
        y_data_MLP.append(MLP_precision)
        y_data_PNR.append(PNR_precision)

    pass
    # 画图
    plt.plot(x_data, y_data_method1,
             marker=markers[0], markersize=markersize,
             label=method_pairs1[method_pair_index],
             color=colors[0])
    plt.plot(x_data, y_data_method2,
             marker=markers[1], markersize=markersize,
             label=method_pairs2[method_pair_index],
             color=colors[1])
    plt.plot(x_data, y_data_plus,
             marker=markers[2], markersize=markersize,
             label=method_pairs1[method_pair_index] + '+' + method_pairs2[method_pair_index],
             color=colors[2])
    plt.plot(x_data, y_data_multiply,
             marker=markers[3], markersize=markersize,
             label=method_pairs1[method_pair_index] + '×' + method_pairs2[method_pair_index],
             color=colors[3])
    plt.plot(x_data, y_data_MLP,
             marker=markers[4], markersize=markersize,
             label='MLP',
             color=colors[4])
    plt.plot(x_data, y_data_PNR,
             marker=markers[5], markersize=markersize,
             label='PNR',
             color=colors[5])

    plt.xlabel(xlabel, color='black', fontsize=x_y_label_fontsize)
    plt.ylabel(ylabel, color='black', fontsize=x_y_label_fontsize)
    plt.xticks(np.arange(len(x_data)), x_data,
               color='black',
               fontsize=ticks_fontsize)
    plt.yticks(color='black', fontsize=ticks_fontsize)
    plt.title(method1 + '&' + method2, fontsize=title_fontsize, color='black')
    plt.legend()

    fig.text(0.24 + 0.21 * method_pair_index, 0.80, x_labels[method_pair_index],
             fontsize=title_fontsize, fontweight='semibold', color='black')


pass

plt.subplots_adjust(wspace=wspace, hspace=hspace)  # 调整子图间距
plt.savefig('D:\hybridrec\临时文件夹//' + dataset + '_precision_paraSen.eps', format='eps',
            bbox_inches='tight')  # pad_inches使得四周的空白都去掉了
plt.savefig('D:\hybridrec\临时文件夹//' + dataset + '_precision_paraSen.png', format='png',
            bbox_inches='tight')  # pad_inches使得四周的空白都去掉了
plt.savefig('D:\hybridrec\临时文件夹//' + dataset + '_precision_paraSen.pdf', format='pdf',
            bbox_inches='tight')  # pad_inches使得四周的空白都去掉了
plt.show()
