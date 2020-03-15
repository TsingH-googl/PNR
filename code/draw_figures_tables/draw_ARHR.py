import matplotlib.pyplot as plt
import numpy as np
import xlrd



xls_file_path  = 'D:\链路预测相关课题\matlab评估代码\实验结果//ARHR.xlsx'
data = xlrd.open_workbook(xls_file_path)
table = data.sheets()[0]
ml100k_org_data = [list() for i in range(7)]
ml100k_rnr_data = [list() for i in range(7)]
delicious_org_data = [list() for i in range(7)]
delicious_rnr_data = [list() for i in range(7)]
lastfm_org_data = [list() for i in range(7)]
lastfm_rnr_data = [list() for i in range(7)]
wikibooks_org_data = [list() for i in range(7)]
wikibooks_rnr_data = [list() for i in range(7)]

# plt.style.use("ggplot")
# plt.style.use('seaborn-darkgrid')

# sns.set_style('whitegrid')

# 读取四个数据集的数据
topL_len = 4
x_data = [5, 10, 15, 20]
for k in  range(7):
    for i in range(topL_len):
        ml100k_org_data[k].append(float(table.cell(1 + 2*k, 2 + i).value))
        ml100k_rnr_data[k].append(float(table.cell(2 + 2*k, 2 + i).value))
    pass
pass
for k in  range(7):
    for i in range(topL_len):
        delicious_org_data[k].append(float(table.cell(16 + 2*k, 2 + i).value))
        delicious_rnr_data[k].append(float(table.cell(17 + 2*k, 2 + i).value))
    pass
pass
for k in  range(7):
    for i in range(topL_len):
        lastfm_org_data[k].append(float(table.cell(31 + 2*k, 2 + i).value))
        lastfm_rnr_data[k].append(float(table.cell(32 + 2*k, 2 + i).value))
    pass
pass
for k in  range(7):
    for i in range(topL_len):
        wikibooks_org_data[k].append(float(table.cell(46 + 2*k, 2 + i).value))
        wikibooks_rnr_data[k].append(float(table.cell(47 + 2*k, 2 + i).value))
    pass
pass



# 画图
plt.figure(figsize=(10, 6.5)) # figsize=(17, 3)或1
# plt.style.use("ggplot")
legend_fontsize=5.7
x_y_label_fontsize=9
fontweight='bold' # [‘light’, ‘normal’, ‘medium’, ‘semibold’, ‘bold’, ‘heavy’, ‘black’]
title_fontsize=10
ticks_fontsize=9
markersize = 4
hspace=0.2
wspace=0.45
my_x_ticks = np.arange(5, 25, 5)

org_color = 'blue'
dataset_colors = ['red', 'green', 'purple', 'hotpink']

subplot_index = 1
org_baselines = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
rnr_baselines = ['RNR-ItemCF', 'RNR-UserCF', 'RNR-PureSVD', 'RNR-FISM', 'RNR-MultiDAE', 'RNR-APR', 'RNR-JCA']
for i in range(7):
    plt.subplot(4, 7, subplot_index)
    subplot_index = subplot_index + 1
    plt.plot(x_data, ml100k_org_data[i], marker='s', markersize=markersize, label=org_baselines[i], color=org_color)
    plt.plot(x_data, ml100k_rnr_data[i], marker='o', markersize=markersize, label=rnr_baselines[i],
             color=dataset_colors[0])
    plt.xticks(my_x_ticks, fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=legend_fontsize)
    if i==0:
        plt.ylabel("ARHR@N", fontsize=x_y_label_fontsize)
    if i==3:
        plt.title("ML100K", fontsize=title_fontsize, fontweight=fontweight)
    # plt.grid()
    ax = plt .gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.yaxis.get_offset_text().set_fontsize(legend_fontsize)

    ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度
    ax.xaxis.grid(True, which='minor')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major')  # y坐标轴的网格使用次刻度

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

pass
for i in range(7):
    plt.subplot(4, 7, subplot_index)
    subplot_index = subplot_index + 1
    plt.plot(x_data, delicious_org_data[i], marker='s', markersize = markersize,label=org_baselines[i], color=org_color)
    plt.plot(x_data, delicious_rnr_data[i], marker='o', markersize = markersize,label=rnr_baselines[i], color=dataset_colors[1])
    plt.xticks(my_x_ticks, fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=legend_fontsize)
    if i==0:
        plt.ylabel("ARHR@N", fontsize=x_y_label_fontsize)
    if i==3:
        plt.title("Delicious", fontsize=title_fontsize, fontweight=fontweight)

    ax = plt .gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.yaxis.get_offset_text().set_fontsize(legend_fontsize)

    ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度
    ax.xaxis.grid(True, which='minor')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major')  # y坐标轴的网格使用次刻度

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
pass
for i in range(7):
    plt.subplot(4, 7, subplot_index)
    subplot_index = subplot_index + 1
    plt.plot(x_data, lastfm_org_data[i], marker='s', markersize = markersize,label=org_baselines[i], color=org_color)
    plt.plot(x_data, lastfm_rnr_data[i], marker='o', markersize = markersize,label=rnr_baselines[i], color=dataset_colors[2])
    plt.xticks(my_x_ticks, fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=legend_fontsize)
    if i==0:
        plt.ylabel("ARHR@N", fontsize=x_y_label_fontsize)
    if i==3:
        plt.title("Lastfm", fontsize=title_fontsize, fontweight=fontweight)

    ax = plt .gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.yaxis.get_offset_text().set_fontsize(legend_fontsize)

    ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度
    ax.xaxis.grid(True, which='minor')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major')  # y坐标轴的网格使用次刻度

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

pass
for i in range(7):
    plt.subplot(4, 7, subplot_index)
    subplot_index = subplot_index + 1
    plt.plot(x_data, wikibooks_org_data[i], marker='s', markersize=markersize, label=org_baselines[i], color=org_color)
    plt.plot(x_data, wikibooks_rnr_data[i], marker='o', markersize=markersize, label=rnr_baselines[i],
             color=dataset_colors[3])

    plt.xticks(my_x_ticks, fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.legend(fontsize=legend_fontsize)
    if i==0:
        plt.ylabel("ARHR@N", fontsize=x_y_label_fontsize)
    if i==3:
        plt.title("Wikibooks", fontsize=title_fontsize, fontweight=fontweight)
    plt.xlabel("N", fontsize=x_y_label_fontsize)

    ax = plt .gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.yaxis.get_offset_text().set_fontsize(legend_fontsize)

    ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度
    ax.xaxis.grid(True, which='minor')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major')  # y坐标轴的网格使用次刻度

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

pass


plt.subplots_adjust(wspace =wspace, hspace =wspace)#调整子图间距
plt.savefig('./figures//' + 'ARHR_all.eps', format='eps',
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了
plt.show()

