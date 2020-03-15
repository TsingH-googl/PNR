import matplotlib.pyplot as plt
import numpy as np
import xlrd



xls_file_path  = 'D:\链路预测相关课题\matlab评估代码\实验结果//precision.xls'
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


# 读取四个数据集的数据
topL_len = 4
topL_start = 2 # 改这里，0, 1, 2, 3, 4
x_data=[]
x_data_group = [5, 10, 15, 20, 25, 30, 40, 50]
for i in range(topL_len):
    x_data.append(x_data_group[i + topL_start])
for k in  range(7):
    for i in range(topL_len):
        ml100k_org_data[k].append(float(table.cell(1 + 2*k, 2 + i + topL_start).value))
        ml100k_rnr_data[k].append(float(table.cell(2 + 2*k, 2 + i + topL_start).value))
    pass
pass
for k in  range(7):
    for i in range(topL_len):
        delicious_org_data[k].append(float(table.cell(16 + 2*k, 2 + i + topL_start).value))
        delicious_rnr_data[k].append(float(table.cell(17 + 2*k, 2 + i + topL_start).value))
    pass
pass
for k in  range(7):
    for i in range(topL_len):
        lastfm_org_data[k].append(float(table.cell(31 + 2*k, 2 + i + topL_start).value))
        lastfm_rnr_data[k].append(float(table.cell(32 + 2*k, 2 + i + topL_start).value))
    pass
pass
for k in  range(7):
    for i in range(topL_len):
        wikibooks_org_data[k].append(float(table.cell(46 + 2*k, 2 + i + topL_start).value))
        wikibooks_rnr_data[k].append(float(table.cell(47 + 2*k, 2 + i + topL_start).value))
    pass
pass



# 画图
plt.figure(figsize=(10, 6.5)) # figsize=(17, 3)或1
# plt.style.use("ggplot")
fontsize=7.5
legend_fontsize=5.5
x_y_label_fontsize=9
fontweight='bold' # [‘light’, ‘normal’, ‘medium’, ‘semibold’, ‘bold’, ‘heavy’, ‘black’]
title_fontsize=10
hspace=1.0
wspace=0.42
my_x_ticks = np.arange(min(x_data), max(x_data) + 1, 5)

colors=['red', 'blue', 'green']

subplot_index = 1
org_baselines = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
rnr_baselines = ['RNR-ItemCF', 'RNR-UserCF', 'RNR-PureSVD', 'RNR-FISM', 'RNR-MultiDAE', 'RNR-APR', 'RNR-JCA']
for i in range(7):
    plt.subplot(4, 7, subplot_index)
    subplot_index = subplot_index + 1
    plt.plot(x_data, ml100k_org_data[i], marker='o', label=org_baselines[i])
    plt.plot(x_data, ml100k_rnr_data[i], marker='*', label=rnr_baselines[i])
    plt.xticks(my_x_ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize)
    if i==0:
        plt.ylabel("precision@N", fontsize=x_y_label_fontsize)
    if i==3:
        plt.title("ML100K", fontsize=title_fontsize, fontweight=fontweight)
    # plt.grid()
    ax = plt .gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.yaxis.get_offset_text().set_fontsize(legend_fontsize)
pass
for i in range(7):
    plt.subplot(4, 7, subplot_index)
    subplot_index = subplot_index + 1
    plt.plot(x_data, delicious_org_data[i], marker='o', label=org_baselines[i])
    plt.plot(x_data, delicious_rnr_data[i], marker='*', label=rnr_baselines[i])
    plt.xticks(my_x_ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize)
    if i==0:
        plt.ylabel("precision@N", fontsize=x_y_label_fontsize)
    if i==3:
        plt.title("Delicious", fontsize=title_fontsize, fontweight=fontweight)
    ax = plt .gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.yaxis.get_offset_text().set_fontsize(legend_fontsize)
pass
for i in range(7):
    plt.subplot(4, 7, subplot_index)
    subplot_index = subplot_index + 1
    plt.plot(x_data, lastfm_org_data[i], marker='o', label=org_baselines[i])
    plt.plot(x_data, lastfm_rnr_data[i], marker='*', label=rnr_baselines[i])
    plt.xticks(my_x_ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize)
    if i==0:
        plt.ylabel("precision@N", fontsize=x_y_label_fontsize)
    if i==3:
        plt.title("Lastfm", fontsize=title_fontsize, fontweight=fontweight)
    ax = plt .gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.yaxis.get_offset_text().set_fontsize(legend_fontsize)
pass
for i in range(7):
    plt.subplot(4, 7, subplot_index)
    subplot_index = subplot_index + 1
    plt.plot(x_data, wikibooks_org_data[i], marker='o', label=org_baselines[i])
    plt.plot(x_data, wikibooks_rnr_data[i], marker='*', label=rnr_baselines[i])
    plt.xticks(my_x_ticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize)
    if i==0:
        plt.ylabel("precision@N", fontsize=x_y_label_fontsize)
    if i==3:
        plt.title("Wikibooks", fontsize=title_fontsize, fontweight=fontweight)
    plt.xlabel("N", fontsize=x_y_label_fontsize)
    ax = plt .gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.yaxis.get_offset_text().set_fontsize(legend_fontsize)
pass


plt.subplots_adjust(wspace =wspace, hspace =wspace)#调整子图间距
plt.savefig('./figures//' + 'precision.eps', format='eps', dpi=600,
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了
plt.show()

