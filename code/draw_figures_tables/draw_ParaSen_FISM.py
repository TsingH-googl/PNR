import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.io import loadmat
import seaborn as sns
import time
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


time_start = time.time()
dataset_name=['ML100K', 'Delicious', 'Lastfm', 'Wikibooks']
# plt.style.use("ggplot")
# plt.style.use('seaborn-darkgrid')

# sns.set_style('whitegrid')
subplot_index = 1
ticks_fontsize=10
legend_fontsize=10
x_y_label_fontsize=10
title_fontsize=14
linewidth=1.5
bwith=2
fontweight='semibold'


x_data = [10, 20, 40, 60, 100, 150, 200]

# topL_array = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]
topL_index = 7
fig=plt.figure(figsize=(17, 3)) # 21, 29.7
layout1=1
layout2=4
for dataset_name_index in range(len(dataset_name)):
    base_path = 'D:\链路预测相关课题\matlab评估代码\实验结果\paraSensitivity//'
    excel_path = base_path + dataset_name[dataset_name_index] + '.xls'
    data = xlrd.open_workbook(excel_path)
    table = data.sheets()[0]

    y_data = np.zeros(shape=(9, 12))
    for i in range (9):
        for j in range(12):
            y_data[i, j] = float(table.cell(27 + i, j).value)


    y_data_org = [y_data[0, topL_index] for i in range(len(x_data))]
    fig.add_subplot(layout1, layout2, dataset_name_index + 1)
    plt.plot(x_data, y_data[2:y_data.shape[0], topL_index], linewidth=linewidth,
            label='RNR methods',
            marker='o',
             color='crimson')
    plt.plot(x_data, y_data_org, linewidth=linewidth,
            label='original methods',
            marker='^',
            linestyle='--',
             color='darkblue')


    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel("the number of uniform bin $b$", fontsize=x_y_label_fontsize)
    plt.ylabel("ARHR@N", fontsize=x_y_label_fontsize)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    # xmajorLocator = MultipleLocator(20)  # 将x主刻度标签设置为20的倍数
    # xminorLocator = MultipleLocator(5)  # 将x轴次刻度标签设置为5的倍数
    # ymajorLocator = MultipleLocator(0.5)  # 将y轴主刻度标签设置为0.5的倍数
    # yminorLocator = MultipleLocator(0.1)  # 将此y轴次刻度标签设置为0.1的倍数
    # ax.xaxis.set_major_locator(xmajorLocator)
    # ax.yaxis.set_major_locator(ymajorLocator)



    ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
    ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度
    ax.xaxis.grid(True, which='minor')  # x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major')  # y坐标轴的网格使用次刻度


    subplot_index = subplot_index + 1
    plt.legend()
    plt.title(dataset_name[dataset_name_index])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    pass
pass

fig.text(0.20, -0.12, '(a) ', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.40, -0.12, '(b) ', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.60, -0.12, '(c) ', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.80, -0.12, '(d) ', fontsize=title_fontsize, fontweight=fontweight)

# plt.subplots_adjust(wspace =wspace, hspace =wspace)#调整子图间距
plt.savefig('./figures//' + 'paraSen.eps', format='eps',
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了

plt.show()
time_end = time.time()

print("It takes: " + str((time_end - time_start)/ 60)  + " mins.")




