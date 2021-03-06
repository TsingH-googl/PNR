import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.io import loadmat
import seaborn as sns
import time

time_start = time.time()
dataset_name=['ML100K', 'Delicious', 'LastFM', 'Wikibooks']
# plt.style.use("ggplot")
# plt.style.use('seaborn-darkgrid')

# sns.set_style('whitegrid')
subplot_index = 1
hspace=0.32
wspace=0.36
ticks_fontsize=16
legend_fontsize=16
x_y_label_fontsize=16
title_fontsize=20
linewidth=1.5
bwith=2
fontweight='semibold'

x_data = [10, 20, 30,40,	50,	60,	70,	80,	90,	100,	110,	120,	130,	140,	150,	160,	170,
          180,	190,	200,	220,	240,	260,	280,	300]
reserve_index = [0, 1, 2,3,4,5,6,7,8,9,11,13,15,17,19]
x_data_reserve=[]
for k_reserve in range(len(reserve_index)):
    x_data_reserve.append(x_data[reserve_index[k_reserve]])
pass

# topL_array = [5, 10, 15, 20, 25, 30, 40, 50]
topL_index = 0
fig=plt.figure(figsize=(21, 26.7)) # 21, 29.7
# fig.text(0, 0.147, 'JCA', fontsize=title_fontsize, fontweight=fontweight)
# fig.text(0, 0.267, 'APR', fontsize=title_fontsize, fontweight=fontweight)
# fig.text(0, 0.375, 'MultiDAE', fontsize=title_fontsize, fontweight=fontweight)
# fig.text(0, 0.485, 'FISM', fontsize=title_fontsize, fontweight=fontweight)
# fig.text(0, 0.60, 'PureSVD', fontsize=title_fontsize, fontweight=fontweight)
# fig.text(0, 0.72, 'UserCF', fontsize=title_fontsize, fontweight=fontweight)
# fig.text(0, 0.825, 'ItemCF', fontsize=title_fontsize, fontweight=fontweight)

fig.text(0.02, 0.147, 'JCA', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.267, 'APR', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.01, 0.375, 'MultiDAE', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.485, 'FISM', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.60, 'PureSVD', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.72, 'UserCF', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.825, 'ItemCF', fontsize=title_fontsize, fontweight=fontweight)

for baseline_name_index in range(7):
    for dataset_name_index in range(len(dataset_name)):
        base_path = 'D:\链路预测相关课题\matlab评估代码\实验结果\paraSensitivity_NDCG//'
        excel_path = base_path + dataset_name[dataset_name_index] + '_ndcg.xls'
        data = xlrd.open_workbook(excel_path)
        table = data.sheets()[0]

        y_data = np.zeros(shape=(26, 8))
        row_index = baseline_name_index * 26
        for i in range(26):
            for j in range(8):
                y_data[i, j] = float(table.cell(row_index + i, j).value)
            pass
        pass


        plt.subplot(7, len(dataset_name), subplot_index)
        y_data_org = [y_data[0, topL_index] for i in range(len(x_data_reserve))]
        y_data_topL = y_data[1:y_data.shape[0], topL_index]
        y_data_topL_reserve = []
        for k_reserve in range(len(reserve_index)):
            y_data_topL_reserve.append(y_data_topL[reserve_index[k_reserve]])
        pass

        plt.plot(x_data_reserve, y_data_topL_reserve, linewidth=linewidth,
                 label='RNR-FISM method',
                 marker='o',
                 color='crimson')
        plt.plot(x_data_reserve, y_data_org, linewidth=linewidth,
                 label='FISM method',
                 marker='^',
                 linestyle='--',
                 color='darkblue')


        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.xlabel("the number of uniform bins $b$", fontsize=x_y_label_fontsize)
        plt.ylabel("NDCG@20", fontsize=x_y_label_fontsize)
        if baseline_name_index == 0:
            plt.title(label=dataset_name[dataset_name_index] + '\n', fontsize=title_fontsize, fontweight=fontweight)

        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
        subplot_index = subplot_index + 1


        ax.xaxis.grid(True, which='major')  # x坐标轴的网格使用主刻度
        ax.yaxis.grid(True, which='minor')  # y坐标轴的网格使用次刻度
        ax.xaxis.grid(True, which='minor')  # x坐标轴的网格使用主刻度
        ax.yaxis.grid(True, which='major')  # y坐标轴的网格使用次刻度

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'


        plt.legend()



    pass
pass



plt.subplots_adjust(wspace =wspace, hspace =wspace)#调整子图间距
plt.savefig('./figures//' + 'paraSen_all_ndcg.png', format='png',
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了

plt.show()
time_end = time.time()

print("It takes: " + str((time_end - time_start)/ 60)  + " mins.")





