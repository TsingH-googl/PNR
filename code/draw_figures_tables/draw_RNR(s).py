import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.io import loadmat
import seaborn as sns
import time

time_start = time.time()
dataset_name=['ML100K', 'Delicious', 'LastFM', 'Wikibooks']
baseline_name=['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# plt.style.use("ggplot")
# plt.style.use('seaborn-darkgrid')

# sns.set_style('whitegrid')
subplot_index = 1
hspace=0.22
wspace=0.28
my_x_ticks = np.arange(0.0, 1.01, 0.2)
ticks_fontsize=15
legend_fontsize=14
x_y_label_fontsize=15
title_fontsize=22
filter_color=['tomato', 'hotpink', 'dodgerblue', 'darkorange', 'seagreen', 'magenta', 'crimson']
org_color='gray'
filter_linewidth=1
org_linewidth=1
bwith=2
fontweight='semibold'
baseline_name_fontsize = 24
baseline_name_count = 0

fig=plt.figure(figsize=(21, 26.7)) # 21, 29.7
fig.text(0.02, 0.147, 'JCA', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.267, 'APR', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.375, 'MultiDAE', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.485, 'FISM', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.60, 'PureSVD', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.72, 'UserCF', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0.02, 0.825, 'ItemCF', fontsize=title_fontsize, fontweight=fontweight)
for baseline_name_index in range(7):
    for dataset_name_index in range(len(dataset_name)):
        mat_file_path = 'C://Users\zmy201\Desktop\实验数据//' + \
                    dataset_name[dataset_name_index] + '-' + \
                    baseline_name[baseline_name_index] + '.mat'
        all_data = loadmat(mat_file_path)
        rnr_org = all_data['pnr']
        rnr_filtered = all_data['pnrGau']
        plt.subplot(7, len(dataset_name), subplot_index)
        plt.plot(rnr_org[:, 0], rnr_org[:, 1],
                 marker='o',
                 label='RNR',
                 color=org_color,
                 linewidth=org_linewidth)
        plt.plot(rnr_filtered[:, 0], rnr_filtered[:, 1],
                 marker='*',
                 label='RNR-GaussianFilter',
                 color='deeppink', #filter_color[baseline_name_index], # 'royalblue'
                 linewidth=filter_linewidth)
        plt.xticks(my_x_ticks, fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel("s", fontsize=x_y_label_fontsize)
        plt.ylabel("RNR(s)", fontsize=x_y_label_fontsize)
        if baseline_name_index == 0:
            plt.title(label=dataset_name[dataset_name_index] + '\n', fontsize=title_fontsize, fontweight=fontweight)

        # if subplot_index in [1, 5, 9, 21, 25]:
        #     ax = plt.gca()
        #     ax.text(0, 9, 'hahahhah')
        #     baseline_name_count=baseline_name_count+1

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
    pass
pass



plt.subplots_adjust(wspace =wspace, hspace =wspace)#调整子图间距
plt.savefig('./figures//' + 'RNR(s).eps', format='eps',
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了

plt.show()
time_end = time.time()

print("It takes: " + str((time_end - time_start)/ 60)  + " mins.")





