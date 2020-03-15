import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy.io import loadmat
import seaborn as sns
import time

time_start = time.time()
dataset_name=['ML100K', 'Delicious', 'LastFM', 'Wikibooks']



# sns.set_style('whitegrid')
subplot_index = 1
hspace=0.28
wspace=0.28
ticks_fontsize=16
legend_fontsize=16
x_y_label_fontsize=16
title_fontsize=20
filter_color=['tomato', 'teal', 'dodgerblue', 'darkorange', 'seagreen', 'magenta', 'crimson']
linewidth=1.5
bwith=2
fontweight='semibold'
lengends=['org', 'b=5', 'b=10', 'b=20', 'b=40', 'b=60', 'b=100', 'b=150', 'b=200']
right_dis=-0.52
upper_dis=0.9

fig=plt.figure(figsize=(21, 26.7)) # 21, 29.7
fig.text(0, 0.147, 'JCA', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0, 0.267, 'APR', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0, 0.375, 'MultiDAE', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0, 0.485, 'FISM', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0, 0.60, 'PureSVD', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0, 0.72, 'UserCF', fontsize=title_fontsize, fontweight=fontweight)
fig.text(0, 0.825, 'ItemCF', fontsize=title_fontsize, fontweight=fontweight)
topL_array = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]
for baseline_name_index in range(7):
    for dataset_name_index in range(len(dataset_name)):
        base_path = 'D:\链路预测相关课题\matlab评估代码\实验结果\paraSensitivity//'
        excel_path = base_path + dataset_name[dataset_name_index] + '.xls'
        data = xlrd.open_workbook(excel_path)
        table = data.sheets()[0]

        y_data = np.zeros(shape=(9, 12))
        row_index = baseline_name_index * 9
        for i in range(9):
            for j in range(12):
                y_data[i, j] = float(table.cell(row_index + i, j).value)
            pass
        pass


        markers =['o', '*', '^', '1', 'd', 'x', 's', 'p', 'h']
        plt.subplot(7, len(dataset_name), subplot_index)
        for k in range(9):
            if k == 0:
                plt.plot(topL_array, y_data[k, :],
                         marker=markers[k], linestyle='--', color='black', label=lengends[k],
                         linewidth=linewidth)
                continue
            plt.plot(topL_array, y_data[k, :],
                 marker=markers[k],
                 linewidth=linewidth,
                 label=lengends[k])
            pass
        pass

        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.xlabel("N", fontsize=x_y_label_fontsize)
        plt.ylabel("ARHR@N", fontsize=x_y_label_fontsize)
        if baseline_name_index == 0:
            plt.title(label=dataset_name[dataset_name_index] + '\n', fontsize=title_fontsize, fontweight=fontweight)

        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 2))
        subplot_index = subplot_index + 1


        # 显示legend
        if (dataset_name_index == 1) & (baseline_name_index == 6):
            plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=len(lengends), fontsize=x_y_label_fontsize)
    pass
pass



plt.subplots_adjust(wspace =wspace, hspace =wspace)#调整子图间距
plt.savefig('./figures//' + 'paraSen.eps', format='eps', dpi=600,
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了

plt.show()
time_end = time.time()

print("It takes: " + str((time_end - time_start)/ 60)  + " mins.")





