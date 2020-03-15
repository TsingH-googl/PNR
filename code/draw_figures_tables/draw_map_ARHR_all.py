import matplotlib.pyplot as plt
import numpy as np
import xlrd



arhr_xls_file_path  = 'D:\链路预测相关课题\matlab评估代码\实验结果//ARHR.xlsx'
map_xls_file_path  = 'D:\链路预测相关课题\matlab评估代码\实验结果//MAP.xls'
arhr_data = xlrd.open_workbook(arhr_xls_file_path)
table_arhr = arhr_data.sheets()[0]
map_data = xlrd.open_workbook(map_xls_file_path)
table_map = map_data.sheets()[0]

# plt.style.use("ggplot")
# plt.style.use('seaborn-darkgrid')
# sns.set_style('whitegrid')
ticks_fontsize=15
legend_fontsize=10
x_y_label_fontsize=12
title_fontsize=14

fig = plt.figure(figsize=(9, 14)) # 21, 29.7
# fig.text(0.42, 0.78, '(a)', fontsize=title_fontsize, fontweight='semibold')
# fig.text(0.855, 0.78, '(b)', fontsize=title_fontsize, fontweight='semibold')

plt.style.use("ggplot")
layout1=4
layout2=2
bar_width=0.3
alpha=1.0
hspace=0.43
wspace=0.33
hatch1="\\\\"
hatch2='//'
right_dis=-0.40
upper_dis=0.07
ncol=1


topL_index = 3
topL_array = [5, 10, 15, 20]
x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
dataset_names = ['ML100K', 'Delicious', 'LastFM', 'Wikibooks']

subplot_index=1
for dataset_name_index in range (len(dataset_names)):
    org_data_map=[]
    rnr_data_map = []
    org_data_arhr=[]
    rnr_data_arhr=[]
    for k in  range(7):
        val_org_map = float(table_map.cell(1 + 2*k + dataset_name_index*15, 2 + topL_index).value)
        val_rnr_map = float(table_map.cell(2 + 2 * k + dataset_name_index * 15, 2 + topL_index).value)

        val_org_arhr = float(table_arhr.cell(1 + 2*k + dataset_name_index*15, 2 + topL_index).value)
        val_rnr_arhr = float(table_arhr.cell(2 + 2 * k + dataset_name_index * 15, 2 + topL_index).value)

        org_data_map.append(val_org_map)
        rnr_data_map.append(val_rnr_map)
        org_data_arhr.append(val_org_arhr)
        rnr_data_arhr.append(val_rnr_arhr)
    pass

    ax_map = fig.add_subplot(layout1, layout2, subplot_index)
    plt.barh(y=range(len(x_data)), width=org_data_map, label='original methods',
             color='steelblue', alpha=alpha, height=bar_width, hatch=hatch1)
    plt.barh(y=np.arange(len(x_data)) + bar_width, width=rnr_data_map,
             label='RNR methods', color='indianred', alpha=alpha, height=bar_width, hatch=hatch2)

    # 为Y轴设置刻度值
    plt.yticks(np.arange(len(x_data)) + bar_width / 2, x_data, color='black')
    plt.xlabel('MAP@' + str(topL_array[topL_index]), color='black', fontsize=x_y_label_fontsize)
    plt.xticks(color='black')
    plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol, fontsize=legend_fontsize)
    # plt.xlim(0.03, 0.069)
    # my_x_ticks = np.arange(0.03, 0.069, 0.01)
    # plt.xticks(my_x_ticks)


    ax_arhr = fig.add_subplot(layout1, layout2, subplot_index + 1)
    plt.barh(y=range(len(x_data)), width=org_data_arhr, label='original methods',
             color='steelblue', alpha=alpha, height=bar_width, hatch=hatch1)
    plt.barh(y=np.arange(len(x_data)) + bar_width, width=rnr_data_arhr,
             label='RNR methods', color='indianred', alpha=alpha, height=bar_width, hatch=hatch2)

    # 为Y轴设置刻度值
    plt.yticks(np.arange(len(x_data)) + bar_width / 2, x_data, color='black')
    plt.xlabel('ARHR@' + str(topL_array[topL_index]), color='black', fontsize=x_y_label_fontsize)
    plt.xticks(color='black')
    plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol, fontsize=legend_fontsize)
    # plt.xlim(0.03, 0.069)
    # my_x_ticks = np.arange(0.03, 0.069, 0.01)
    # plt.xticks(my_x_ticks)





    subplot_index = subplot_index + 2
pass



plt.subplots_adjust(wspace=wspace, hspace=hspace)#调整子图间距
plt.savefig('./figures//' +'MAP_ARHR.pdf', format='pdf',
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了
plt.show()