import xlwt
import xlrd
from xlutils.copy import copy
import os
from scipy.special import comb, perm
import matplotlib.pyplot as plt
import numpy as np

source_dir='D:\hybridrec//results//'

datasets=['facebook_combined', 'ca-hepph',
          'yeast', 'email-eucore'] # petster-friendships-hamster
methods_heuristic = ['cn', 'ja', 'aa', 'cosine', 'pearson', 'degreeproduct']
methods_embedding = ['drne', 'prone', 'deepwalk', 'node2vec']  # 'graph2gauss', 'attentionwalk', 'splitter'



save_xls_file='D:\hybridrec\临时文件夹//embedding_AP.xls'
if not os.path.isfile(save_xls_file):
    xls = xlwt.Workbook()
    sht1 = xls.add_sheet('Sheet1', cell_overwrite_ok=True)
    xls.save(save_xls_file)

# 打开要写入的文件
write_f = xlrd.open_workbook(save_xls_file, formatting_info=True)
f = copy(write_f)
table_write = f.get_sheet(0)

start_row=1 # 从excel表格的第2行开始保存数据

for datasets_i in range(len(datasets)):
    for method_i in range(len(methods_embedding)):
        for method_j in range(method_i+1, len(methods_embedding)):
            method1=methods_embedding[method_i]
            method2=methods_embedding[method_j]
            excel_name=datasets[datasets_i] + '--' + method1 + '--' + method2
            excel_path=source_dir + excel_name + '.xls'


            # 打开读入的文件
            read_f = xlrd.open_workbook(excel_path, formatting_info=True)
            table_read = read_f.sheet_by_index(0)

            start_col=1+datasets_i # dataset_i*2或dataset_i*1


            # 写single baseline的结果

            table_write.write(start_row+method_i, # 写列名
                             0,
                             methods_embedding[method_i], style=xlwt.XFStyle())
            val= table_read.cell(2+12, 7-1).value
            table_write.write(start_row+method_i,
                            start_col,
                            val, style=xlwt.XFStyle())

            table_write.write(start_row+method_j,
                            0,
                            methods_embedding[method_j], style=xlwt.XFStyle())

            val= table_read.cell(3+12, 7-1).value
            table_write.write(start_row+method_j,
                            start_col,
                            val, style=xlwt.XFStyle())



            # 写MLP的结果
            combine_index=1/2 * (2*len(methods_embedding) -method_i -1)*(method_i+1-1) + (method_j-method_i)
            combine_index=int(combine_index-1)

            table_write.write(start_row+len(methods_embedding)+combine_index,
                                0,
                                'MLP-('+methods_embedding[method_i] + '+' + methods_embedding[method_j] + ')',
                                style=xlwt.XFStyle())
            val= table_read.cell(21+3, 7-1).value
            table_write.write(start_row+len(methods_embedding)+combine_index,
                            start_col,
                            val, style=xlwt.XFStyle())
            pass


            # 写PNR的结果
            combine_index=1/2 * (2*len(methods_embedding) -method_i -1)*(method_i+1-1) + (method_j-method_i)
            combine_index=int(   combine_index-1 + comb(len(methods_embedding), 2)   )

            table_write.write(start_row + len(methods_embedding)+combine_index,
                            0,
                            'PNR-('+methods_embedding[method_i] + '+' + methods_embedding[method_j] + ')',
                             style=xlwt.XFStyle())

            val= table_read.cell(1+12, 7 - 1).value
            table_write.write(start_row + len(methods_embedding)+combine_index,
                            start_col,
                            val, style=xlwt.XFStyle())


        pass
    pass
pass

os.remove(save_xls_file)
f.save(save_xls_file)

# 画图
# plt.style.use("ggplot")
plt.style.use('seaborn-darkgrid')
# sns.set_style('whitegrid')
hatch1="\\\\"
hatch2='//'
bar_width=0.2
alpha=1.0
xlabel='fused baselines'
ylabel='AP'
ticks_fontsize=13
legend_fontsize=12
x_y_label_fontsize=12
title_fontsize=14
hspace=1.0
wspace=0.2

data = xlrd.open_workbook(save_xls_file)
table = data.sheets()[0]
fig = plt.figure(figsize=(10, 9)) # figsize=(17, 3)或1

for datasets_index in range(1, len(datasets)+1):
    ax = fig.add_subplot(2, 2, datasets_index)

    baseline_data = []
    x_data = []
    y_data1 = []
    y_data2 = []
    n = len(methods_embedding)
    rotation = 90  # 改这里

    for i in range(len(methods_embedding)):
        for j in range(i + 1, len(methods_embedding)):
            val = methods_embedding[i] + '+' + methods_embedding[j]
            x_data.append(val)
        pass
    pass

    for i in range(len(methods_embedding)):
        val = table.cell(start_row + i,
                         datasets_index).value
        baseline_data.append(val)

    for y_data_index in range(int(n * (n - 1) / 2)):
        val = table.cell(start_row + n + y_data_index,
                         datasets_index).value
        y_data1.append(val)
    pass
    for y_data_index in range(int(n * (n - 1) / 2)):
        val = table.cell(start_row + n + int(n * (n - 1) / 2) + y_data_index,
                         datasets_index).value
        y_data2.append(val)
    pass

    plt.bar(np.arange(len(x_data)), y_data1, label='MLP', width=bar_width,
            color='steelblue', alpha=alpha, hatch=hatch1)
    plt.bar(np.arange(len(x_data)) + bar_width, y_data2, label='PNR', width=bar_width,
            color='indianred', alpha=alpha, hatch=hatch2)

    plt.xlabel(xlabel, color='black', fontsize=x_y_label_fontsize)
    plt.ylabel(ylabel, color='black', fontsize=x_y_label_fontsize)
    plt.xticks(np.arange(len(x_data)), x_data,
               color='black', rotation=rotation,
               fontsize=ticks_fontsize)
    plt.yticks(color='black', fontsize=ticks_fontsize)
    plt.title(datasets[datasets_index - 1], fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize)
    # if datasets[datasets_index - 1] == 'facebook_combined':
    #     plt.ylim(0.03, 0.59)
    #     my_y_ticks = np.arange(0.04, 0.59, 0.1)
    #     plt.yticks(my_y_ticks)
    # if datasets[datasets_index - 1] == 'ca-hepph':
    #     plt.ylim(0.25, 0.88)
    #     my_y_ticks = np.arange(0.25, 0.88, 0.1)
    #     plt.yticks(my_y_ticks)
    # if datasets[datasets_index - 1] == 'yeast':
    #     plt.ylim(0.05, 0.40)
    #     my_y_ticks = np.arange(0.05, 0.40, 0.05)
    #     plt.yticks(my_y_ticks)
    # if datasets[datasets_index - 1] == 'email-eucore':
    #     plt.ylim(0.09, 0.29)
    #     my_y_ticks = np.arange(0.09, 0.29, 0.05)
    #     plt.yticks(my_y_ticks)

    colors = []
    offset = 0.5
    # for i in range(len(methods_embedding)):
    #     plt.annotate(methods_embedding[i],
    #                  xy=(int(n * (n - 1) / 2), baseline_data[i]),
    #                  xytext=(int(n * (n - 1) / 2) + offset, baseline_data[i])
    #                  )  # , arrowprops=dict(facecolor='red', shrink=0.1, width=2)
    #     plt.hlines(baseline_data[i],
    #                0 - offset,
    #                int(n * (n - 1) / 2) + offset,
    #                linestyles='--', colors='black')
    # pass




plt.subplots_adjust(wspace=wspace, hspace=hspace)#调整子图间距
plt.savefig('D:\hybridrec\临时文件夹//' +'embedding_AP.eps', format='eps',
            bbox_inches='tight') # pad_inches使得四周的空白都去掉了
plt.show()





