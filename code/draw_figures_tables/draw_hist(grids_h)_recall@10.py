import matplotlib.pyplot as plt
import numpy as np
import xlrd


# topL: -1 0 1 2 3 4 5 6
#       5 10 15 20 25 30 40 50
topL=0
xlabel="Recall@10"
fig = plt.figure(figsize=(9, 10)) # figsize=(17, 3)或1
title_fontsize = 17
fig.text(0.40, 0.79, '(a)', fontsize=title_fontsize, fontweight='semibold')
fig.text(0.80, 0.79, '(b)', fontsize=title_fontsize, fontweight='semibold')
fig.text(0.40, 0.34, '(c)', fontsize=title_fontsize, fontweight='semibold')
fig.text(0.80, 0.34, '(d)', fontsize=title_fontsize, fontweight='semibold')


plt.style.use("ggplot")
# plt.rcParams['font.sans-serif'] = ['Times new romans']
layout1=2
layout2=2
add_baselines=False
bar_width=0.3
alpha=1.0
hspace=0.8
wspace=0.4
x_label1=True
x_label2=True
x_label3=True
x_label4=True
# bbox_to_anchor=(-0.5, -0.3): 右边、上边的距离来调整legend
right_dis=1.0 # 调整上下
upper_dis=0.0 # 调整左右
ncol=1
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
hatch1="\\\\"
hatch2='//'
fontdict={'color': 'black'}
rotation=30


ax1 = fig.add_subplot(layout1, layout2,1)
xls_file_path  = 'D:\链路预测相关课题\matlab评估代码\实验结果\加了实验后的画图\KDD\Table//recall4.xlsx'
data = xlrd.open_workbook(xls_file_path)
table = data.sheets()[0]
# ML100K: 2/4
# Delicious: 2/15
# Lastfm: 18/4
# Wikibooks：18/15
row_start = 2 # 改这里
col_start = 4 + topL # 改这里
plt.title("ML100K", fontsize=title_fontsize) # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.bar(np.arange(len(x_data)), y_data, label='original methods', width=0.35,
         color='steelblue', alpha=alpha, hatch=hatch1)
plt.bar(np.arange(len(x_data)) + bar_width, y_data2, label='RNR methods', width=0.35,
         color='indianred', alpha=alpha, hatch=hatch2)


# 为Y轴设置刻度值
if x_label1:
   plt.xticks(np.arange(len(x_data))+bar_width/2, x_data)
else:
    ax1.set_xticklabels([], fontsize='small')
plt.ylabel(xlabel, fontdict=fontdict)
plt.yticks(color='black')
if add_baselines:
    plt.xlabel("Baselines", fontdict=fontdict)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)
plt.xticks(color='black', rotation=rotation)

plt.ylim(0.038, 0.069)
my_y_ticks = np.arange(0.038, 0.069, 0.01)
plt.yticks(my_y_ticks)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)




ax2 = fig.add_subplot(layout1,layout2,2)
# ML100K: 2/4
# Delicious: 2/15
# Lastfm: 18/4
# Wikibooks：18/15
row_start = 2 # 改这里
col_start = 15 + topL # 改这里
plt.title("Delicious", fontsize=title_fontsize) # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.bar(np.arange(len(x_data)), y_data, label='original methods', width=0.35,
         color='steelblue', alpha=alpha, hatch=hatch1)
plt.bar(np.arange(len(x_data)) + bar_width, y_data2, label='RNR methods', width=0.35,
         color='indianred', alpha=alpha, hatch=hatch2)


# 为Y轴设置刻度值
if x_label2:
   plt.xticks(np.arange(len(x_data))+bar_width/2, x_data)
else:
    ax2.set_xticklabels([], fontsize='small')
plt.ylabel(xlabel, fontdict=fontdict)
plt.yticks(color='black')
if add_baselines:
    plt.xlabel("Baselines", fontdict=fontdict)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)
plt.xticks(color='black', rotation=rotation)


# plt.ylim(0.0, 0.24)
# my_y_ticks = np.arange(0.0, 0.24, 0.05)
# plt.yticks(my_y_ticks)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)
plt.xticks(color='black', rotation=rotation)


ax3 = fig.add_subplot(layout1,layout2,3)
# ML100K: 2/4
# Delicious: 2/15
# Lastfm: 18/4
# Wikibooks：18/15
row_start = 18 # 改这里
col_start = 4 + topL# 改这里
plt.title("Lastfm", fontsize=title_fontsize) # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.bar(np.arange(len(x_data)), y_data, label='original methods', width=0.35,
         color='steelblue', alpha=alpha, hatch=hatch1)
plt.bar(np.arange(len(x_data)) + bar_width, y_data2, label='RNR methods', width=0.35,
         color='indianred', alpha=alpha, hatch=hatch2)


# 为Y轴设置刻度值
if x_label3:
   plt.xticks(np.arange(len(x_data))+bar_width/2, x_data)
else:
    ax3.set_xticklabels([], fontsize='small')
plt.ylabel(xlabel, fontdict=fontdict)
plt.yticks(color='black')
if add_baselines:
    plt.xlabel("Baselines", fontdict=fontdict)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)
plt.xticks(color='black', rotation=rotation)

plt.ylim(0.0, 0.04)
my_y_ticks = np.arange(0.0, 0.04, 0.01)
plt.yticks(my_y_ticks)





ax4 = fig.add_subplot(layout1,layout2,4)
# ML100K: 2/4
# Delicious: 2/15
# Lastfm: 18/4
# Wikibooks：18/15
row_start = 18 # 改这里
col_start = 15 + topL # 改这里
plt.title("Wikibooks", fontsize=title_fontsize) # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.bar(np.arange(len(x_data)), y_data, label='original methods', width=0.35,
         color='steelblue', alpha=alpha, hatch=hatch1)
plt.bar(np.arange(len(x_data)) + bar_width, y_data2, label='RNR methods', width=0.35,
         color='indianred', alpha=alpha, hatch=hatch2)


# 为Y轴设置刻度值
if x_label4:
   plt.xticks(np.arange(len(x_data))+bar_width/2, x_data)
else:
    ax4.set_xticklabels([], fontsize='small')
plt.ylabel(xlabel, fontdict=fontdict)
plt.yticks(color='black')
if add_baselines:
    plt.xlabel("Baselines", fontdict=fontdict)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)
plt.xticks(color='black', rotation=rotation)

plt.ylim(0.085, 0.14)
my_y_ticks = np.arange(0.085, 0.14, 0.02)
plt.yticks(my_y_ticks)

# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)


plt.subplots_adjust(wspace=wspace, hspace=wspace)#调整子图间距
plt.savefig('./figures//' + xlabel +'.eps', format='eps',
            bbox_inches='tight') # pad_inches使得四周的空白都去掉了
plt.show()
