import matplotlib.pyplot as plt
import numpy as np
import xlrd


# topL: -1 0 1 2 3 4 5 6
#       5 10 15 20 25 30 40 50
topL=5
xlabel="Recall@40"
fig = plt.figure(figsize=(17, 3)) # figsize=(17, 3)或1
plt.style.use("ggplot")
# plt.rcParams['font.sans-serif'] = ['Times new romans']
layout1=1
layout2=4
add_baselines=False
bar_width=0.3
alpha=1.0
hspace=0
wspace=0.08
y_label1=True
y_label2=False
y_label3=False
y_label4=False
# bbox_to_anchor=(-0.5, -0.3): 右边、上边的距离来调整legend
right_dis=-0.5
upper_dis=0.42
ncol=1
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
hatch1="\\\\"
hatch2='//'
fontdict={'color': 'black'}



ax1 = fig.add_subplot(layout1,layout2,1)
xls_file_path  = 'D:\链路预测相关课题\matlab评估代码\实验结果\加了实验后的画图\KDD\Table//recall4.xlsx'
data = xlrd.open_workbook(xls_file_path)
table = data.sheets()[0]
# ML100K: 2/4
# Delicious: 2/15
# Lastfm: 18/4
# Wikibooks：18/15
row_start = 2 # 改这里
col_start = 4 + topL # 改这里
plt.title("ML100K") # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.barh(y=range(len(x_data)), width=y_data, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
    label='RNR methods', color='indianred', alpha=alpha, height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
if y_label1:
   plt.yticks(np.arange(len(x_data))+bar_width/2, x_data, color='black')
else:
    plt.set_yticks([])
plt.xlabel(xlabel, fontdict=fontdict)
plt.xticks(color='black')
if add_baselines:
    plt.ylabel("Baselines", fontdict=fontdict)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)




ax2 = fig.add_subplot(layout1,layout2,2)
# ML100K: 2/4
# Delicious: 2/15
# Lastfm: 18/4
# Wikibooks：18/15
row_start = 2 # 改这里
col_start = 15 + topL # 改这里
plt.title("Delicious") # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.barh(y=range(len(x_data)), width=y_data, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
    label='RNR methods', color='indianred', alpha=alpha, height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
if y_label2:
   plt.yticks(np.arange(len(x_data))+bar_width/2, x_data)
else:
    ax2.set_yticks([])
plt.xlabel(xlabel, fontdict=fontdict)
plt.xticks(color='black')
if add_baselines:
    plt.ylabel("Baselines", fontdict=fontdict)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)


ax3 = fig.add_subplot(layout1,layout2,3)
# ML100K: 2/4
# Delicious: 2/15
# Lastfm: 18/4
# Wikibooks：18/15
row_start = 18 # 改这里
col_start = 4 + topL# 改这里
plt.title("Lastfm") # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.barh(y=range(len(x_data)), width=y_data, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
    label='RNR methods', color='indianred', alpha=alpha, height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
if y_label3:
   plt.yticks(np.arange(len(x_data))+bar_width/2, x_data)
else:
    ax3.set_yticks([])
plt.xlabel(xlabel, fontdict=fontdict)
plt.xticks(color='black')
if add_baselines:
    plt.ylabel("Baselines", fontdict=fontdict)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)





ax4 = fig.add_subplot(layout1,layout2,4)
# ML100K: 2/4
# Delicious: 2/15
# Lastfm: 18/4
# Wikibooks：18/15
row_start = 18 # 改这里
col_start = 15 + topL # 改这里
plt.title("Wikibooks") # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.barh(y=range(len(x_data)), width=y_data, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
    label='RNR methods', color='indianred', alpha=alpha, height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
if y_label4:
   plt.yticks(np.arange(len(x_data))+bar_width/2, x_data)
else:
    ax4.set_yticks([])
plt.xlabel(xlabel, fontdict=fontdict)
plt.xticks(color='black')
if add_baselines:
    plt.ylabel("Baselines", fontdict=fontdict)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol)


# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)


plt.subplots_adjust(wspace =wspace, hspace =wspace)#调整子图间距
plt.savefig('./figures//' + 'recall.png', format='png', dpi=1000,
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了
plt.show()
