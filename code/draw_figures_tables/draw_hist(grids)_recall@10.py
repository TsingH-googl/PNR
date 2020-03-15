import matplotlib.pyplot as plt
import numpy as np
import xlrd


# topL: -1 0 1 2 3 4 5 6
#       5 10 15 20 25 30 40 50
topL=0
xlabel="Recall@10"
fig = plt.figure(figsize=(12, 9)) # figsize=(17, 3)或1
title_fontsize = 17
fig.text(0.41, 0.85, '(a)', fontsize=title_fontsize, fontweight='semibold')
fig.text(0.86, 0.85, '(b)', fontsize=title_fontsize, fontweight='semibold')
fig.text(0.41, 0.40, '(c)', fontsize=title_fontsize, fontweight='semibold')
fig.text(0.86, 0.40, '(d)', fontsize=title_fontsize, fontweight='semibold')


plt.style.use("ggplot")
# plt.rcParams['font.sans-serif'] = ['Times new romans']
layout1=2
layout2=2
add_baselines=False
bar_width=0.3
alpha=1.0
hspace=0.9
wspace=0.4
y_label1=True
y_label2=True
y_label3=True
y_label4=True
# bbox_to_anchor=(-0.5, -0.3): 右边、上边的距离来调整legend
right_dis=-0.33 # 调整上下
upper_dis=0.1 # 调整左右
ncol=1
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
hatch1="\\\\"
hatch2='//'
ticks_fontsize=13
legend_fontsize=12
x_y_label_fontsize=12
title_fontsize=14


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
plt.title("ML100K", fontsize=title_fontsize) # 改这里
y_data = []
y_data2 = []
for i in range(7):
    y_data.append(float(table.cell(row_start + i * 2, col_start).value))
    y_data2.append(float(table.cell(row_start + i * 2 + 1, col_start).value))
    pass


x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.barh(y=range(len(x_data)), width=y_data, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, lw=1,hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
    label='RNR methods', color='indianred', alpha=alpha, lw=1,height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
if y_label1:
   plt.yticks(np.arange(len(x_data))+bar_width/2, x_data, color='black', fontsize=ticks_fontsize)
else:
    plt.set_yticks([])
plt.xlabel(xlabel, color='black')
plt.xticks(color='black')
if add_baselines:
    plt.ylabel("Baselines", color='black')

plt.xlim(0.038, 0.069)
my_x_ticks = np.arange(0.038, 0.069, 0.01)
plt.xticks(my_x_ticks, color='black', fontsize=ticks_fontsize)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol, fontsize=legend_fontsize)




ax2 = fig.add_subplot(layout1,layout2,2)
ax2.set_yticklabels([], fontsize='small')
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
plt.barh(y=range(len(x_data)), width=y_data, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, lw=1, hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
    label='RNR methods', color='indianred', alpha=alpha,lw=1, height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
if y_label2:
   plt.yticks(np.arange(len(x_data))+bar_width/2, x_data, color='black', fontsize=ticks_fontsize)
else:
    ax2.set_yticklabels([], fontsize='small')
plt.xlabel(xlabel, color='black')
plt.xticks(color='black', fontsize=ticks_fontsize)
if add_baselines:
    plt.ylabel("Baselines",  color='black')
plt.xlim(0.0, 0.24)
my_x_ticks = np.arange(0.0, 0.24, 0.05)
plt.xticks(my_x_ticks)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol, fontsize=legend_fontsize)



ax3 = fig.add_subplot(layout1, layout2, 3)
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
plt.barh(y=range(len(x_data)), width=y_data, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, lw=1,hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
    label='RNR methods', color='indianred', alpha=alpha,lw=1, height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
if y_label3:
   plt.yticks(np.arange(len(x_data))+bar_width/2, x_data, color='black', fontsize=ticks_fontsize)
else:
    ax3.set_yticklabels([], fontsize='small')
plt.xlabel(xlabel,  color='black')
plt.xticks(color='black', fontsize=ticks_fontsize)
if add_baselines:
    plt.ylabel("Baselines",  color='black')
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol,  fontsize=legend_fontsize)


plt.xlim(0.0, 0.04)
my_x_ticks = np.arange(0.0, 0.04, 0.01)
plt.xticks(my_x_ticks)





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
plt.barh(y=range(len(x_data)), width=y_data, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, lw=1,hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2,
    label='RNR methods', color='indianred', alpha=alpha, lw=1,height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
if y_label4:
   plt.yticks(np.arange(len(x_data))+bar_width/2, x_data, color='black', fontsize=ticks_fontsize)
else:
    ax4.set_yticklabels([], fontsize='small')
plt.xlabel(xlabel,  color='black')
plt.xticks(color='black', fontsize=ticks_fontsize)
if add_baselines:
    plt.ylabel("Baselines",  color='black')
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol
           ,fontsize=legend_fontsize)


plt.xlim(0.085, 0.14)
my_x_ticks = np.arange(0.085, 0.14, 0.02)
plt.xticks(my_x_ticks)

# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)


plt.subplots_adjust(wspace=wspace, hspace=wspace)#调整子图间距
plt.savefig('./figures//' + xlabel +'.pdf', format='pdf',
            bbox_inches='tight') # pad_inches使得四周的空白都去掉了
plt.show()
