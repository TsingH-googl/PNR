import matplotlib.pyplot as plt
import numpy as np
import xlrd


ticks_fontsize=15
legend_fontsize=10
x_y_label_fontsize=12
title_fontsize=14

# lastfm N=20
y_data_map = [0.095222935,0.105873801,0.092289441,
              0.00368374,0.096274328,0.09575723,0.090838743]
y_data2_map = [0.099719303,0.109018681,0.0903588,0.063240027,
               0.10794889,0.10287687,0.095756176]


# # ml100k N=10
# y_data_arhr = [0.05379,0.06525,0.05685,0.03951,0.05877,0.06293,0.06307]
# y_data2_arhr = [0.05719,0.06623,0.05926,0.0459,0.05789,0.06431,0.06286]

# lastfm N=20
y_data_arhr = [0.03856,0.0254,0.01584,0.02128,0.04378,0.04227,0.0347]
y_data2_arhr = [0.04147,0.02616,0.01733,0.02399,0.04573,0.04355,0.03536]



fig = plt.figure(figsize=(10, 2.8)) # figsize=(17, 3)或1
fig.text(0.42, 0.78, '(a)', fontsize=title_fontsize, fontweight='semibold')
fig.text(0.855, 0.78, '(b)', fontsize=title_fontsize, fontweight='semibold')

plt.style.use("ggplot")
layout1=1
layout2=2
bar_width=0.3
alpha=1.0
hspace=0.2
wspace=0.30
hatch1="\\\\"
hatch2='//'
right_dis=1.0
upper_dis=1.16
ncol=2


ax1 = fig.add_subplot(layout1,layout2,1)

x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.barh(y=range(len(x_data)), width=y_data_map, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2_map,
    label='RNR methods', color='indianred', alpha=alpha, height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
plt.yticks(np.arange(len(x_data))+bar_width/2, x_data, color='black')

plt.xlabel('MAP@20', color='black', fontsize=x_y_label_fontsize)
plt.xticks(color='black')


# plt.xlim(0.03, 0.069)
# my_x_ticks = np.arange(0.03, 0.069, 0.01)
# plt.xticks(my_x_ticks)
plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol, fontsize=legend_fontsize)


ax2 = fig.add_subplot(layout1,layout2, 2)

x_data = ['ItemCF', 'UserCF', 'PureSVD', 'FISM', 'MultiDAE', 'APR', 'JCA']
# hatch 关键字可用来设置填充样式，可取值为： / , \ , | , - , + , x , o , O , . , * 。
plt.barh(y=range(len(x_data)), width=y_data_arhr, label='original methods',
    color='steelblue', alpha=alpha, height=bar_width, hatch=hatch1)
plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2_arhr,
    label='RNR methods', color='indianred', alpha=alpha, height=bar_width, hatch=hatch2)

# 为Y轴设置刻度值
plt.yticks(np.arange(len(x_data))+bar_width/2, x_data, color='black')

plt.xlabel('ARHR@20', color='black', fontsize=x_y_label_fontsize)
plt.xticks(color='black')


plt.xlim(0.015, 0.05)
my_x_ticks = np.arange(0.015, 0.05, 0.01)
plt.xticks(my_x_ticks)
# plt.legend(loc='lower center', bbox_to_anchor=(upper_dis, right_dis), ncol=ncol,
#            fontsize=legend_fontsize)


plt.subplots_adjust(wspace=wspace, hspace=wspace)#调整子图间距
plt.savefig('./figures//' +'MAP_ARHR.pdf', format='pdf',
            bbox_inches='tight',
            pad_inches=0) # pad_inches使得四周的空白都去掉了
plt.show()