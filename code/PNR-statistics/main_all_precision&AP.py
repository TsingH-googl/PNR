import xlwt
import xlrd
from xlutils.copy import copy
import os
from scipy.special import comb, perm
import matplotlib.pyplot as plt
import numpy as np


source_dir='D:\hybridrec//results//'
datasets=['facebook_combined', 'ca-hepph', 'yeast', 'email-eucore']
methods= ['cn', 'ja', 'aa', 'pearson', 'deepwalk', 'node2vec', 'drne', 'prone']
save_xls_file='D:\hybridrec\临时文件夹//all_precision_ap_incre.xls'



ds1_result=np.zeros(shape=(len(methods), len(methods)), dtype=float)
ds2_result=np.zeros(shape=(len(methods), len(methods)), dtype=float)
ds3_result=np.zeros(shape=(len(methods), len(methods)), dtype=float)
ds4_result=np.zeros(shape=(len(methods), len(methods)), dtype=float)

for dataset_index in range(len(datasets)):
    temp_array_precision = np.zeros(shape=(len(methods), len(methods)), dtype=float)
    temp_array_ap = np.zeros(shape=(len(methods), len(methods)), dtype=float)
    for method_i in range(len(methods)):
        for method_j in range(method_i+1, len(methods)):
            # 获得excel的句柄
            method1 = methods[method_i]
            method2 = methods[method_j]
            excel_name1 = datasets[dataset_index] + '--' + method1 + '--' + method2
            excel_name2 = datasets[dataset_index] + '--' + method2 + '--' + method1
            excel_path1 = source_dir + excel_name1 + '.xls'
            excel_path2 = source_dir + excel_name2 + '.xls'
            if os.path.isfile(excel_path1):
                read_f = xlrd.open_workbook(excel_path1, formatting_info=True)
                table_read = read_f.sheet_by_index(0)
            else:
                read_f = xlrd.open_workbook(excel_path2, formatting_info=True)
                table_read = read_f.sheet_by_index(0)
            pass

            # 获取precision
            method1_precision = table_read.cell(2, 6).value
            method2_precision = table_read.cell(3, 6).value
            # plus_precision = table_read.cell(4, 6).value
            # multiply_precision = table_read.cell(49, 6).value
            # MLP_precision = table_read.cell(21, 6).value
            PNR_precision = table_read.cell(1, 6).value

            # 获取AP
            method1_AP = table_read.cell(14, 6).value
            method2_AP = table_read.cell(15, 6).value
            # plus_AP = table_read.cell(16, 6).value
            # multiply_AP = table_read.cell(52, 6).value
            # MLP_AP = table_read.cell(24, 6).value
            PNR_AP = table_read.cell(13, 6).value

            # 计算precision的增长百分比
            temp_max_precision=max(method1_precision, method2_precision)
            temp_array_precision[method_i, method_j] = round(100.0*((PNR_precision-temp_max_precision) / temp_max_precision), 2)

            # 计算AP的增长百分比
            temp_max_ap = max(method1_AP, method2_AP)
            temp_array_ap[method_i, method_j] = round(100.0*((PNR_AP - temp_max_ap) / temp_max_ap), 2)
        pass # method_j
    pass # method_i

    if dataset_index ==0:
        ds1_result=temp_array_precision + temp_array_ap.T
    elif dataset_index ==1:
        ds2_result = temp_array_precision + temp_array_ap.T
    elif dataset_index ==2:
        ds3_result = temp_array_precision + temp_array_ap.T
    elif dataset_index ==3:
        ds4_result = temp_array_precision + temp_array_ap.T

pass #dataset_index

# 把ds1_result到ds4_result写入xls
if not os.path.isfile(save_xls_file):
    xls = xlwt.Workbook()
    sht1 = xls.add_sheet('Sheet1', cell_overwrite_ok=True)
    xls.save(save_xls_file)
write_f = xlrd.open_workbook(save_xls_file, formatting_info=True)
f = copy(write_f)
table_write = f.get_sheet(0)

start_row=2
start_col=1
offset=len(methods) + 2
# 先写标题
for i in range(len(methods)):
    # 行
    table_write.write(start_row -1, start_col + i, methods[i], style=xlwt.XFStyle())
    table_write.write(start_row -1, start_col + offset + i,  methods[i], style=xlwt.XFStyle())
    table_write.write(start_row -1 + offset, start_col + i, methods[i], style=xlwt.XFStyle())
    table_write.write(start_row -1 + offset, start_col + offset + i, methods[i], style=xlwt.XFStyle())

    # 列
    table_write.write(start_row + i , start_col - 1, methods[i], style=xlwt.XFStyle())
    table_write.write(start_row + offset + i, start_col-1,  methods[i], style=xlwt.XFStyle())
    table_write.write(start_row + i, start_col-1 + offset, methods[i], style=xlwt.XFStyle())
    table_write.write(start_row + offset + i, start_col -1 + offset, methods[i], style=xlwt.XFStyle())

pass

#再写数据
for i in range(len(methods)):
    for j in range(len(methods)):
        table_write.write(start_row + i, start_col + j, ds1_result[i, j], style=xlwt.XFStyle())
        table_write.write(start_row + i, start_col + offset + j, ds2_result[i, j], style=xlwt.XFStyle())
        table_write.write(start_row + offset + i, start_col + j, ds3_result[i, j], style=xlwt.XFStyle())
        table_write.write(start_row + offset + i, start_col + offset +j, ds4_result[i, j], style=xlwt.XFStyle())
pass

os.remove(save_xls_file)
f.save(save_xls_file)
