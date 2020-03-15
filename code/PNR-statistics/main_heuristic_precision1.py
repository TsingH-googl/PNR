import xlwt
import xlrd
from xlutils.copy import copy
import os
from scipy.special import comb, perm


source_dir='D:\hybridrec//results//'

datasets=['facebook_combined', 'ca-hepph',
          'yeast', 'email-eucore'] # petster-friendships-hamster
methods_heuristic = ['cn', 'ja', 'aa', 'cosine', 'pearson', 'degreeproduct']
methods_embedding = ['drne', 'prone', 'deepwalk', 'node2vec']  # 'graph2gauss', 'attentionwalk', 'splitter'


# 保留几个k=L, k=L/2，。。。
k_num = 3  # 改这里


save_xls_file='D:\hybridrec\临时文件夹//heuristic_precision.xls'
if not os.path.isfile(save_xls_file):
    xls = xlwt.Workbook()
    sht1 = xls.add_sheet('Sheet1', cell_overwrite_ok=True)
    xls.save(save_xls_file)

# 打开要写入的文件
write_f = xlrd.open_workbook(save_xls_file, formatting_info=True)
f = copy(write_f)
table_write = f.get_sheet(0)

for datasets_i in range(len(datasets)):
    for method_i in range(len(methods_heuristic)):
        for method_j in range(method_i+1, len(methods_heuristic)):
            method1=methods_heuristic[method_i]
            method2=methods_heuristic[method_j]
            excel_name=datasets[datasets_i] + '--' + method1 + '--' + method2
            excel_path=source_dir + excel_name + '.xls'


            # 打开读入的文件
            read_f = xlrd.open_workbook(excel_path, formatting_info=True)
            table_read = read_f.sheet_by_index(0)

            start_row=1
            start_col=1+datasets_i*k_num # dataset_i*2或dataset_i*1


            # 写single baseline的结果
            for i in range(k_num):
                val= table_read.cell(2, 7-k_num+i).value
                table_write.write(start_row+method_i, # 写列名
                                  0,
                                  methods_heuristic[method_i], style=xlwt.XFStyle())
                table_write.write(start_row+method_i,
                                  start_col+i,
                                  val, style=xlwt.XFStyle())
            pass
            for i in range(k_num):
                table_write.write(start_row+method_j,
                                  0,
                                  methods_heuristic[method_j], style=xlwt.XFStyle())

                val= table_read.cell(3, 7-k_num+i).value
                table_write.write(start_row+method_j,
                                  start_col+i,
                                  val, style=xlwt.XFStyle())
            pass


            # 写MLP的结果
            combine_index=1/2 * (2*len(methods_heuristic) -method_i -1)*(method_i+1-1) + (method_j-method_i)
            combine_index=int(combine_index-1)
            for i in range(k_num):
                table_write.write(start_row+len(methods_heuristic)+combine_index,
                                  0,
                                  'MLP-('+methods_heuristic[method_i] + '+' + methods_heuristic[method_j] + ')',
                                  style=xlwt.XFStyle())
                val= table_read.cell(21, 7-k_num+i).value
                table_write.write(start_row+len(methods_heuristic)+combine_index,
                                  start_col+i,
                                  val, style=xlwt.XFStyle())
            pass


            # 写PNR的结果
            combine_index=1/2 * (2*len(methods_heuristic) -method_i -1)*(method_i+1-1) + (method_j-method_i)
            combine_index=int(   combine_index-1 + comb(len(methods_heuristic), 2)   )
            for i in range(k_num):
                table_write.write(start_row + len(methods_heuristic)+combine_index,
                                  0,
                                  'PNR-('+methods_heuristic[method_i] + '+' + methods_heuristic[method_j] + ')',
                                  style=xlwt.XFStyle())

                val= table_read.cell(1, 7 - k_num + i).value
                table_write.write(start_row + len(methods_heuristic)+combine_index,
                                  start_col + i,
                                  val, style=xlwt.XFStyle())
            pass
        pass
    pass

os.remove(save_xls_file)
f.save(save_xls_file)








