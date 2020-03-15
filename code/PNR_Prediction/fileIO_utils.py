import xlwt
import xlrd
from xlutils.copy import copy
import os


# plus的结果写入excel对应位置
def plus_write_to_excel(dataset_name=None, method1=None, method2=None,
                   precision_plus=None,
                   recall_plus=None,
                   F1score_plus=None,
                   AP_plus=None):
    excel_save_path = get_excel_save_path(dataset_name=dataset_name, method1=method1, method2=method2)
    if not is_excel_file_exist(excel_save_path):
        excel_save_path = get_excel_save_path(dataset_name=dataset_name, method1=method2, method2=method1)
        if not is_excel_file_exist(excel_save_path):
            print(excel_save_path + ": does not exists!")
            return None
    exist_f = xlrd.open_workbook(excel_save_path, formatting_info=True)
    f = copy(exist_f)
    sheet1 = f.get_sheet(0)

    column0 = ["precision", "recall", "F1score", "AP"]
    column1 = "plus-" + method1 + "&" + method2
    # 写第二列
    for i in range(0, len(column0)):
        sheet1.write(4 + i*4, 1, column1, style=xlwt.XFStyle())
    pass



    # 分别按行写入precision、recall、F1score、AP
    for i in range(0, len(precision_plus)):
        sheet1.write(4, i+2, precision_plus[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(recall_plus)):
        sheet1.write(8, i+2, recall_plus[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(F1score_plus)):
        sheet1.write(12, i+2, F1score_plus[i], style=xlwt.XFStyle())
    pass
    sheet1.write(16, 6, AP_plus, style=xlwt.XFStyle())


    os.remove(excel_save_path)
    f.save(excel_save_path)


# 把multiply的结果写入excel文件
def multiply_write_to_excel(dataset_name=None, method1=None, method2=None,
                   precision_multiply=None,
                   recall_multiply=None,
                   F1score_multiply=None,
                   AP_multiply=None):
    excel_save_path = get_excel_save_path(dataset_name=dataset_name, method1=method1, method2=method2)
    if not is_excel_file_exist(excel_save_path):
        excel_save_path = get_excel_save_path(dataset_name=dataset_name, method1=method2, method2=method1)
        if not is_excel_file_exist(excel_save_path):
            print(excel_save_path + ": does not exists!")
            return None
    exist_f = xlrd.open_workbook(excel_save_path, formatting_info=True)
    f = copy(exist_f)
    sheet1 = f.get_sheet(0)

    multiply_row_index = int(21+4*7)
    column0 = ["precision", "recall", "F1score", "AP"]
    column1 = "multiply-" + method1 + "&" + method2
    # 写第一列
    for i in range(0, len(column0)):
        sheet1.write(multiply_row_index + i, 0, column0[i], style=xlwt.XFStyle())
    pass

    # 写第二列
    for i in range(0, len(column0)):
        sheet1.write(multiply_row_index + i, 1, column1, style=xlwt.XFStyle())
    pass



    # 分别按行写入precision、recall、F1score、AP
    for i in range(0, len(precision_multiply)):
        sheet1.write(0 + multiply_row_index, i+2, precision_multiply[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(recall_multiply)):
        sheet1.write(1 + multiply_row_index, i+2, recall_multiply[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(F1score_multiply)):
        sheet1.write(2 + multiply_row_index, i+2, F1score_multiply[i], style=xlwt.XFStyle())
    pass
    sheet1.write(3 + multiply_row_index, 6, AP_multiply, style=xlwt.XFStyle())


    os.remove(excel_save_path)
    f.save(excel_save_path)

# 把深度学习的模型的结果写入excel文件
def DNN_write_to_excel(DL_name=None, dataset_name=None, method1=None, method2=None,
                   precision_DL=None,
                   recall_DL=None,
                   F1score_DL=None,
                   AP_DL=None):
    excel_save_path = get_excel_save_path(dataset_name=dataset_name, method1=method1, method2=method2)
    if not is_excel_file_exist(excel_save_path):
        excel_save_path = get_excel_save_path(dataset_name=dataset_name, method1=method2, method2=method1)
        if not is_excel_file_exist(excel_save_path):
            print(excel_save_path + ": does not exists!")
            return None
    exist_f = xlrd.open_workbook(excel_save_path, formatting_info=True)
    f = copy(exist_f)
    sheet1 = f.get_sheet(0)

    dl_row_index_group = {"mlp":0, "svm":1, "lr":2, "lgbm":3, "xgb":4, "ld":5, "rf":6}
    dl_row_index = int(dl_row_index_group[DL_name])
    column0 = ["precision", "recall", "F1score", "AP"]
    column1 = DL_name + "-" + method1 + "&" + method2
    # 写第一列
    for i in range(0, len(column0)):
        sheet1.write(21 + dl_row_index*4 + i, 0, column0[i], style=xlwt.XFStyle())
    pass

    # 写第二列
    for i in range(0, len(column0)):
        sheet1.write(21 + dl_row_index*4 + i, 1, column1, style=xlwt.XFStyle())
    pass



    # 分别按行写入precision、recall、F1score、AP
    for i in range(0, len(precision_DL)):
        sheet1.write(21 + dl_row_index*4, i+2, precision_DL[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(recall_DL)):
        sheet1.write(22 + dl_row_index*4, i+2, recall_DL[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(F1score_DL)):
        sheet1.write(23 + dl_row_index*4, i+2, F1score_DL[i], style=xlwt.XFStyle())
    pass
    sheet1.write(24 + dl_row_index*4, 6, AP_DL, style=xlwt.XFStyle())


    os.remove(excel_save_path)
    f.save(excel_save_path)




def get_excel_save_path(dataset_name=None, method1=None, method2=None):
    return  'D:\hybridrec/results//' + dataset_name + '--'+method1 + '--' + method2 + '.xls'


def is_excel_file_exist(file_path):
    is_exist = os.path.isfile(file_path)
    return is_exist
    pass



def write_to_excel(dataset_name=None, method1=None, method2=None,
                   precision_PNR=None, precision_method1=None, precision_method2=None,precision_weighted=None,
                   recall_PNR=None,recall_method1=None, recall_method2=None,recall_weighted=None,
                   F1score_PNR=None,F1score_method1=None, F1score_method2=None,F1score_weighted=None,
                   AP_PNR=None, AP_method1=None,AP_method2=None,AP_weighted=None,
                   AUC_PNR=None, AUC_method1=None, AUC_method2=None, AUC_weighted=None):

    excel_save_path=get_excel_save_path(dataset_name=dataset_name, method1=method1, method2=method2)
    f = xlwt.Workbook()

    sheet1 = f.add_sheet(dataset_name, cell_overwrite_ok=True)
    column0 = ["precision", "", "", "", "recall", "", "", "",
              "F1score", "", "", "", "AP", "", "", "",
              "AUC", "", "", ""]
    # 写第0列
    for i in range(0, len(column0)):
        sheet1.write(i+1, 0, column0[i], style=xlwt.XFStyle())

    # 写第0行
    row0 = ["", "", "1/20L", "1/10L", "1/5L", "1/2L", "L"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], style=xlwt.XFStyle())
    pass

    column1=["PNR-" + method1 + "&" + method2, method1, method2, "weight-" + method1 + "&" + method2,
             "PNR-" + method1 + "&" + method2, method1, method2, "weight-" + method1 + "&" + method2,
             "PNR-" + method1 + "&" + method2, method1, method2, "weight-" + method1 + "&" + method2,
             "PNR-" + method1 + "&" + method2, method1, method2, "weight-" + method1 + "&" + method2,
             "PNR-" + method1 + "&" + method2, method1, method2, "weight-" + method1 + "&" + method2]
    # 写第1列
    for i in range(0, len(column1)):
        sheet1.write(i+1, 1, column1[i], style=xlwt.XFStyle())
    pass

    # 写precision
    for i in range(0, len(precision_PNR)):
        sheet1.write(1, i+2, precision_PNR[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(precision_method1)):
        sheet1.write(2, i+2, precision_method1[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(precision_method2)):
        sheet1.write(3, i+2, precision_method2[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(precision_weighted)):
        sheet1.write(4, i+2, precision_weighted[i], style=xlwt.XFStyle())
    pass


    # 写recall
    for i in range(0, len(recall_PNR)):
        sheet1.write(5, i+2, recall_PNR[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(recall_method1)):
        sheet1.write(6, i+2, recall_method1[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(recall_method2)):
        sheet1.write(7, i+2, recall_method2[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(recall_weighted)):
        sheet1.write(8, i+2, recall_weighted[i], style=xlwt.XFStyle())
    pass


    # 写F1score
    for i in range(0, len(F1score_PNR)):
        sheet1.write(9, i+2, F1score_PNR[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(F1score_method1)):
        sheet1.write(10, i+2, F1score_method1[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(F1score_method2)):
        sheet1.write(11, i+2, F1score_method2[i], style=xlwt.XFStyle())
    pass
    for i in range(0, len(F1score_weighted)):
        sheet1.write(12, i+2, F1score_weighted[i], style=xlwt.XFStyle())
    pass

    # 写AP
    sheet1.write(13, 6, AP_PNR, style=xlwt.XFStyle())
    sheet1.write(14, 6, AP_method1, style=xlwt.XFStyle())
    sheet1.write(15, 6, AP_method2, style=xlwt.XFStyle())
    sheet1.write(16, 6, AP_weighted, style=xlwt.XFStyle())

    # AUC
    sheet1.write(17, 6, AUC_PNR, style=xlwt.XFStyle())
    sheet1.write(18, 6, AUC_method1, style=xlwt.XFStyle())
    sheet1.write(19, 6, AUC_method2, style=xlwt.XFStyle())
    sheet1.write(20, 6, AUC_weighted, style=xlwt.XFStyle())



    # f.save(r'D:\hybridrec/results/test.xls')
    f.save(excel_save_path)