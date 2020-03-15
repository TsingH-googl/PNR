import xlrd
import os
from shutil import copyfile

# 确定数据的好坏的类别
def divide_into_category(cell_Precision_PNR, cell_Precision_method1, cell_Precision_method2,
                         cell_AP_PNR, cell_AP_method1, cell_AP_method2):
    if (cell_Precision_PNR > cell_Precision_method1) & (cell_Precision_PNR > cell_Precision_method2) and \
        (cell_AP_PNR > cell_AP_method1) & (cell_AP_PNR > cell_AP_method2):
        return 'better'
    elif ((cell_Precision_PNR > cell_Precision_method1) & (cell_Precision_PNR > cell_Precision_method2)) and \
            (not((cell_AP_PNR > cell_AP_method1) & (cell_AP_PNR > cell_AP_method2))):
        return 'good-precision'
    elif (not((cell_Precision_PNR > cell_Precision_method1) & (cell_Precision_PNR > cell_Precision_method2))) and \
            ((cell_AP_PNR > cell_AP_method1) & (cell_AP_PNR > cell_AP_method2)):
        return 'good-ap'
    else:
        return 'worse'



if __name__ == '__main__':

    xls_results_dir = 'D:\hybridrec//results//'  #改这里
    xls_results_dir_filtered= 'D:\hybridrec//results//' # 改这里

    xls_files = os.listdir(xls_results_dir)

    for i in range(len(xls_files)):
        if '.xls' in xls_files[i]:
            xls_file_path = xls_results_dir + xls_files[i]
            data = xlrd.open_workbook(xls_file_path)
            table = data.sheets()[0]
            cell_Precision_PNR = table.cell(1, 6).value
            cell_Precision_method1 = table.cell(2, 6).value
            cell_Precision_method2 = table.cell(3, 6).value
            cell_AP_PNR = table.cell(13, 6).value
            cell_AP_method1 = table.cell(14, 6).value
            cell_AP_method2 = table.cell(15, 6).value
            category = divide_into_category(cell_Precision_PNR=cell_Precision_PNR,
                                            cell_Precision_method1=cell_Precision_method1,
                                            cell_Precision_method2=cell_Precision_method2,
                                            cell_AP_PNR=cell_AP_PNR,
                                            cell_AP_method1=cell_AP_method1,
                                            cell_AP_method2=cell_AP_method2)

            xls_target_dir = xls_results_dir_filtered + category + '//'
            if not os.path.isdir(xls_target_dir):
                os.makedirs(xls_target_dir)

            src_file_path = xls_results_dir + xls_files[i]
            dst_file_path = xls_target_dir + xls_files[i]
            copyfile(src_file_path, dst_file_path)

        pass
    pass





