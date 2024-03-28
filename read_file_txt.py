import glob
# from pathlib import Path
# from fileinput import filename
# import os

# функция на парсинг списка файлов формата .txt
# в котором содержатся наши данные, возвращаем список файлов

# def read_sort_file_txt():
#     list_file = list(sorted(Path('.').glob('*.txt')))
#     return list_file

# извлекаем название из файлов данных
# def get_name_datafiles():
#     i = 0
#     while i < len(list_file_str):
#         list_name_file = list_file_str[i].split('_')
#         i += 1
#     return list_name_file

# def get_name_datafiles():
#     i = 0
#     list_name_file = list_file_str[i].split('_')
#     return list_name_file


def read_sort_file_txt():
    #way_data = r'sorted_data_txt'
    #os.chdir(way_data)
    list_file_txt = glob.glob('./sorted_data_txt/*.txt')
    return list_file_txt

def get_name_datafiles():
    id_file = 0
    list_name_file = []
    list_id_file = []
    list_run = []
    list_type_detector = []
    list_contur = []
    list_degree = []
    # dict_list_file = {}
    for id_file in range(len(read_sort_file_txt())):
        list_name_file.append(read_sort_file_txt()[id_file].split('_'))
        list_id_file.append(id_file)
        list_run.append(list_name_file[id_file][2].replace('txt\\RUN', ''))
        list_type_detector.append(list_name_file[id_file][3])
        list_contur.append(list_name_file[id_file][4])
        list_degree.append(list_name_file[id_file][5].replace('.txt', ''))

    return list_id_file, list_run, list_degree, list_type_detector, list_contur


print(get_name_datafiles())
# while i < len(read_sort_file_txt()):
#     print(get_name_datafiles())
#     i += 1


# for file in files:
#     fname = os.path.basename(file)
#     dict_[fname] = (pd.read_csv(file, header=0, dtype=str, encoding='cp1252')
#                       .fillna(''))