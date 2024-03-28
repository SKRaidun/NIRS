import glob


# from pathlib import Path
# from fileinput import filename
# import os

# функция на парсинг списка файлов формата .txt
# в котором содержатся наши данные, возвращаем список файлов

def read_sort_file_txt():
    list_file_txt = glob.glob('./sorted_data_txt/*.txt')
    return list_file_txt


# создаем словарь с параметрами рабочих файлов.
def dict_datafiles():
    list_name_file = []
    list_id_file = []
    list_run = []
    list_type_detector = []
    list_contur = []
    list_degree = []
    dict_list_file = {}
    file = []

    for id_file in range(len(read_sort_file_txt())):
        list_name_file.append(read_sort_file_txt()[id_file].split('_'))
        list_id_file.append(id_file)
        list_run.append(list_name_file[id_file][2].replace('txt\\RUN', ''))
        list_type_detector.append(list_name_file[id_file][3])
        list_contur.append(list_name_file[id_file][4])
        list_degree.append(list_name_file[id_file][5].replace('.txt', ''))

    dict_list_file['id_file'] = list_id_file
    #dict_list_file['name_of_file'] = read_sort_file_txt()
    dict_list_file['run'] = list_run
    dict_list_file['type_detector'] = list_type_detector
    dict_list_file['contur'] = list_contur
    dict_list_file['degree'] = list_degree

    return dict_list_file


print(dict_datafiles())
