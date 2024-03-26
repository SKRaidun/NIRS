from pathlib import Path
from fileinput import filename

# функция на парсинг списка файлов формата .txt
# в котором содержатся наши данные, возвращаем список файлов

def read_sort_file_txt():
    list_file = list(sorted(Path('.').glob('*.txt')))
    return list_file

print(len(read_sort_file_txt()))

def get_name_datafiles():
    i = 0
    while i < len(read_sort_file_txt()):
        list_name_file = read_sort_file_txt().split('_')
        i += 1
    return list_name_file

#print(get_name_datafiles())


# for file in files:
#     fname = os.path.basename(file)
#     dict_[fname] = (pd.read_csv(file, header=0, dtype=str, encoding='cp1252')
#                       .fillna(''))