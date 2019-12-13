import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Прочитать файл CSV (определенные столбцы и определенное кол-во строк) и перевести его в DataFrame
 
df = pd.read_csv('https://raw.githubusercontent.com/Grossmend/CSV/master/titanic/data.csv', nrows=10, usecols=['Name', 'Sex', 'Survived'])
print(df)
# Прочитать файл CSV и перевести каждую 100-ую строку в DataFrame
 
df = pd.read_csv('https://raw.githubusercontent.com/Grossmend/CSV/master/titanic/data.csv', chunksize=100)
df_each = pd.DataFrame()
for chunk in df:
    df_each = df_each.append(chunk.iloc[0,:])
print(df_each)
# 36. Преобразовать индексы объекта Series в столбец DataFrame
 
test_list = 'abcedf'
test_arr = np.arange(len(test_list))
test_dict = dict(zip(test_list, test_arr))
s = pd.Series(test_dict)
 
df = s.to_frame().reset_index()
df.columns=['letter', 'number']
print(df)
# Посмотреть информацию объекта DataFrame
 
df = pd.read_csv('https://raw.githubusercontent.com/Grossmend/CSV/master/titanic/data.csv', nrows=10)
 
# Вывести формат каждого столбца
print('\n', 'Формат столбцов:')
print(df.dtypes)
 
# Вывести размерность DataFrame
print('\n', 'Размерность:')
print(df.shape)
 
# Вывести общую статистику
print('\n', 'Общая статистика')
print(df.describe())
