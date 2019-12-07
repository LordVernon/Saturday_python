import pandas as pd
import numpy as np
from dateutil.parser import parse
from collections import Counter
import re

# Получить от объекта Series показатели описательной статистики

A = np.random.RandomState(42)
s = pd.Series(A.normal(10, 5, 25))
pkz = s.describe()
print(pkz)

# Узнать частоту уникальных элементов объекта Series (гистограмма)

data = 'qwertyuiop'
len_series = 30
s = pd.Series(np.take(list(data), np.random.randint(len(data), size=len_series)))
res = s.value_counts() 
print(res)

# Заменить все элементы объекта Series на "Other", кроме двух наиболее часто встречающихся
 
state = np.random.RandomState(42)
s = pd.Series(state.randint(low=1, high=5, size=[13]))
print(s.value_counts())
s[s.isin(s.value_counts().index[:2])] = 'Other'
print(s)

# Создать объект Series в индексах дата каждый день 2018 года, в значениях случайное значение

dti = pd.date_range(start='2018-01-01', end='2018-12-31', freq='B') 
s = pd.Series(np.random.rand(len(dti)), index=dti)
 
# Найти сумму всех вторников

res = s[s.index.weekday == 2].sum()
print('Сумма всех "вторников"', res)
print()
 
# Для каждого месяца найти среднее значение

res = s.resample('M').mean()
print('Средние значения по месяцам:\n', res)
print()

# Преобразовать объект Series в DataFrame заданной формы (shape)

s = pd.Series(np.random.randint(low=1, high=10, size=[35]))

r = (7, 5)   
df = pd.DataFrame(s.values.reshape(r))
print(df)

# Найти индексы объекта Series кратные 3
 
s = pd.Series(np.random.randint(low=1, high=10, size=[7]))
res = s[s % 3 == 0].index
print(res)

# Получить данные по индексам объекта Series
 
s = pd.Series(list("abcdefghijklmnopqrstuvwxyz"))
p = [0, 4, 8, 14, 20, 10] 
res = s.take(p)
print(res)

# Объединить два объекта Series вертикально и горизонтально

s1 = pd.Series(range(5))
s2 = pd.Series(list('abcde'))
vertical = s1.append(s2)
horizontal = pd.concat([s1, s2], axis=1)
print(f"{vertical}\n{horizontal}")

# Получить индексы объекта Series A, данные которых содержатся в объетке Series B
 
s1 = pd.Series([5, 3, 2, 1, 4, 11, 13, 8, 7])
s2 = pd.Series([1, 5, 13, 2])
res = np.argwhere(s1.isin(s2)).flatten()
print(res)

# Получить объект Series B, котоырй содержит элементы без повторений объекта A

s = pd.Series(np.random.randint(low=1, high=10, size=[10]))
res = pd.Series(s.unique())
print(res)

# 18. Преобразовать каждый символ объекта Series в верхний регистр
 
s = pd.Series(['life', 'is', 'interesting'])
 
# преобразование данных Series в строку
s = pd.Series(str(i) for i in s)
res = pd.Series(i.title() for i in s)
print(res)

# Рассчитать количество символов в объекте Series
 
s = pd.Series(['one', 'two', 'three', 'four', 'five'])
# преобразование в строковый тип
s = pd.Series(str(i) for i in s)
res = np.asarray([len(i) for i in s])
print(res)

# Найти разность между объектом Series и смещением объекта Series на n

n = 1
s = pd.Series([1, 5, 7, 8, 12, 15, 17])
res = s.diff(periods=n)
print(res)

# Преобразовать разыне форматы строк объекта Series в дату

s = pd.Series(['2019/01/01', '2019-12-12', '15 Jan 2020'])
res = pd.to_datetime(s) 
print(res)

# Поскольку работа с датой часто встречается в работе, то см. еще один пример
# все данные должны иметь одинаковый формат (часто бывает выгрузка из SQL)

s = pd.Series(['14.02.2019', '04.06.1997', '01.03.2019'])
# преобразование в дату

res = pd.to_datetime(s, format='%d.%m.%Y')
print(res)

# Получить год, месяц, день, день недели, номер дня в году от объекта Series (string)
s = pd.Series(['01 Jan 2018', '02-02-2011', '20120303', '2013/04/04', '2018-12-31'])
 
# парсим в дату и время
s_ts = s.map(lambda x: parse(x, yearfirst=True))
 
# получаем года
print(s_ts.dt.year) 
# получаем месяца
print(s_ts.dt.month)
# получаем дни
print(s_ts.dt.day) 
# получаем номер недели
print(s_ts.dt.weekofyear) 
# получаем номер дня в году
print(s_ts.dt.dayofyear)

#  Отобрать элементы объекта Series, кторые содержат не менее двух гласных
 
s = pd.Series(['Яблоко', 'Orange', 'Plan', 'Python', 'Апельсин', 'Стол', 'Reliance'])
mask = s.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiouаоиеёэыуюя')]) >= 2)
res = s[mask]
print(res)

# Отобрать e-маилы из объекта Series
 
emails = pd.Series(['test text @test.com', 'test@mail.ru', 'test.2ru', 'test@pp'])
pattern = '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
mask = emails.map(lambda x: bool(re.match(pattern, x)))
res = emails[mask]
print(res)

# Получить среднее значение каждого уникального объекта Series s1 через "маску" другого объекта Series s2
 
n = 10
s1 = pd.Series(np.random.choice(['dog', 'cat', 'horse', 'bird'], n))
s2 = pd.Series(np.linspace(1,n,n))
res = s2.groupby(s1).mean()
print(res)
