import pandas as pd
import numpy as np


# Найти евклидово расстоняие между двумя объектами Series

n = 3 
s1 = pd.Series(np.random.randint(low=1, high=10, size=[n]))
s2 = pd.Series(np.random.randint(low=1, high=10, size=[n]))
 
res = np.linalg.norm(s1-s2)
print(res)

# Найти индексы локальных максимумов в объекте Series
 
s = pd.Series([1, 5, 7, 11, 8, 4, 12, 5, 8, 16, 8])
dd = np.diff(np.sign(np.diff(s)))
res = np.where(dd == -2)[0] + 1
print(res)

# Заменить пробелы наименее часто встречающимся символом
 
str_test = "Hello perfect world"
 
s = pd.Series(list(str_test))
freq = s.value_counts()
least_freq = freq.dropna().index[-1]
print(''.join(s.replace(' ', least_freq)))

# Создать объект Series, который содержит в индексах даты выходных дней субботы,
# а в значениях случайные числа от 1 до 10
 
s = pd.Series(np.random.randint(low=1,high=10,size=[10]), pd.date_range('2018-01-01', periods=10, freq='W-SAT'))
print(s)

# Заполнить пропущенные даты, значением выше (заполненной даты)
 
s = pd.Series([2, 5, 8, np.nan], index=pd.to_datetime(['2018-01-01', '2018-01-03', '2018-01-06', '2018-01-08']))
res = s.resample('D').ffill()
print(res)

# Вычислить автокорреляцию объекта Series
 
n = 20
 
s = pd.Series(np.arange(n))
s = s + np.random.normal(1, 3, n)
 
autocorr = [s.autocorr(lag=i).round(2) for i in range(n)]
 
print(autocorr)
