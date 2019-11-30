import pandas as pd
import numpy as np
import sys

A = pd.Series([[1 , 2, 3], np.zeros(3), {'a':1, 'b':2}])
B = pd.Series(['a', 'b', 'c', 'd', [1, 2, 3]])
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([4, 5, 6, 7, 8])

df = A.to_frame()
df1 = pd.DataFrame(pd.concat([A, B]))
print(df1)

# индексы
ans = s1[s1.isin(s2)]
print(ans) 
#  значения
ans2 = np.setdiff1d(s1, s2, assume_unique=False)
print(ans2)
 
# возвращает вместе с индексами
 
# получаем объединенный Series без повтороений
s_union = pd.Series(np.union1d(s1, s2))
# получаем пересекающиеся данные
s_intersect = pd.Series(np.intersect1d(s1, s2))
# отбираем все данные, кроме пересекающихся
ans3 = s_union[~s_union.isin(s_intersect)]
print(ans3)

# возвразает значения
ans4 = np.setxor1d(s1, s2, assume_unique=False)
 
print(ans4)
