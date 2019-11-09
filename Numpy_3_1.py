import numpy as np
from numpy.lib import stride_tricks

A = np.array([1, 2, 3, 4, 5])
nz = 3
A0 = np.zeros(len(A) + (len(A) - 1) * nz)
A0[::nz + 1] = A
print(f"Дан вектор [1, 2, 3, 4, 5], построить новый вектор с тремя нулями между каждым значением: {A0}")

B = np.random.randint(0, 15, (3, 3))
print(B)
B[[0, 1]] = B[[1, 0]]
print(f"Поменять 2 строки в матрице:\n {B}")

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(f"Рассмотрим набор из 10 троек, описывающих 10 треугольников (с общими вершинами), найти множество уникальных отрезков, составляющих все треугольники:\n{G}")

#Дан массив C; создать массив A, что np.bincount(A) == C
C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)

D = np.arange(20)
n = 3
ret = np.cumsum(D, dtype=float)
ret[n:] = ret[n:] - ret[:-n]
print(f"Посчитать среднее, используя плавающее окно: {ret[n - 1:] / n}")

#Дан вектор Z, построить матрицу, первая строка которой (Z[0],Z[1],Z[2]),
#  каждая последующая сдвинута на 1 (последняя (Z[-3],Z[-2],Z[-1]))
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)

Z = rolling(np.arange(10), 3)
print(Z)

Z = np.random.uniform(-1.0,1.0,100)
print(Z)
np.negative(Z, out=Z)
print(f"Инвертировать булево значение, или поменять знак у числового массива без создания нового:\n{Z}")

#Рассмотрим 2 набора точек P0, P1 описания линии (2D) и точку р, как вычислить расстояние от р до каждой линии i (P0[i],P1[i])
def distance(P0, P1, p):
    T = P1 - P0
    L = (T ** 2).sum(axis=1)
    U = -((P0[:,0] - p[...,0]) * T[:,0] + (P0[:,1] - p[...,1]) * T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U * T - p
    return np.sqrt((D ** 2).sum(axis=1))

P0 = np.random.uniform(-10, 10,(10,2))
P1 = np.random.uniform(-10, 10,(10, 2))
p  = np.random.uniform(-10, 10,(1, 2))
print(distance(P0, P1, p))

#Дан массив. Написать функцию, выделяющую часть массива фиксированного размера с центром в данном элементе 
# (дополненное значением fill если необходимо)
Z = np.random.randint(0,10, (10,10))
shape = (5, 5)
fill  = 0
position = (1, 1)

R = np.ones(shape, dtype=Z.dtype) * fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P - Rs // 2)
Z_stop  = (P + Rs // 2) + Rs % 2

R_start = (R_start - np.minimum(Z_start, 0)).tolist()
Z_start = (np.maximum(Z_start, 0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start, R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start, Z_stop)]
R[r] = Z[z]
print(Z)
print(R)

Z = np.random.uniform(0, 1,(10, 10))
rank = np.linalg.matrix_rank(Z)
print(f"Посчитать ранг матрицы: {Z}")

Z = np.random.randint(0,10,50)
print(f"Найти наиболее частое значение в массиве: {np.bincount(Z).argmax()}")

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0] - n)
j = 1 + (Z.shape[1] - n)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(f"Извлечь все смежные 3x3 блоки из 10x10 матрицы:\n{C}")
