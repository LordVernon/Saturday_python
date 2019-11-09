import numpy as np

#Создать подкласс симметричных 2D массивов (Z[i,j] == Z[j,i])
class Symetric(np.ndarray):
    def __setitem__(self, (i,j), value):
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)

#Рассмотрим множество матриц (n,n) и множество из p векторов (n,1). Посчитать сумму p произведений матриц 
# (результат имеет размерность (n,1))
p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(f"Дан массив 16x16, посчитать сумму по блокам 4x4:\n{S}")

Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
print(f"Найти n наибольших значений в массиве:\n{Z[np.argpartition(-Z,n)[:n]]}")

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = map(len, arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print(f"Построить прямое произведение массивов (все комбинации с каждым элементом):\n{cartesian(([1, 2, 3], [4, 5], [6, 7]))}")

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))
C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(f"Даны 2 массива A (8x3) и B (2x2). Найти строки в A, которые содержат элементы из каждой строки в B, независимо от порядка элементов в B:
      {rows}")

Z = np.random.randint(0,5,(10,3))
E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print("Дана 10x3 матрица, найти строки из неравных значений (например [2,2,3])")
print(Z)
print(U)

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(f"Преобразовать вектор чисел в матрицу бинарных представлений{np.unpackbits(I[:, np.newaxis], axis=1)}")

Z = np.random.randint(0, 2, (6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(f"Дан двумерный массив. Найти все различные строки:\n{uZ}")

# np.inner(A, B)
np.einsum('i,i', A, B) 
# np.outer(A, B)
np.einsum('i,j', A, B) 
# np.sum(A)
np.einsum('i->', A)
# A * B
np.einsum('i,i->i', A, B)    
