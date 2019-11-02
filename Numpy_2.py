import numpy as np
import scipy.spatial as spatial

A = np.random.random_integers(0, 10, (5, 3))
B = np.random.random_integers(0, 10, (3, 2))
Res = np.dot(A, B)
print(f"Перемножить матрицы 5x3 и 3x2:\n {A} \n {B} \n {Res}")

C = np.arange(0, 15)
#for i in range(len(C)):
#    if C[i] > 3 and C[i] < 8:
#        C[i] *= -1
C[(C > 3) & (C < 8)] *= -1
print(f"Дан массив, поменять знак у элементов, значения которых между 3 и 8:\n {C}")

D = [0, 1, 2, 3, 4]
i = 0
Res = np.arange(5)
while i < 4:
     Res = np.vstack([Res, D])
     i += 1
print(f"Создать 5x5 матрицу со значениями в строках от 0 до 4:\n {Res}")

def generator():
    for i in range(15):
        yield i

E = np.fromiter(generator(), dtype=int, count=-1)
print(f"Есть генератор, сделать с его помощью массив: {E}")

F = np.random.random_sample(10)
print(f"Создать вектор размера 10 со значениями от 0 до 1, не включая ни то, ни другое:\n{F}")

G = np.random.random(10)
G.sort()
print(f"Отсортировать вектор: \n{G}")

H = np.random.random(10)
I = np.random.random(10)
print(f"Проверить, одинаковы ли 2 numpy массива: \n{np.allclose(H, I)}")

# Сделать массив неизменяемым
J = np.random.random(10)
J.flags.writeable = False
#J[0] = 1

# Дан массив 10x2 (точки в декартовой системе координат), преобразовать в полярную
K = np.random.random((10,2))
X,Y = K[:,0], K[:,1]
L = np.hypot(X, Y)
M = np.arctan2(Y,X)
print(L)
print(M)

N = np.random.random(10)
print(N)
N[N.argmax()] = 0
print(f"Заменить максимальный элемент на ноль:\n{N}")

Z = np.zeros((10,10), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
print(f"Создать структурированный массив с координатами x, y на сетке в квадрате [0,1]x[0,1]:\n{Z}")

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(f"Из двух массивов сделать матрицу Коши C (Cij = 1/(xi - yj)):\n {np.linalg.det(C)}")

print("Найти минимальное и максимальное значение, принимаемое каждым числовым типом numpy: ")
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)

np.set_printoptions()
Z = np.zeros((25,25))
print(f"Напечатать все значения в массиве:\n{Z}")

Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(f"Найти ближайшее к заданному значению число в заданном массиве:\n{Z[index]}")

Z = np.zeros(10, [ ('position', [ ('x', float, 1), ('y', float, 1)]),
                    ('color', [ ('r', float, 1), ('g', float, 1), ('b', float, 1)])])
print(f"Создать структурированный массив, представляющий координату (x,y) и цвет (r,g,b):\n{Z}")

Z = np.random.random((10,2))
D = spatial.distance.cdist(Z,Z)
print(f"Дан массив (100,2) координат, найти расстояние от каждой точки до каждой:\n{D}")

# Преобразовать массив из float в int
Z = np.arange(10, dtype=np.int32)
Z = Z.astype(np.float32, copy=False)

Z = np.genfromtxt("input.txt", delimiter=",")
print(f"Дан файл. Необходимо прочитать его:\n{Z}")

Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])

# Каков эквивалент функции enumerate для numpy массивов?
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)

X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
D = np.hypot(X, Y)
sigma, mu = 1.0, 0
G = np.exp(-((D - mu) ** 2 / (2 * sigma ** 2)))
print(f"Сформировать 2D массив с распределением Гаусса:\n{G}")

p = 3
Z = np.zeros((10,10))
np.put(Z, np.random.choice(range(10*10), p, replace=False), 1)
print(f"Случайно расположить p элементов в 2D массив:\n{Z}")

X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)
print(f"Отнять среднее из каждой строки в матрице:\n{Y}")

Z = np.random.randint(0,10,(3,3))
n = 1  # Нумерация с нуля
print(f"Отсортировать матрицу по n-ому столбцу:\n{Z}\n {Z[Z[:,n].argsort()]}")

Z = np.random.randint(0,3,(3,10))
print(f"Определить, есть ли в 2D массиве нулевые столбцы\n {(~Z.any(axis=0)).any()}")

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(f"Дан массив, добавить 1 к каждому элементу с индексом, заданным в другом массиве:\n {Z}")

w,h = 16,16
I = np.random.randint(0, 2, (h,w,3)).astype(np.ubyte)
F = I[...,0] * 256 * 256 + I[...,1] * 256 + I[...,2]
n = len(np.unique(F))
print(f"Дан массив (w,h,3) (картинка) dtype=ubyte, посчитать количество различных цветов: {np.unique(I)}")

A = np.random.randint(0,10, (3,4,3,4))
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(f"Дан четырехмерный массив, посчитать сумму по последним двум осям: {sum}")

A = np.random.random((5, 5))
B = np.random.random((5, 5))
C = np.einsum("ij,ji->i", A, B)
print(f"Найти диагональные элементы произведения матриц: {C}")
