import numpy as np

print(f"Версия numpy: {np.version.full_version}")

Null = np.zeros(10)
print(f"Вектор (одномерный массив) размера 10, заполненный нулями: \n {Null}")

Ones = np.ones(10)
print(f"Вектор (одномерный массив) размера 10, заполненный единицами:\n {Ones}")

A = np.full(10, 2.5)
print(f"Вектор размера 10, заполненный числом 2.5:\n {A}")

Null = np.zeros(10)
Null[4] = 1
print(f"Вектор размера 10, заполненный нулями, но пятый элемент равен 1:\n {Null}")

B = np.arange(10, 50, 1)
print(f"вектор со значениями от 10 до 49: {B}")

C = np.arange(6)
print("Развернуть вектор (первый становится последним)")
print(C)
print(C[::-1])

D = np.arange(9).reshape(3, 3)
print(f"Создать матрицу (двумерный массив) 3x3 со значениями от 0 до 8:\n {D}")

Arr = np.array([1, 2, 0, 0, 4, 0])
index = np.nonzero(Arr)
print(f"Найти индексы ненулевых элементов в [1,2,0,0,4,0]: {index}")
#for i in range(len(Arr)):
    #if Arr[i] != 0:
    #    print(i, end=' ')

E = np.random.random((3, 3, 3))
print(f"Создать массив 3x3x3 со случайными значениями:\n {E}")

F = np.random.random((10, 10))
max_F, min_F = F.max(), F.min()
print(f"Матрица: {F}")
print(f"Максимальное значение {max_F}, минимальное {min_F}")

G = np.random.random(30)
print(f"Среднее значение: {G.mean()}")

Ones = np.ones((3, 3))
Ones[1:-1, 1:-1] = 0
print(f"Создать матрицу с 0 внутри, и 1 на границах:\n {Ones}")

Diag = np.diag(np.arange(1, 5), k=-1)
print(f"Создать 5x5 матрицу с 1,2,3,4 под диагональю:\n {Diag}")

Chess = np.ones((8, 8))
Chess[1::2,::2] = 0
Chess[::2, 1::2] = 0
print(f"Создать 8x8 матрицу и заполнить её в шахматном порядке:\n {Chess}")

print(f"Дан массив размерности (6,7,8). Каков индекс (x,y,z) сотого элемента?\n {np.unravel_index(100, (6, 7, 8))}")

Chess = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(f"Создать 8x8 матрицу и заполнить её в шахматном порядке, используя функцию tile\n {Chess}")

print("Выяснить результат следующих выражений")
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
