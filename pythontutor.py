def Task_1_1():
    #Напишите программу, которая считывает три числа и выводит их сумму. Каждое число записано в отдельной строке.
    
    a = int(input())
    b = int(input())
    c = int(input())
    sum = a + b + c
    print(sum)

def Task_2_1():
    #Даны два целых числа. Выведите значение наименьшего из них.

    a = int(input())
    b = int(input())
    print(min(a, b))

def Task_3_16():
    '''С начала суток часовая стрелка повернулась на угол в α градусов. Определите сколько полных часов, минут и секунд прошло с начала суток, 
                  то есть решите задачу, обратную задаче «Часы — 1». Запишите ответ в три переменные и выведите их на экран.''' 

    angle = float(input())
    print(int(angle // 30), int(angle % 30 * 2), int(angle % 0.5 * 120))

def Task_4_2():
    #Даны два целых числа A и В. Выведите все числа от A до B включительно, в порядке возрастания, если A < B, или в порядке убывания в противном случае.

    a = int(input())
    b = int(input())
    if a < b:
        for i in range(a, b + 1):
            print(i)
            i += 1
    else:
        for i in range(a, b - 1, -1):
            print(i)
            i += 1

def Task_5_12():
    #Дана строка. Удалите из нее все символы, чьи индексы делятся на 3.

    string = input()
    res = ""
    for i in range(len(string)):
        if i % 3 != 0:
            res += string[i]
    print(res)

Task_1_1()
Task_2_1()
Task_3_16()
Task_4_2()
Task_5_12()