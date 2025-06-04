import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import statistics as stats

import time


def calcA(a, b):
    return (1. - b) / a


def calcB(a, b):
    return b / (1. - a)


def pfx(x, lam):
    return lam ** x / (math.factorial(x) * math.exp(lam))


def calcZ(x, theta0, theta1):
    return math.log(pfx(x, theta1) / pfx(x, theta0))



n = 1000
obses = 1000
steps = 200
teta0 = 48
teta1 = 57
teta0s = 48
teta1s = 57
teta0ss = 48
teta1ss = 84
a = 0.05
b = 0.05
A = calcA(a, b)
B = calcB(a, b)
workNum = steps // 2


def drawplot(mas, rotate, name):
    mas = np.array(mas)
    mas = mas.transpose()
    fig = plt.figure()
    axes = Axes3D(fig)
    xval = np.linspace(teta0, teta1, teta1 - teta0 + 1)
    yval = np.linspace(0, 0.5, workNum)
    x, y = np.meshgrid(xval, yval)
    z = mas
    axes.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.jet, label=name)
    axes.set_zlabel(name)
    axes.scatter(x, y, z, cmap=cm.jet)
    axes.view_init(rotate[0], rotate[1])
    plt.savefig(name + '.jpg')
    plt.show()


def drawplot2(mas, rotate, name):
    mas = np.array(mas)
    mas = mas.transpose()
    fig = plt.figure()
    axes = Axes3D(fig)
    xval = np.linspace(teta0, teta1, 2)
    yval = np.linspace(0, 0.5, workNum)
    x, y = np.meshgrid(xval, yval)
    z = mas
    axes.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.jet, label=name)
    axes.set_zlabel(name)
    axes.scatter(x, y, z, cmap=cm.jet)
    axes.view_init(rotate[0], rotate[1])
    plt.savefig(name + '.jpg')
    plt.show()


def drawplot2d_h1(mas, name, isWrongs):
    mas = np.array(mas)
    print(mas)
    # f.write(*mas)
    # f.write("\n")
    yval = np.linspace(0, 0.5, workNum)
    index = 1 if isWrongs else 0
    z = mas  # [mas[0][i] for i in range(len(mas[0]))]
    # print()
    plt.plot(yval, z)
    plt.xlabel(name)
    plt.savefig(name + '.jpg')
    plt.show()


def drawplot2d_h2(mas, name):
    mas = np.array(mas)
    print(mas)
    yval = np.linspace(0, 0.5, workNum)
    z = mas  # [mas[1][i] for i in range(len(mas[0]))]
    # print()
    plt.plot(yval, z)
    plt.xlabel(name)
    plt.savefig(name + '.jpg')
    plt.show()


def drawplot2d_h2s(mas0, mas1, name):
    mas0 = np.array(mas0)

    yval = np.linspace(0, 0.5, workNum)
    z0 = mas0  # [mas[1][i] for i in range(len(mas[0]))]
    z1 = mas1  # [mas[1][i] for i in range(len(mas[0]))]
    # print()
    plt.plot(yval, z0, label='H0=12, H1=21')
    plt.plot(yval, z1, label='H0=48, H1=57')
    plt.legend(loc='best')
    plt.xlabel(name)
    plt.savefig(name + '.jpg')
    plt.show()


def drawplot2d_h2ss(mas0, mas1, mas2, name):
    mas0 = np.array(mas0)

    yval = np.linspace(0, 0.5, workNum)
    z0 = mas0  # [mas[1][i] for i in range(len(mas[0]))]
    z1 = mas1  # [mas[1][i] for i in range(len(mas[0]))]
    z2 = mas2  # [mas[1][i] for i in range(len(mas[0]))]
    # print()
    plt.plot(yval, z0, label='H0=12, H1=21')
    plt.plot(yval, z1, label='H0=48, H1=57')
    plt.plot(yval, z2, label='H0=48, H1=84')
    plt.legend(loc='best')
    plt.xlabel(name)
    plt.savefig(name + '.jpg')
    plt.show()


def test(var, k=10):
    wrongs = [[]] * 2  # (teta1 - teta0 + 1)
    observations = [[]] * 2  # (teta1 - teta0 + 1)
    cumWrongs = [[]] * 2  # (teta1 - teta0 + 1)
    cumObservations = [[]] * 2  # (teta1 - teta0 + 1)
    index = 0
    for tetaR in [teta0, teta1]:
        # result = [False, 0.]
        cumWrong = 0.
        cumObs = 0.
        wrongs[index] = [0.] * workNum
        observations[index] = [0.] * workNum

        cumObservations[index] = [0.] * workNum
        cumWrongs[index] = [0.] * workNum

        # 50000 запусков теста, к ним считается число ошибок
        for eps in range(0, workNum):
            wrong = 0.
            obs = 0.
            for i in range(0, n + 1):
                result = testNoisy(index, tetaR, float(eps / steps), k)
                obs += result[1]
                if not result[0]:
                    wrong += 1
            print("eps = " + str(eps / steps) + "\twrongs = " + str(wrong) + "\tobservations = " + str(obs))
            wrongs[index][eps] = wrong
            observations[index][eps] = obs
            cumObs += obs
            cumWrong += wrong
            cumObservations[index][eps] = cumObs
            cumWrongs[index][eps] = cumWrong

        minW = min(wrongs[index])
        maxW = max(wrongs[index])
        avgW = stats.mean(wrongs[index])
        minO = min(observations[index])
        maxO = max(observations[index])
        avgO = stats.mean(observations[index])
        index += 1
        # print("eps = "+str(eps / steps)+"\tmin wrong = "+str(minW))
        # write_res("task #" + str(var), tetaR, minO, minW, maxO, maxW, avgO, avgW)
        print("\nПри teta = " + str(tetaR))
        print("min obs = " + str(minO) + "\nmin wrong = " + str(minW))
        print("max obs = " + str(maxO) + "\nmax wrong = " + str(maxW))
        print("\navg obs = " + str(avgO) + "\navg wrong = " + str(avgW))
    return observations, wrongs, cumObservations, cumWrongs


def test4(var, sub, k=10):
    tetaR = teta0 if var == 0 else teta1
    if sub == 1:
        tetaR = teta0s if var == 0 else teta1s
    if sub == 2:
        tetaR = teta0ss if var == 0 else teta1ss
    # result = [False, 0.]
    cumWrong = 0.
    cumObs = 0.
    wrongs = [0.] * workNum
    observations = [0.] * workNum

    cumObservations = [0.] * workNum
    cumWrongs = [0.] * workNum

    # 50000 запуск теста, к ним считается число ошибок
    for eps in range(0, workNum):
        # f.write("\nfor "+str(eps))
        # result = [False, 0.]
        wrong = 0.
        obs = 0.
        for i in range(0, n + 1):

            result = testNoisy(var, tetaR, float(eps / steps), sub, k)

            # result = fun2(var, tetaR, float(eps / steps))
            # result = Test1(teta0, teta1)
            obs += result[1]
            if not result[0]:
                wrong += 1
        print("eps = " + str(eps / steps) + "\twrongs = " + str(wrong) + "\tobservations = " + str(obs))
        f.write("\neps = " + str(eps / steps) + "\twrongs = " + str(wrong) + "\tobservations = " + str(obs))
        wrongs[eps] = wrong
        observations[eps] = obs
        cumObs += obs
        cumWrong += wrong
        cumObservations[eps] = cumObs
        cumWrongs[eps] = cumWrong
    # print("При eps = " + str(eps / steps) + "\t wrong = " + str(wrong))
    # +"\tres[0] = "+str(result[0])+"\tobs = "+str(result[1]))
    minW = min(wrongs)
    maxW = max(wrongs)
    avgW = stats.mean(wrongs)
    minO = min(observations)
    maxO = max(observations)
    avgO = stats.mean(observations)
    # for t in range(10):
    #     print(wrongs[tetaR - teta0][t*5:(t+1)*5])

    # print("eps = "+str(eps / steps)+"\tmin wrong = "+str(minW))
    # write_res("task #" + str(var), tetaR, minO, minW, maxO, maxW, avgO, avgW)
    print("\nПри teta = " + str(tetaR))
    print("min obs = " + str(minO) + "\nmin wrong = " + str(minW))
    print("max obs = " + str(maxO) + "\nmax wrong = " + str(maxW))
    print("\navg obs = " + str(avgO) + "\navg wrong = " + str(avgW))
    f.write("\n\nПри teta = " + str(tetaR))
    f.write("\nmin obs = " + str(minO) + "\nmin wrong = " + str(minW))
    f.write("\nmax obs = " + str(maxO) + "\nmax wrong = " + str(maxW))
    f.write("\navg obs = " + str(avgO) + "\navg wrong = " + str(avgW))
    # print(*observations)
    return observations, wrongs, cumObservations, cumWrongs


def write_res(task, tetaR, minO, minW, maxO, maxW, avgO, avgW):
    f.write(task)
    f.write("\nПри teta = " + str(tetaR))
    f.write("\nmin obs = " + str(minO) + "\nmin wrong = " + str(minW))
    f.write("\nmax obs = " + str(maxO) + "\nmax wrong = " + str(maxW))
    f.write("\navg obs = " + str(avgO) + "\navg wrong = " + str(avgW))
    f.write("\n\n")


r1_ = 30
r2_ = 170


def f1():
    print("\n****Верна первая гипотеза****\n")
    res = test(0)
    drawplot(res[0], [45, 80], 'наблюдения H0')
    drawplot(res[0], [r1_, 170], 'наблюдения H0_')
    drawplot(res[1], [25, -15], 'ошибки первого рода')
    drawplot(res[1], [r1_, 195], 'ошибки первого рода_')

    drawplot(res[2], [25, -15], 'накопленные наблюдения H0')
    drawplot(res[3], [25, 170], 'накопленные ошибки первого рода')
    # res(wrongs0, observations0, "t1_0")


def f4_1():
    print("\n****Верна первая гипотеза****\n")
    res = test4(0, 0)
    drawplot2d_h1(res[0], 'наблюдения при верной первой гипотезе', False)
    drawplot2d_h1(res[1], 'ошибки первого рода', True)
    # drawplot2d_h1(res[2], 'накопленные наблюдения H0')
    # drawplot2d_h1(res[3], 'накопленные ошибки первого рода')


def f4_2():
    print("\n****Верна вторая гипотеза****\n")
    res = test4(1, 0)
    res1 = test4(1, 1)
    res2 = test4(1, 2)
    # drawplot2d_h2(res[0], 'наблюдения при верной второй гипотезе')
    drawplot2d_h2ss(res[0], res1[0], res2[0], 'наблюдения при верной второй гипотезе')
    # drawplot2d_h2(res[1], 'ошибки второго рода')
    drawplot2d_h2ss(res[1], res1[1], res2[1], 'ошибки второго рода')
    # drawplot2d_h2(res[2], 'накопленные наблюдения H1')
    # drawplot2d_h2(res[3], 'накопленные ошибки второго рода')


def f2():
    print("\n****Верна вторая гипотеза****\n")
    res = test(1)
    drawplot(res[0], [25, -5], 'наблюдения H1')
    drawplot(res[0], [r1_, 195], 'наблюдения H1_')
    drawplot(res[1], [25, -25], 'ошибки второго рода')
    drawplot(res[1], [r1_, r2_], 'ошибки второго рода_')

    drawplot(res[2], [25, -15], 'накопленные наблюдения Н1')
    drawplot(res[3], [25, -25], 'накопленные ошибки второго рода')
    # res(wrongs1, observations1, "t1_1")


def f3():
    print("\n****Верна первая гипотеза****\n")
    res = test(2)
    drawplot(res[1], [35, 5], 't2 ошибки первого рода')
    drawplot(res[1], [r1_, 170], 't2 ошибки первого рода_')
    drawplot(res[0], [35, 5], 't2 наблюдения H0')
    drawplot(res[0], [r1_, 170], 't2 наблюдения H0_')

    drawplot(res[2], [25, -15], 't2 накопленные наблюдения H0')
    drawplot(res[3], [25, -15], 't2 накопленные ошибки первого рода')
    # res(wrongs2, observations2, "t2_0")


def f4():
    print("\n****Верна вторая гипотеза****\n")
    res = test(3)

    drawplot(res[1], [35, 5], 't2 ошибки второго рода')
    drawplot(res[1], [r1_, 170], 't2 ошибки второго рода_')
    drawplot(res[0], [35, 5], 't2 наблюдения H1')
    drawplot(res[0], [r1_, 170], 't2 наблюдения H1_')

    drawplot(res[2], [25, -45], 't2 накопленные наблюдения H1')
    drawplot(res[3], [25, -15], 't2 накопленные ошибки второго рода')
    # res(wrongs3, observations3, "t2_1")


def second():
    print("\n************ПУНКТ 1************\n")
    f1()
    f2()
    print("\n************ПУНКТ 2************\n")
    f3()
    f4()


def third():
    print("\"Загрязнение\" биномиальным распределением:")
    f4_1()
    f4_2()
    # f2()


def corrupt(hyp, eps, tetaR):
    return round(float((1 - eps) * hyp + eps * np.random.poisson(tetaR)), 5)


logB = math.log(B)
logA = math.log(A)


def fun2(scenario, tetaR, eps):
    global A
    global B
    global logA
    global logB
    global teta0
    global teta1

    tetaW = teta0 if scenario % 2 == 0 else teta1
    accept = False  # True if scenario % 2 == 0 else
    number = 0
    sumZ = 0
    values = []
    Zs = []

    while True:
        t = 0.
        picker = np.random.uniform(0, 1)
        if picker <= eps:
            if scenario < 2:
                t = np.random.poisson(tetaR)
            else:
                t = tetaR
        else:
            t = np.random.poisson(tetaW)
        values.append(t)
        Zs.append(calcZ(t, teta0, teta1))
        sumZ += Zs[-1]
        if scenario % 2 == 0:
            if sumZ >= logA:
                accept = True  # отколняем H0
                break
            if sumZ <= logB:
                accept = False  # принимаем H0
                break
            if logB < sumZ < logA:
                number += 1
        elif scenario % 2 == 1:
            if sumZ >= logA:
                accept = False  # принимаем H1
                break
            if sumZ <= logB:
                accept = True  # отклоняем H1
                break
            if logB < sumZ < logA:
                number += 1
    return [accept, number]


def Test0(teta0, teta1):
    sumZ = 0
    accept = False
    global A
    global B
    number = 0
    values = []
    Zs = []
    while True:
        k = np.random.poisson(teta0)
        values.append(k)
        Zs.append(calcZ(k, teta0, teta1))
        sumZ += Zs[-1]
        if sumZ >= math.log(A):
            accept = True  # принимаем H0
            break
        if sumZ <= math.log(B):
            accept = False  # отклоняем H0
            break
        if math.log(B) < sumZ < math.log(A):
            number += 1
    return [accept, number]


def Test1(teta0, teta1):
    sumZ = 0
    accept = False
    global A
    global B
    number = 0
    values = []
    Zs = []
    while True:
        k = np.random.poisson(teta1)
        Zs.append(calcZ(k, teta0, teta1))
        sumZ += Zs[-1]
        if sumZ >= math.log(A):
            accept = True  # принимаем H1
            break
        if sumZ <= math.log(B):
            accept = False  # отклоняем H1
            break
        if math.log(B) < sumZ < math.log(A):
            number += 1
    return [accept, number]


def testNoisy(scenario, tetaR, eps, sub, k=10):
    global A
    global B
    global logA
    global logB
    global teta0
    global teta1
    if sub == 1:
        teta0 = teta0s
        teta1 = teta1s
    if sub == 2:
        teta0 = teta0ss
        teta1 = teta1ss
    accept = False  # True if scenario % 2 == 0 else
    thetaR = teta0 if scenario % 2 == 0 else teta1
    number = 0
    sumZ = 0
    values = []
    Zs = []
    N = n / k
    P = thetaR * k / n

    while True:
        xt = 0
        isBin = False
        picker = np.random.uniform(0, 1)

        if picker >= eps:
            xt = np.random.poisson(thetaR)
        else:
            xt = np.random.poisson(thetaR)
            # Binomial
            # xt = int(np.random.binomial(N, P, 1)[0])
            # while xt == 0:
            #     xt = int(np.random.binomial(N, P, 1)[0])

            # Geometrical
            # xt = int(np.random.geometric(1 / thetaR))
            # while xt == 0 or xt > 100:
            #     xt = int(np.random.geometric(1 / thetaR))
            isBin = True
        values.append(xt)
        Zs.append(calcZ(xt, teta0, teta1))
        sumZ += Zs[-1]
        if scenario % 2 == 0:
            if sumZ >= logA:
                accept = False  # отклоняем H0
                break
            if sumZ <= logB:
                accept = True  # принимаем H0
                break
            if logB < sumZ < logA:
                number += 1
        elif scenario % 2 == 1:
            if sumZ >= logA:
                accept = True  # принимаем H1
                break
            if sumZ <= logB:
                accept = False  # отклоняем H1
                break
            if logB < sumZ < logA:
                number += 1
    return [accept, number]


# first(1)
# start_time = time.time()
# print(fun2(at, bt, 1, 65, float(0.1 / steps)))
print("Входные:")
print("H0 = " + str(teta0))
print("H1 = " + str(teta1))
print("a = " + str(a))
print("b = " + str(b))
# print("шагов: " + str(steps))

f = open("results.txt", 'w')

third()
# second()

f.close()
# print("--- %s seconds ---" % (time.time() - start_time))
# print(np.linspace(0,1,21))
# np.random.seed(123)
#
# x = np.random.randint(-5, 5, 40)
# y = np.random.randint(0, 10, 40)
# z = np.random.randint(-5, 5, 40)
# s = np.random.randint(10, 100, 40)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, s=s)
# plt.show()
