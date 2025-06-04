import math
from math import factorial as f
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import statistics as stats
from scipy.stats import multinomial
import pandas as pd


def calcA(a, b):
    return (1. - b) / a


def calcB(a, b):
    return b / (1. - a)


def calcZ(n, X, theta0, theta1):
    return math.log(pmf(n, X, theta1) / pmf(n, X, theta0))


def pmf(n, X, P):
    xFactM = 1
    for x in X:
        xFactM *= f(x)
    pDegsM = 1
    for i in range(len(X)):
        pDegsM *= P[i] ** X[i]
    return f(n) * pDegsM / xFactM


def test1(scenario, theta0, theta1, a, b, N, isPrint):
    A = calcA(a, b)
    B = calcB(a, b)
    logB = math.log(B)
    logA = math.log(A)
    wrongs = 0
    observations = 0
    global iterations
    # 1000 запусков теста, к ним считается число ошибок
    for i in range(iterations):
        result = simpleTest(scenario, theta0, theta1, logA, logB, N)
        observations += result[1]
        if not result[0]:
            wrongs += 1
    if isPrint:
        print("simple test result for " + str(iterations) + " iterations")
        print("scenario: " + str(scenario))
        print("obeservations = " + str(observations))
        print("wrongs = " + str(wrongs))
    return observations, wrongs


def test1b(scenario, theta0, theta1, a, b, N, isPrint):
    A = calcA(a, b)
    B = calcB(a, b)
    logB = math.log(B)
    logA = math.log(A)
    wrongs = 0
    observations = 0
    global iterations
    # 500 запуск теста, к ним считается число ошибок
    for i in range(iterations):
        result = simpleTestWithCounter(scenario, theta0, theta1, logA, logB, N)
        observations += result[1]
        if not result[0]:
            wrongs += 1
    if isPrint:
        print("simple test result for " + str(iterations) + " iterations")
        print("scenario: " + str(scenario))
        print("obeservations = " + str(observations))
        print("wrongs = " + str(wrongs))
    return observations, wrongs


def test2(scenario, theta0, theta1, thetaNoise, a, b, N):
    w = 3


def simpleTest(scenario, theta0, theta1, logA, logB, N):
    accept = False  # True if scenario % 2 == 0 else
    thetaR = theta0 if scenario % 2 == 0 else theta1

    number = 0
    sumZ = 0
    values = []
    Zs = []
    counter = 0
    limit = 1
    while True:
        X = np.random.multinomial(N, thetaR, size=1)[0]
        values.append(X)
        Zs.append(calcZ(N, X, theta0, theta1))
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


def testNoisy1(scenario, theta0, theta1, logA, logB, N, eps, thetaN):
    thetaR = theta0 if scenario % 2 == 0 else theta1
    X = np.random.multinomial(N, thetaR, size=1)[0]
    number = 0
    sumZ = 0
    values = []
    Zs = []
    while True:
        picker = np.random.uniform(0, 1)
        if picker < eps:
            X = np.random.multinomial(N, thetaN, size=1)[0]
        else:
            X = np.random.multinomial(N, thetaR, size=1)[0]
        values.append(X)
        Zs.append(calcZ(N, X, theta0, theta1))
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


def simpleTestWithCounter(scenario, theta0, theta1, logA, logB, N):
    accept = False  # True if scenario % 2 == 0 else
    thetaR = theta0 if scenario % 2 == 0 else theta1

    X = np.random.multinomial(N, thetaR, size=1)[0]
    number = 0
    sumZ = 0
    values = []
    Zs = []
    counter = 0
    limit = 10
    while True:
        values.append(X)
        Zs.append(calcZ(N, X, theta0, theta1))
        sumZ += Zs[-1]
        if scenario % 2 == 0:
            if sumZ >= logA:
                accept = False  # отклоняем H0
                if counter >= limit:
                    break
                counter += 1
            if sumZ <= logB:
                accept = True  # принимаем H0
                if counter >= limit:
                    break
                counter += 1
            if logB < sumZ < logA:
                number += 1
        elif scenario % 2 == 1:
            if sumZ >= logA:
                accept = True  # принимаем H1
                if counter >= limit:
                    break
                counter += 1
            if sumZ <= logB:
                accept = False  # отклоняем H1
                if counter >= limit:
                    break
                counter += 1
            if logB < sumZ < logA:
                number += 1

    return [accept, number]


def avg(mas):
    return sum(mas) / len(mas)


def drawplot2d_h1(mas, name, isWrongs):
    mas = np.array(mas)
    print(mas)
    # f.write(*mas)
    # f.write("\n")
    yval = np.linspace(1, runs, runs)
    index = 1 if isWrongs else 0
    z = mas  # [mas[0][i] for i in range(len(mas[0]))]
    # print()
    plt.plot(yval, z)
    plt.xlabel(name)
    plt.savefig(name + '.jpg')
    plt.show()


def drawplot2d_2lines(mas1, mas2, line1, line2, plotname, isWrongs):
    mas1 = np.array(mas1)
    mas2 = np.array(mas2)
    # f.write(*mas)
    # f.write("\n")
    yval = np.linspace(1, runs, runs)
    index = 1 if isWrongs else 0
    z1 = mas1  # [mas[0][i] for i in range(len(mas[0]))]
    # print()
    plt.plot(yval, mas1, label=line1)
    plt.plot(yval, mas2, label=line2)
    plt.legend(loc='best')
    plt.xlabel(plotname)
    plt.savefig(plotname + '.jpg')
    plt.show()


def drawplot2d_2lines_c3(mas1, mas2, line1, line2, plotname, isWrongs):
    mas1 = np.array(mas1)
    mas2 = np.array(mas2)
    # f.write(*mas)
    # f.write("\n")
    yval = np.linspace(N, N + runs, runs)
    index = 1 if isWrongs else 0
    z1 = mas1  # [mas[0][i] for i in range(len(mas[0]))]
    # print()
    plt.plot(yval, mas1, label=line1)
    plt.plot(yval, mas2, label=line2)
    plt.legend(loc='best')
    plt.xlabel(plotname)
    plt.savefig(plotname + '.jpg')
    plt.show()


def visualise_pt1(result, case):
    drawplot2d_2lines(result[0][0], result[1][0], "верна H0", "верна H1", case + str(" наблюдения"), False)
    drawplot2d_2lines(result[0][1], result[1][1], "первого рода", "второго рода", case + str(" ошибки"), True)


def visualise_pt1_c3(result, case):
    drawplot2d_2lines(result[0][0], result[1][0], "верна H0", "верна H1", case + str(" наблюдения"), False)
    drawplot2d_2lines(result[0][1], result[1][1], "первого рода", "второго рода", case + str(" ошибки"), True)


def smallPrint(header, obses, wrongs, wrongsProb):
    print(header)
    print("Наблюдения|Ошибки|Вероятности ошибки")
    print(avg(obses), end='\t|')
    print(avg(wrongs), end='\t|')
    print(avg(wrongsProb))


def smallTable(mas1, mas2, name):
    # mas1 = mas1.append[-1, "1"]
    # mas2 = mas2.append[-1, "2"]
    tabledata = ["1" + mas1, "2" + mas2]
    table = pd.DataFrame(tabledata, columns=["Верна гипотеза", "Наблюдения", "Ошибки", "Вероятность ошибки"]).style.hide_index()
    table.set_caption(name)



def simpleTestCase1(theta0, theta1, at, bt, runs, iterations):
    print("Входные:")
    print("H0 = " + str(theta0))
    print("H1 = " + str(theta1))
    print("a = " + str(at))
    print("b = " + str(bt))
    wrongs0 = []
    wrongsProb0 = []
    obses0 = []
    wrongs1 = []
    wrongsProb1 = []
    obses1 = []

    print("pt1")
    for i in range(runs):
        res = test1(0, theta0, theta1, at, bt, N, False)
        obses0.append(res[0])
        wrongs0.append(res[1])
        wrongsProb0.append(res[1] / iterations)
    print("pt2")
    for i in range(runs):
        res = test1(1, theta0, theta1, at, bt, N, False)
        obses1.append(res[0])
        wrongs1.append(res[1])
        wrongsProb1.append(res[1] / iterations)
    res0 = [avg(obses0), avg(wrongs0), avg(wrongsProb0)]
    res1 = [avg(obses1), avg(wrongs1), avg(wrongsProb1)]
    smallTable(res0, res1, "table1")
    # smallPrint("Верная первая гипотеза", obses0, wrongs0, wrongsProb0)
    # smallPrint("Верная вторая гипотеза", obses1, wrongs1, wrongsProb1)
    return [obses0, wrongs0], [obses1, wrongs1]


steps = 200
workNum = steps // 2
segments = 9


def testCase2(theta0, theta1, at, bt):
    print("\nТест 2 с помехами\nВходные:")
    print("H0 = " + str(theta0))
    print("H1 = " + str(theta1))
    global logAg
    global logBg
    wrongs = [[]] * segments  # (teta1 - teta0 + 1)
    observations = [[]] * segments  # (teta1 - teta0 + 1)
    cumWrongs = [[]] * segments  # (teta1 - teta0 + 1)
    cumObservations = [[]] * segments  # (teta1 - teta0 + 1)
    index = 0
    thetaNoise = theta0
    delta = [(theta1[i] - theta0[i]) / segments for i in range(len(theta0))]
    print("test 2 pt 1")
    while thetaNoise[0] > theta1[0]:
        cumWrong = 0.
        cumObs = 0.
        wrongs[index] = [0.] * workNum
        observations[index] = [0.] * workNum
        cumObservations[index] = [0.] * workNum
        cumWrongs[index] = [0.] * workNum

        for eps in range(0, workNum):
            wrong = 0.
            obs = 0.
            for i in range(iterations):
                res = testNoisy1(0, theta0, theta1, logAg, logBg, N, float(eps / workNum), thetaNoise)
                obs += res[1]
                if not res[0]:
                    wrong += 1
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

        print("\nПри teta = " + str(thetaNoise) + "\nindex = " + str(index))
        print("min obs = " + str(minO) + "\nmin wrong = " + str(minW))
        print("max obs = " + str(maxO) + "\nmax wrong = " + str(maxW))
        print("\navg obs = " + str(avgO) + "\navg wrong = " + str(avgW))
        index += 1
        for i in range(len(thetaNoise)):
            thetaNoise[i] += delta[i]

    return observations, wrongs, cumObservations, cumWrongs


def drawplot(mas, rotate, name):
    mas = np.array(mas)
    mas = mas.transpose()
    fig = plt.figure()
    axes = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(axes)
    xval = np.linspace(1, segments, segments)
    yval = np.linspace(0, 0.5, workNum)
    x, y = np.meshgrid(xval, yval)
    z = mas
    axes.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.jet, label=name)
    axes.set_zlabel(name)
    axes.scatter(x, y, z, cmap=cm.jet)
    axes.view_init(rotate[0], rotate[1])
    plt.savefig(name + '.jpg')
    plt.show()


def simpleTestCase1b(theta0, theta1, at, bt, runs, iterations):
    print("Входные:")
    print("H0 = " + str(theta0))
    print("H1 = " + str(theta1))
    wrongs0 = []
    wrongsProb0 = []
    obses0 = []
    wrongs1 = []
    wrongsProb1 = []
    obses1 = []

    print("pt1")
    for i in range(runs):
        res = test1b(0, theta0, theta1, at, bt, N, False)
        obses0.append(res[0])
        wrongs0.append(res[1])
        wrongsProb0.append(res[1] / iterations)
    print("pt2")
    for i in range(runs):
        res = test1b(1, theta0, theta1, at, bt, N, False)
        obses1.append(res[0])
        wrongs1.append(res[1])
        wrongsProb1.append(res[1] / iterations)
    drawplot2d_2lines(obses0, obses1, "наблюдения H0", "наблюдения H1", "наблюдения при N in 100 - 150", False)
    drawplot2d_2lines(wrongs0, wrongs1, "ошибки 1 рода", "ошибки 2 рода", "ошибки при N in 100-150", True)
    # smallPrint("Верная первая гипотеза", obses0, wrongs0, wrongsProb0)
    # smallPrint("Верная вторая гипотеза", obses1, wrongs1, wrongsProb1)
    return [obses0, wrongs0], [obses1, wrongs1]


def simpleTestCase2(theta0, theta1, at, bt, runs, iterations):
    print("Входные:")
    print("H0 = " + str(theta0))
    print("H1 = " + str(theta1))
    wrongs0 = []
    wrongsProb0 = []
    obses0 = []
    wrongs1 = []
    wrongsProb1 = []
    obses1 = []

    print("pt1")
    for i in range(runs):
        res = test1(0, theta0, theta1, at, bt, N + i, False)
        obses0.append(res[0])
        wrongs0.append(res[1])
        wrongsProb0.append(res[1] / iterations)
    print("pt2")
    for i in range(runs):
        res = test1(1, theta0, theta1, at, bt, N + i, False)
        obses1.append(res[0])
        wrongs1.append(res[1])
        wrongsProb1.append(res[1] / iterations)

    print("obs 0: " + str(obses0))
    print("wrongs 0: " + str(wrongs0))
    print("obs 1: " + str(obses1))
    print("wrongs 1: " + str(wrongs1))

    drawplot2d_2lines_c3(obses0, obses1, "наблюдения H0", "наблюдения H1", "наблюдения при N in 50 - 150", False)
    drawplot2d_2lines_c3(wrongs0, wrongs1, "ошибки 1 рода", "ошибки 2 рода", "ошибки при N in 50 - 150", True)
    smallPrint("Верная первая гипотеза", obses0, wrongs0, wrongsProb0)
    smallPrint("Верная вторая гипотеза", obses1, wrongs1, wrongsProb1)
    return [obses0, wrongs0], [obses1, wrongs1]


N = 50
runs = 100
iterations = 100
theta0 = [1 / 6.] * 6
theta1 = [1 / 7.] * 5 + [2 / 7.]
theta1_a = [1 / 7.] * 5 + [2 / 7.]
theta1_b = [1 / 8.] * 5 + [3 / 8.]
theta1_c = [2 / 8.] + [1 / 8.] * 4 + [2 / 8.]
theta1_d = [1 / 8.] * 2 + [2 / 8.] + [1 / 8.] * 2 + [2 / 8.]
a = 0.05
b = 0.05
Ag = calcA(a, b)
Bg = calcB(a, b)
logBg = math.log(Bg)
logAg = math.log(Ag)

print("norm0")
print(sum(theta0))
print("norm1a = " + str(sum(theta1_a)))
print("norm1b = " + str(sum(theta1_b)))
print("norm1c = " + str(sum(theta1_c)))
print("norm1d = " + str(sum(theta1_d)))

print("a = " + str(a))
print("b = " + str(b))
print("запусков: " + str(runs))
print("итераций теста: " + str(iterations))


def part1():
    print("\ncase A\n")
    resA1 = simpleTestCase1(theta0, theta1_a, a, b, runs, iterations)
    print("\ncase B\n")
    resB1 = simpleTestCase1(theta0, theta1_b, a, b, runs, iterations)
    print("\ncase C\n")
    resC1 = simpleTestCase1(theta0, theta1_c, a, b, runs, iterations)
    print("\ncase D\n")
    resD1 = simpleTestCase1(theta0, theta1_d, a, b, runs, iterations)

    visualise_pt1(resA1, "case A")
    visualise_pt1(resB1, "case B")
    visualise_pt1(resC1, "case C")
    visualise_pt1(resD1, "case D")


def part2():
    print("\ncase A1\n")
    resA1 = simpleTestCase1(theta0, theta1_a, a, b, runs, iterations)

    # print("\ncase A2\n")
    # resA2 = simpleTestCase1(theta0, theta1_a, a, 2*b, runs, iterations)
    #
    # print("\ncase A3\n")
    # resA3 = simpleTestCase1(theta0, theta1_a, 2*a, b, runs, iterations)
    #
    # print("\ncase A4\n")
    # resA4 = simpleTestCase1(theta0, theta1_a, a, b/5, runs, iterations)

    print("\ncase A5\n")
    resA5 = simpleTestCase1(theta0, theta1_a, a * 2, b / 5, runs, iterations)

    # visualise_pt1(resA1, "case A1")
    # visualise_pt1(resA2, "case A2")
    # visualise_pt1(resA3, "case A3")
    # visualise_pt1(resA4, "case A4")
    # visualise_pt1(resA5, "case A5")


def part3():
    print("\ncase A pt 3\n")
    resA = simpleTestCase2(theta0, theta1_a, a, b, runs, iterations)


r1_ = 30
r2_ = 170


def part4():
    print("pt 4 Noisy Test")
    res = testCase2(theta0, theta1, a, b)
    print(res[0])
    drawplot(res[0], [45, 70], 'наблюдения H0')
    drawplot(res[0], [r1_, 170], 'наблюдения H0_')
    drawplot(res[1], [25, -15], 'ошибки первого рода')
    drawplot(res[1], [r1_, 195], 'ошибки первого рода_')

    drawplot(res[2], [25, -15], 'накопленные наблюдения H0')
    drawplot(res[3], [25, 170], 'накопленные ошибки первого рода')


# part1()

#part2()
part3()
# part4()
# rv = multinomial(8, [0.3, 0.2, 0.5])
# vector_p = [0.3, 0.2, 0.5]
# vector_x = np.random.multinomial(8, vector_p, size=1)[0]
# print(vector_x)
# print(pmf(8, vector_x, vector_p))
# print(rv.pmf(vector_x))
# for i in range(20):
#     print(np.random.binomial(N, 0.5, 3))
# print("multi")
# for i in range(20):
# print(np.random.multinomial(40, [1/7.]*5 + [2/7.], size=1))
# print(np.random.multinomial(40, [1/6.]*6, size=1))
