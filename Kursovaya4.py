import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import geom
import numpy as np
import pandas as pd

#
# r_values = list(range(n + 1))
#
# dist = [binom.pmf(r, n, p) for r in r_values]
#
# print(mean)

num = 500
n = 20
N = 10
p = 0.6
s = np.random.binomial(n, p, num)
s1 = [np.random.binomial(n, p, 1)[0] for i in range(num)]
s2 = [np.random.geometric(1/36) for i in range(num)]
mean, var = binom.stats(n, p)
print('E binomial = ' + str(mean))

pd.Series(s).hist(bins=(N + 1), range=[0, (n + 1)])
plt.xlabel('binomial')
plt.show()
# print(*s)
poisson = [np.random.poisson(12) for i in range(num)]
print(*poisson)
pd.Series(poisson).hist(bins=(N + 1), range=[0, (n + 1)], color='green')
plt.xlabel('poisson')
plt.show()
print('E s1 = ' + str(n * p))
#print(*s1)

pd.Series(s2).hist(bins=(N + 1), range=[0, (50)], color='red')
plt.xlabel('geometric')
plt.show()
print(sum(s2)/num)
print('E s2 = ' + str(geom.mean(1/12)))
#print(*s2)
# def fun2(scenario, tetaR, eps):
#     global A
#     global B
#     global logA
#     global logB
#     global teta0
#     global teta1
#
#     tetaW = teta0 if scenario % 2 == 0 else teta1
#     accept = False  # True if scenario % 2 == 0 else
#
#     number = 0
#     sumZ = 0
#     values = []
#     Zs = []
#
#     while True:
#         t = 0.
#         picker = np.random.uniform(0, 1)
#
#         if picker <= eps:
#             if scenario < 2:
#                 t = np.random.poisson(tetaR)
#             else:
#                 t = tetaR
#         else:
#             t = np.random.poisson(tetaW)
#         # t = np.random.poisson(tetaW)
#         values.append(t)
#         Zs.append(calcZ(t, teta0, teta1))
#         sumZ += Zs[-1]
#
#         if scenario % 2 == 0:
#             if sumZ >= logA:
#                 accept = True  # отколняем H0
#                 break
#             if sumZ <= logB:
#                 accept = False  # принимаем H0
#                 break
#             if logB < sumZ < logA:
#                 number += 1
#         elif scenario % 2 == 1:
#             if sumZ >= logA:
#                 accept = False  # принимаем H1
#                 break
#             if sumZ <= logB:
#                 accept = True  # отклоняем H1
#                 break
#             if logB < sumZ < logA:
#                 number += 1
#
#     return [accept, number]
#
#
# def test():
#
#     wrongs = [[]] * (teta1 - teta0 + 1)
#     observations = [[]] * (teta1 - teta0 + 1)
#     cumWrongs = [[]] * (teta1 - teta0 + 1)
#     cumObservations = [[]] * (teta1 - teta0 + 1)
#     for tetaR in range(teta0, teta1 + 1):
#         # result = [False, 0.]
#         cumWrong = 0.
#         cumObs = 0.
#         wrongs[tetaR - teta0] = [0.] * workNum
#         observations[tetaR - teta0] = [0.] * workNum
#
#         cumObservations[tetaR - teta0] = [0.] * workNum
#         cumWrongs[tetaR - teta0] = [0.] * workNum
#         # 50000 запуск теста, к ним считается число ошибок
#         for eps in range(0, workNum):
#             # result = [False, 0.]
#             wrong = 0.
#             obs = 0.
#             for i in range(0, n + 1):
#                 result = fun2(var, tetaR, float(eps / steps))
#                 # result = Test1(teta0, teta1)
#                 obs += result[1]
#                 if not result[0]:
#                     wrong += 1
#             wrongs[tetaR - teta0][eps] = wrong
#             observations[tetaR - teta0][eps] = obs
#             cumObs += obs
#             cumWrong += wrong
#             cumObservations[tetaR - teta0][eps] = cumObs
#             cumWrongs[tetaR - teta0][eps] = cumWrong
#             # print("При eps = " + str(eps / steps) + "\t wrong = " + str(wrong))
#             # +"\tres[0] = "+str(result[0])+"\tobs = "+str(result[1]))
#         minW = min(wrongs[tetaR - teta0])
#         maxW = max(wrongs[tetaR - teta0])
#         avgW = stats.mean(wrongs[tetaR - teta0])
#         minO = min(observations[tetaR - teta0])
#         maxO = max(observations[tetaR - teta0])
#         avgO = stats.mean(observations[tetaR - teta0])
#
#         # for t in range(10):
#         #     print(wrongs[tetaR - teta0][t*5:(t+1)*5])
#
#         # print("eps = "+str(eps / steps)+"\tmin wrong = "+str(minW))
#         write_res("task #" + str(var), tetaR, minO, minW, maxO, maxW, avgO, avgW)
#         print("\nПри teta = " + str(tetaR))
#         print("min obs = " + str(minO) + "\nmin wrong = " + str(minW))
#         print("max obs = " + str(maxO) + "\nmax wrong = " + str(maxW))
#         print("\navg obs = " + str(avgO) + "\navg wrong = " + str(avgW))
#
#     return observations, wrongs, cumObservations, cumWrongs
