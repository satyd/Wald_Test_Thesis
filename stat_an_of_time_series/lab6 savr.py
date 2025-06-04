import matplotlib.pyplot as plt
import math
import numpy as np
import statsmodels.api as sm

# def sturges_formula(n):
#     return 1 + math.floor(math.log2(n))
#
#
# def chi_2(m1, m1_, N):
#     res = 0
#     print("та самая формула:")
#     for i in range(N):
#         r = ((m1[i] - m1_[i]) ** 2) / m1_[i]
#         res += r
#         print(r)
#     return res


# f = open("/Users/mac/Desktop/вучоба/Д:С/статАналВР/лаб 1/source.txt", "r")

# t = []

# april = []
# may = []
# june = []

april = [9.1, 9.0, 9.3, 8.7, 8.7, 8.2, 9.0, 8.8, 8.9, 8.8, 8.6, 8.4, 8.5, 8.4, 8.5, 8.5, 7.8, 8.1, 7.9, 7.9, 7.2, 7.9,
         7.4, 7.1, 8.5, 7.8, 9.3, 7.7, 7.8, 7.9, 7.7, 7.8, 8.0, 7.5, 7.6, 7.6, 7.4]
may = [9.6, 9.4, 10.1, 9.6, 9.9, 9.1, 8.8, 9.4, 9.3, 9.1, 9.1, 8.8, 9.1, 8.9, 9.4, 8.9, 8.6, 8.9, 7.9, 8.3, 8.9, 8.5,
       8.9, 9.1, 8.8, 9.3, 9.6, 8.5, 8.9, 8.4, 8.5, 8.8, 8.7, 8.6, 8.6, 8.6, 8.4]
june = [11.3, 10.9, 10.4, 11.3, 10.4, 10.7, 9.8, 10.4, 10.9, 10.7, 10.5, 10.2, 10.6, 9.6, 10.9, 9.7, 10.0, 10.1, 9.2,
        9.4, 9.9, 8.5, 9.4, 10.4, 10.1, 9.9, 9.5, 9.5, 9.7, 9.6, 9.0, 10.0, 9.9, 10.2, 10.8, 9.8, 9.3]

# t = [int(x) for x in next(f).split()]
# counter = 1
# for line in f:
#     if counter == 1:
#         april = [float(x) for x in line.split()]
#     if counter == 2:
#         may = [float(x) for x in line.split()]
#     if counter == 3:
#         june = [float(x) for x in line.split()]
#     counter += 1
#
# f.close()

# june.sort()
for x in june:
    print(x)

print("")

T = int(len(june))
lags = 18
mul = 1
s = np.linspace(0, mul*lags, mul*lags+1)
rs = []

sum_l_t_s = []
for i in s:
    j = int(i)
    sum_l_t_s.append(0)
    for y in range(T - j):
        sum_l_t_s[j] += june[y]

sum_l_t_plus_s = []
for i in s:
    j = int(i)
    sum_l_t_plus_s.append(0)
    for y in range(T - j):
        sum_l_t_plus_s[j] += june[y + j]

sum_pow2_s = []
for i in s:
    j = int(i)
    sum_pow2_s.append(0)
    for y in range(T - j):
        sum_pow2_s[j] += (june[y] - 1/(T - j) * sum_l_t_s[j]) ** 2

sum_pow2_plus_s = []
for i in s:
    j = int(i)
    sum_pow2_plus_s.append(0)
    for y in range(T - j):
        sum_pow2_plus_s[j] += (june[y + j] - 1/(T - j) * sum_l_t_plus_s[j]) ** 2

sum_two_sums = []
for i in s:
    j = int(i)
    sum_two_sums.append(0)
    for y in range(T - j):
        sum_two_sums[j] += ((june[y] - 1/(T - j) * sum_l_t_s[j]) * (june[y + j] - 1/(T - j) * sum_l_t_plus_s[j]))

for s_ in s:
    j = int(s_)
    rs.append(1/(T - j) * sum_two_sums[j] / (math.sqrt(1/(T - j) * sum_pow2_s[j]) * math.sqrt(1/(T - j) *
                                                                                              sum_pow2_plus_s[j])))

for s_ in rs[1:]:
    print(round(s_, 5))

plt.plot(s[1:], rs[1:])
plt.scatter(s[1:], rs[1:])
plt.show()
#
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(rs, lags=13, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(rs, lags=13, ax=ax2)
# plt.show()



# plt.plot(t, april, color='blue', label='седьмой месяц')  # 7
# plt.plot(t, may, color='green', label='восьмой месяц')  # 8
# plt.plot(t, june, color='black', label='девятый месяц')  # 9
# plt.legend()
#
# plt.show()

# april_min = min(april)
# april_max = max(april)
#
# R = april_max - april_min
# n = len(april)
# N = sturges_formula(n)
# h = R / N
#
# a1 = []
# april_int_med = []
# for i in range(N + 1):
#     a1.append(april_min + h * i)
#     if i < N:
#         april_int_med.append(a1[i] + h / 2)  # середины интервалов
# april_int_med = np.array(april_int_med)
#
# a1 = np.array(a1)
#
# h1 = np.histogram(april, N, (april_min, april_max))
# plt.hist(april, bins=N, range=(april_min, april_max))
# plt.show()
#
# m1 = h1[0]  # абсолютные частоты
#
# w1 = m1 * 1 / N  # относительные частоты
#
# april_mean = 1 / n * sum(m1 * april_int_med)
#
# april_sqr = []
# for x in april_int_med:
#     april_sqr.append((x - april_mean) ** 2)
#
# april_sqr = np.array(april_sqr)
#
# S1 = math.sqrt(1 / n * sum(m1 * april_sqr))
#
# z1 = []
#
# for i in range(N + 1):
#     if i == 0:
#         z1.append(- np.inf)
#     elif i == N:
#         z1.append(np.inf)
#     else:
#         z1.append((a1[i] - april_mean) / S1)
#
# z1 = np.array(z1)
#
# f = open("/Users/mac/Desktop/вучоба/Д:С/статАналВР/лаб 1/Laplace.txt", "r")
# Phi = {}
# for line in f:
#     k, v = [float(x) for x in line.split()]
#     Phi[k] = v
#
# f.close()
#
#
# Phi1 = []
# for i in range(N + 1):
#     if z1[i] not in Phi.keys():
#         if z1[i] >= 5:
#             Phi1.append(0.5)
#         elif z1[i] == -np.inf or z1[i] <= -5:
#             Phi1.append(-0.5)
#         elif 0 > z1[i] > -5:
#             Phi1.append(-Phi[-round(z1[i], 2)])
#         else:
#             Phi1.append(Phi[round(z1[i], 2)])
#
# P = []
# for i in range(N):
#     P.append(Phi1[i + 1] - Phi1[i])
#
# m1_ = []
#
# for i in range(N):
#     m1_.append(n * P[i])
#
# m1_ = np.array(m1_)
#
# print("m1:")
# for m in m1:
#     print(m)
#
# print("\nm1_:")
# for m in m1_:
#     print(m)
# print("\n\n\n\n\n")
#
# counter = 0
# for m in m1:
#     if m < 5 and (counter == 0 or counter == N - 2):
#         m1[counter + 1] += m
#         m1_[counter + 1] += m1_[counter]
#         m1 = np.delete(m1, counter)
#         m1_ = np.delete(m1_, counter)
#         N -= 1
#         counter -= 1
#         a1 = np.delete(a1, counter + 1)
#     counter += 1
#
# vyb_sr1 = 1 / n * sum(april)
# vyb_sr2 = 1 / n * sum(may)
# vyb_sr3 = 1 / n * sum(june)
#
#
# vyb_disp1 = 1 / n * sum([(i - vyb_sr1) ** 2 for i in april])
# vyb_disp2 = 1 / n * sum([(i - vyb_sr1) ** 2 for i in may])
# vyb_disp3 = 1 / n * sum([(i - vyb_sr1) ** 2 for i in june])
#
# ispr_vyb_disp1 = 1 / (n - 1) * sum([(i - vyb_sr1) ** 2 for i in april])
# ispr_vyb_disp2 = 1 / (n - 1) * sum([(i - vyb_sr1) ** 2 for i in may])
# ispr_vyb_disp3 = 1 / (n - 1) * sum([(i - vyb_sr1) ** 2 for i in june])
#
# std_otkl1 = vyb_disp1 ** (1/2)
# std_otkl2 = vyb_disp2 ** (1/2)
# std_otkl3 = vyb_disp3 ** (1/2)
#
# koeff_var1 = vyb_disp1 / vyb_sr1
# koeff_var2 = vyb_disp2 / vyb_sr2
# koeff_var3 = vyb_disp3 / vyb_sr3
#
# skewness1 = 1 / (n * std_otkl1 ** 3) * sum([(i - vyb_sr1) ** 3 for i in april])
# skewness2 = 1 / (n * std_otkl2 ** 3) * sum([(i - vyb_sr2) ** 3 for i in may])
# skewness3 = 1 / (n * std_otkl3 ** 3) * sum([(i - vyb_sr3) ** 3 for i in june])
#
# exc1 = 1 / (n * std_otkl1 ** 4) * sum([(i - vyb_sr1) ** 4 for i in april]) - 3
# exc2 = 1 / (n * std_otkl2 ** 4) * sum([(i - vyb_sr2) ** 4 for i in may]) - 3
# exc3 = 1 / (n * std_otkl3 ** 4) * sum([(i - vyb_sr3) ** 4 for i in june]) - 3
#
# april_range = april
# april_range.sort()
# may_range = may
# may_range.sort()
# june_range = june
# june_range.sort()
#
# med1 = april_range[(n - 1) // 2] if (n % 2) == 1 else april_range[(n // 2 + (n + 2) // 2) // 2]
# med2 = may_range[(n - 1) // 2] if (n % 2) == 1 else april_range[(n // 2 + (n + 2) // 2) // 2]
# med3 = june_range[(n - 1) // 2] if (n % 2) == 1 else april_range[(n // 2 + (n + 2) // 2) // 2]
#
# print("Выборочное среднее:")
# print(vyb_sr1)
# print(vyb_sr2)
# print(vyb_sr3)
# print("Выборочная дисперсия:")
# print(vyb_disp1)
# print(vyb_disp2)
# print(vyb_disp3)
# print("Исправленная выборочная дисперсия:")
# print(ispr_vyb_disp1)
# print(ispr_vyb_disp2)
# print(ispr_vyb_disp3)
# print("Стандартное отклонение:")
# print(std_otkl1)
# print(std_otkl2)
# print(std_otkl3)
# print("Коэффициент вариации:")
# print(koeff_var1)
# print(koeff_var2)
# print(koeff_var3)
# print("Коэффициент асимметрии:")
# print(skewness1)
# print(skewness2)
# print(skewness3)
# print("Коэффициент эксцесса:")
# print(exc1)
# print(exc2)
# print(exc3)
# print("Медиана:")
# print(med1)
# print(med2)
# print(med3)
# print("\n\n\n\n\n\n")
#
# print("m1:")
# for m in m1:
#     print(m)
#
# print("\nm1_:")
# for m in m1_:
#     print(m)
# print("\n\n\n\n\n")
#
# chi = chi_2(m1, m1_, N)
# print("Наблюдаемое значение критерия: " + str(chi))
# if chi < 3.841458821:
#     print(str(chi) + " < 3.84 (критическая точка. Получена с помощью MS Excel).")
#     print("Нет оснований отвергнуть гипотезу о нормальном распределении генеральной совокупности.")
# else:
#     print(str(chi) + " > 3.84 (критическая точка. Получена с помощью MS Excel).")
#     print("Гипотеза о нормальном распределении генеральной совокупности отвергается.")
