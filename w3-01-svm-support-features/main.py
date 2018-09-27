import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


# 1. Загрузите выборку из файла svm-data.csv. В нем записана двумерная выборка
# (целевая переменная указана в первом столбце, признаки — во втором и третьем).

df = pd.read_csv('svm-data.csv', header=None)
print(df)
y = df[0]
X = df.loc[:, 1:]

# 2. Обучите классификатор с линейным ядром, параметром C = 100000 и random_state=241.
# Такое значение параметра нужно использовать, чтобы убедиться, что SVM работает с выборкой как с линейно разделимой.
# При более низких значениях параметра алгоритм будет настраиваться с учетом слагаемого в функционале,
# штрафующего за маленькие отступы, из-за чего результат может не совпасть с решением классической
# задачи SVM для линейно разделимой выборки.

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)

# 3. Найдите номера объектов, которые являются опорными (нумерация с единицы). Они будут являться ответом на задание.
# Обратите внимание, что в качестве ответа нужно привести номера объектов в возрастающем порядке через запятую
# или пробел. Нумерация начинается с 1.

sv = [n + 1 for n in sorted(list(clf.support_))]
print(sv)

# Plotting
x_min, x_max = X[1].min(), X[1].max()
y_min, y_max = X[2].min(), X[2].max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
# plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
            levels=[-.5, 0, .5])
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
            facecolors='none', zorder=10, edgecolors='k')
plt.scatter(X[1], X[2], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.show()
