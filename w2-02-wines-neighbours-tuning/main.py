import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import scale

from pandas import read_csv
import pandas as pd

df = pd.read_csv("wine.data", header=None)

# print(df.head())

# 1. Загрузите выборку Wine по адресу https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
# 2. Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта), признаки — в столбцах
# со второго по последний. Более подробно о сути признаков можно прочитать по адресу
# https://archive.ics.uci.edu/ml/datasets/Wine

y = df[0]
X = df.loc[:, 1:]

# 3. Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold). Создайте генератор разбиений,
# который перемешивает выборку перед формированием блоков (shuffle=True). Для воспроизводимости результата,
# создавайте генератор KFold с фиксированным параметром random_state=42. В качестве меры качества используйте
# долю верных ответов (accuracy).

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Найдите точность классификации на кросс-валидации для метода k ближайших соседей
# (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. При каком k получилось оптимальное качество?
# Чему оно равно (число в интервале от 0 до 1)? Данные результаты и будут ответами на вопросы 1 и 2.

param_test1 = {'n_neighbors': range(1,50)}
gsearch1 = GridSearchCV(estimator = KNeighborsClassifier(),
param_grid = param_test1, scoring='accuracy',n_jobs=4, cv=kf)

gsearch1.fit(X, y)

print(gsearch1.best_params_, gsearch1.best_score_)

# 5. Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale.
# Снова найдите оптимальное k на кросс-валидации.

X = scale(X)

gsearch1.fit(X, y)

# 6. Какое значение k получилось оптимальным после приведения признаков к одному масштабу?
# Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?

print(gsearch1.best_params_, gsearch1.best_score_)
