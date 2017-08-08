import numpy as np

from sklearn.model_selection import KFold, GridSearchCV

import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

# 1. Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект, у которого признаки записаны в поле data,
# а целевой вектор — в поле target.

Boston = load_boston()

X = Boston['data']
y = Boston['target']

# 2. Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.

X = scale(X)

# 3. Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, чтобы всего
# было протестировано 200 вариантов (используйте функцию numpy.linspace). Используйте KNeighborsRegressor
# с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса, зависящие от расстояния
# до ближайших соседей. В качестве метрики качества используйте среднеквадратичную ошибку
# (параметр scoring='mean_squared_error' у cross_val_score; при использовании библиотеки scikit-learn
# версии 0.18.1 и выше необходимо указывать scoring='neg_mean_squared_error'). Качество оценивайте,
# как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42, не забудьте включить
# перемешивание выборки (shuffle=True).

params = {'p': np.linspace(1, 10, 200) }

kf = KFold(n_splits=5, shuffle=True, random_state=42)


gsearch1 = GridSearchCV(estimator = KNeighborsRegressor(n_neighbors=5, weights='distance'),
param_grid = params, scoring='neg_mean_squared_error', cv=kf)

gsearch1.fit(X, y)

# 4. Определите, при каком p качество на кросс-валидации оказалось оптимальным. Обратите внимание,
# что cross_val_score возвращает массив показателей качества по блокам; необходимо максимизировать среднее
# этих показателей. Это значение параметра и будет ответом на задачу.

print(gsearch1.best_params_, gsearch1.best_score_)


