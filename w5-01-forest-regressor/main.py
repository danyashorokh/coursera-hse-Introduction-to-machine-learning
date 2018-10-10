
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

# 1. Загрузите данные из файла abalone.csv. Это датасет, в котором требуется предсказать возраст ракушки
# (число колец) по физическим измерениям.

df = pd.read_csv('abalone.csv')
print(df.head())

# 2. Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
# Если вы используете Pandas, то подойдет следующий код: data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M'
# else (-1 if x == 'F' else 0))

df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# 3. Разделите содержимое файлов на признаки и целевую переменную.
# В последнем столбце записана целевая переменная, в остальных — признаки.

y = df['Rings']
X = df.drop('Rings', axis=1)

# 4. Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев: от 1 до 50
# (не забудьте выставить "random_state=1" в конструкторе). Для каждого из вариантов оцените качество работы
# полученного леса на кросс-валидации по 5 блокам. Используйте параметры "random_state=1" и "shuffle=True"
# при создании генератора кросс-валидации sklearn.cross_validation.KFold. В качестве меры качества воспользуйтесь
# коэффициентом детерминации (sklearn.metrics.r2_score).

scores = []
k_fold = KFold(n_splits=5, shuffle=True, random_state=1)

for n in range(1, 51):
    model = RandomForestRegressor(n_estimators=n, random_state=1)
    scores.append(np.mean(cross_val_score(model, X, y, cv=k_fold, scoring='r2')))

# 5. Определите, при каком минимальном количестве деревьев случайный лес показывает качество
# на кросс-валидации выше 0.52. Это количество и будет ответом на задание.

for i, score in enumerate(scores):
    if score > 0.52:
        print(i + 1)
        break

# 6. Обратите внимание на изменение качества по мере роста числа деревьев. Ухудшается ли оно?

plt.plot(scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()
plt.savefig('estimators_score.png')
