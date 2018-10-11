
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from  sklearn.metrics import log_loss
import math
import matplotlib.pyplot as plt

# 1. Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy
# (параметр values у датафрейма). В первой колонке файла с данными записано, была или нет реакция.
# Все остальные колонки (d1 - d1776) содержат различные характеристики молекулы, такие как размер, форма и т.д.
# Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами test_size = 0.8
# и random_state = 241.

df = pd.read_csv('gbm-data.csv')
X = df.loc[:, 'D1':'D1776'].values
y = df['Activity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

# 2. Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241 и для
# каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:


def sigmoid(y_pred):
    return 1.0 / (1.0 + math.exp(-y_pred))


loss_results = {}
for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=250, verbose=True, random_state=241)
    clf.fit(X_train, y_train)

    # Используйте метод staged_decision_function для предсказания качества на обучающей и
    # тестовой выборке на каждой итерации.
    #
    # Преобразуйте полученное предсказание с помощью сигмоидной функции по формуле 1 / (1 + e^{−y_pred}),
    #  где y_pred — предсказанное значение.

    train_loss = []
    for pred in clf.staged_decision_function(X_train):
        train_loss.append(log_loss(y_train, [sigmoid(y_pred) for y_pred in pred]))

    test_loss = []
    for pred in clf.staged_decision_function(X_test):
        test_loss.append(log_loss(y_test, [sigmoid(y_pred) for y_pred in pred]))

    # Вычислите и постройте график значений log-loss (которую можно посчитать с помощью функции
    # sklearn.metrics.log_loss) на обучающей и тестовой выборках, а также найдите минимальное значение
    # метрики и номер итерации, на которой оно достигается.

    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2, label='test')
    plt.plot(train_loss, 'g', linewidth=2, label='train')
    plt.legend(loc=0, fontsize='small')
    plt.savefig('rate_' + str(lr) + '.png')

    min_loss_value = min(test_loss)
    min_loss_index = test_loss.index(min_loss_value)

    loss_results[lr] = min_loss_value, min_loss_index

# 3. Как можно охарактеризовать график качества на тестовой выборке, начиная с некоторой итерации:
# переобучение (Как можно охарактеризовать график качества на тестовой выборке, начиная с некоторой итерации:
# переобучение (overfitting) или недообучение (underfitting)? В ответе укажите одно из слов overfitting либо
# underfitting.) или недообучение (underfitting)? В ответе укажите одно из слов overfitting либо underfitting.

# overfitting

# 4. Приведите минимальное значение log-loss на тестовой выборке и номер итерации,
# на котором оно достигается, при learning_rate = 0.2

print('min-log-loss = %s on iteration = %s with learning rate = 0.2' % (loss_results[0.2]))

# 5. На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций,
# на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта, c random_state=241
# и остальными параметрами по умолчанию. Какое значение log-loss на тесте получается у этого случайного леса?
# (Не забывайте, что предсказания нужно получать с помощью функции predict_proba. В данном случае брать
# сигмоиду от оценки вероятности класса не нужно)

clf = RandomForestClassifier(n_estimators=min_loss_index, random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
test_loss = log_loss(y_test, y_pred)
print(test_loss)
