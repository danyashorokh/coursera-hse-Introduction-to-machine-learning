from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score


# 1. Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.

train = pd.read_csv("perceptron-train.csv",header=None)
test = pd.read_csv("perceptron-test.csv",header=None)

y_train = train[0]
X_train = train.loc[:,1:]

y_test = test[0]
X_test = test.loc[:,1:]

# 2. Обучите персептрон со стандартными параметрами и random_state=241.

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

# 3. Подсчитайте качество (долю правильно классифицированных объектов, accuracy)
# полученного классификатора на тестовой выборке.

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 4. Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.

clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, y_train)

predictions_scaled = clf.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, predictions_scaled)

# 6. Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
# Это число и будет ответом на задание.

diff = accuracy_scaled - accuracy

print(diff)


