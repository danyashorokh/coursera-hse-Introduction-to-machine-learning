
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold
# https://ru.wikipedia.org/wiki/TF-IDF

# 1. Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"
# (инструкция приведена выше). Обратите внимание, что загрузка данных может занять несколько минут

# import os
# import sys
# os.environ["https_proxy"] = "https://user:password@proxy_name:8080"

newsgroups = datasets.fetch_20newsgroups(data_home=sys.path[0] + '/', subset='all',
                                         categories=['alt.atheism', 'sci.space'])
X = newsgroups.data
y = newsgroups.target

# 2. Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам
# вычислить TF-IDF по всем данным. При таком подходе получается, что признаки на обучающем множестве используют
# информацию из тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем значения
# целевой переменной из теста. На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки
# известны на момент обучения, и поэтому можно ими пользоваться при обучении алгоритма

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(X)

# 3. Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM
# с линейным ядром (kernel='linear') при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241
# и для SVM, и для KFold. В качестве меры качества используйте долю верных ответов (accuracy).

params = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, param_grid=params, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(X), y)

# 4. Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.

C = gs.best_params_['C']
clf = SVC(C=C, kernel='linear', random_state=241)
clf.fit(vectorizer.transform(X), y)

# 5. Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC).
# Они являются ответом на это задание. Укажите эти слова через запятую или пробел, в нижнем регистре,
# в лексикографическом порядке.

words_mapping = vectorizer.get_feature_names()
data = pd.DataFrame(clf.coef_.data, clf.coef_.indices, columns=['coef'])
data['abs_coef'] = abs(data['coef'])
data['word'] = data.index.map(lambda i: words_mapping[i])
data = data.sort_values(by='abs_coef', ascending=False)
print(data.head(15))

top_words = sorted(list(data.head(10)['word']))
print(top_words)
