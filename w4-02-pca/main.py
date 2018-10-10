
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 1. Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии торгов
# за каждый день периода.

df = pd.read_csv('close_prices.csv')
print(df.head())

# 2. На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
# Скольких компонент хватит, чтобы объяснить 90% дисперсии?

X = df.drop(['date'], axis=1)
pca = PCA(n_components=10)
pca.fit(X)

n = d = 0
while d <= 0.9:
    d += pca.explained_variance_ratio_[n]
    n += 1

print('Объяснения %.2f процентов дисперсии хватит %s компонент' % (d, n))

# 3. Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.

df_comp = pd.DataFrame(pca.transform(X))
first_comp = df_comp[0]

# 4. Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?

djia = pd.read_csv('djia_index.csv')
print(djia.head())
print('corr = %.2f' % first_comp.corr(djia['^DJI']))

# 5. Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.

first_comp_w = pd.Series(pca.components_[0])
first_comp_w.sort_values(ascending=False, inplace=True)
print(X.columns[first_comp_w.index[0]])
