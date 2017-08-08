from pandas import read_csv
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats.stats import pearsonr
import re
from matplotlib import pyplot as plt
import os

# data = read_csv(os.getcwd()+'/titanic.csv', index_col='PassengerId')
data = read_csv('titanic.csv', index_col='PassengerId')

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.

print("#1")
print(data['Sex'].value_counts())

# print(data[data.Sex=="male"].count()['Sex'])

# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).

surv = data[data['Survived'] == 1].count()['Survived'] / data.shape[0]
print("#2")
print(round(surv, 2))

surv_counts = data['Survived'].value_counts()
surv_percent = 100.0 * surv_counts[1] / surv_counts.sum()
print(2, "{:0.2f}".format(surv_percent))

print(data['Survived'].value_counts(normalize=True))

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).

fc = data[data['Pclass'] == 1].count()['Pclass'] / data.shape[0]
print("#3")
print(round(100*fc, 2))
print(data['Pclass'].value_counts(normalize=True))

# 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.

print("#4")
ages = data['Age'].dropna()
print("{:0.2f}".format(ages.mean()))
print(ages.median())

# 5. Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.

print("#5")
# CorrKoef = data.corr(method='pearson')
# CorField = []
# for i in CorrKoef:
#     for j in CorrKoef.index[CorrKoef[i] > 0.3]:
#         if i != j and j not in CorField and i not in CorField:
#             CorField.append(j)
#             print("%s-->%s: r^2=%f" % (i,j, CorrKoef[i][CorrKoef.index==j].values[0]))

print("{:0.2f}".format(data['SibSp'].corr(data['Parch'])))

print(np.corrcoef(data['SibSp'], data['Parch']))

print(pearsonr(data['SibSp'], data['Parch']))

# 6. Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name)
# его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию. Попробуйте вручную разобрать
# несколько значений столбца Name и выработать правило для извлечения имен, а также разделения их на женские и мужские.

print("#6")
def clean_name(name):
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)

    # Если есть скобки - то имя пассажира в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)

    # Удаляем обращения
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)

    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(' ')[0].replace('"', '')

    return name


names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()
print(name_counts.head(5))


# print(data.sort_values(by='Age', ascending=False).head())

print()
# print(data[(data['Survived']==1) & (data['Age'] > 30)][['Name', 'Age']].head())

# print(data.groupby(['Survived'])[['Age']].describe(percentiles=[]))

# print(pd.crosstab(data['Survived'], data['Pclass'], margins=True))

# print(data.pivot_table(['Survived', 'Age'], ['Pclass'], aggfunc='mean').head(10))



# cols = ['Survived', 'Pclass', 'Fare']
# sns_plot = sns.pairplot(data[cols])
# sns_plot.savefig('pairplot.png')

# sns.distplot(data.Fare)




# sns.plt.show()
# a = pd.DataFrame(data = [1,2,3], index = [1,2,3])
# b = pd.DataFrame(data = [4,5,6], index = [1,2,3])
#
# print(a[0].corr(b[0]))

