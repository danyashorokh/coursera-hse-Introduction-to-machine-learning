
import pandas as pd

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, \
    precision_score, recall_score, f1_score, precision_recall_curve


# 1. Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true)
# и ответы некоторого классификатора (колонка pred).

df = pd.read_csv('classification.csv')
print(df.head())

# 2. Заполните таблицу ошибок классификации

tn, fp, fn, tp = confusion_matrix(df.true, df.pred).ravel()
print(tp, fp, fn, tn)

# 3. Посчитайте основные метрики качества классификатора:
# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
# Precision (точность) — sklearn.metrics.precision_score
# Recall (полнота) — sklearn.metrics.recall_score
# F-мера — sklearn.metrics.f1_score

accuracy = (tp + tn)/(tn + fp + fn + tp)
precision = tp/(tp + fp)
recall = tp/(tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(accuracy, precision, recall, f1)
print(accuracy_score(df.true, df.pred), precision_score(df.true, df.pred),
      recall_score(df.true, df.pred), f1_score(df.true, df.pred))

# 4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и
# значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).

scores = pd.read_csv('scores.csv')
print(scores.head())

# 5. Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее значение
# метрики AUC-ROC (укажите название столбца)? Воспользуйтесь функцией sklearn.metrics.roc_auc_score.

for model in scores.columns[1:]:
    print('%s = %s' % (model, roc_auc_score(scores.true, scores[model])))

# 6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?

for model in scores.columns[1:]:
    data = precision_recall_curve(scores.true, scores[model])
    model_df = pd.DataFrame({'precision': data[0], 'recall': data[1]})
    print('%s = %s' % (model, model_df[model_df.recall >= 0.7]['precision'].max()))
