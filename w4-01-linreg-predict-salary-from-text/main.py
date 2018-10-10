
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

# 1. Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv.

train = pd.DataFrame()  # Create result data frame
chunks = pd.read_table('salary-train.csv', chunksize=1000, iterator=True, sep=',')
for chunk in chunks:  # For each part in parts
    train = pd.concat([train, chunk], axis=0)  # Join file parts

print(train.head())
print(train.info())

# 2. Проведите предобработку:
# Приведите тексты к нижнему регистру (text.lower()).
#
# Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова.
# Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text).
# Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты.

# for col in train.columns[:-1]:
train['FullDescription'] = train['FullDescription'].map(lambda x: x.lower() if isinstance(x, str) else x)
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

# Примените TfidfVectorizer для преобразования текстов в векторы признаков. Оставьте только те слова,
# которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).

vectorizer = TfidfVectorizer(min_df=5)
vectorizer.fit(train['FullDescription'])
X_train_text = vectorizer.transform(train['FullDescription'])

# Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.
# Код для этого был приведен выше

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

# Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime

enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объедините все полученные признаки в одну матрицу "объекты-признаки".
# Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
# Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack

X_train = hstack([X_train_text, X_train_categ])

# 3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241.
# Целевая переменная записана в столбце SalaryNormalized.

model = Ridge(alpha=1, random_state=241)
y_train = train['SalaryNormalized']
model.fit(X_train, y_train)

# 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.

test = pd.read_csv('salary-test-mini.csv', sep=',')
print(test.head())
X_test_text = vectorizer.transform(test['FullDescription'])
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_categ])

y_test = model.predict(X_test)
print(y_test)
