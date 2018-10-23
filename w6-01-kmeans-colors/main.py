
from skimage import io, img_as_float
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import math

# 1. Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал от 0 до 1.
# Для этого можно воспользоваться функцией img_as_float из модуля skimage. Обратите внимание на этот шаг,
# так как при работе с исходным изображением вы получите некорректный результат.

image = img_as_float(io.imread('parrots.jpg'))
w, h, d = image.shape

# io.imshow(float_image)
# io.show()

# 2.Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями
# интенсивности в пространстве RGB.

df = pd.DataFrame(np.reshape(image, (w*h, d)), columns=['r', 'g', 'b'])
print(df.head())

# 3. Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241.
# После выделения кластеров все пиксели, отнесенные в один кластер, попробуйте заполнить двумя способами:
# медианным и средним цветом по кластеру.


def cluster_pixels(df, n_clusters):
    print('clusters = %s' % n_clusters)

    df = df.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    model.fit(df)
    df['cluster'] = model.predict(df)

    mean_values = df.groupby('cluster').mean().values
    mean_pixels = [mean_values[cluster] for cluster in df['cluster']]
    mean_image = np.reshape(mean_pixels, (w, h, d))

    median_values = df.groupby('cluster').median().values
    median_pixels = [median_values[cluster] for cluster in df['cluster']]
    median_image = np.reshape(median_pixels, (w, h, d))

    io.imsave('parrots_mean_' + str(n_clusters) + '.jpg', mean_image)
    io.imsave('parrots_median_' + str(n_clusters) + '.jpg', median_image)

    return mean_image, median_image


# 4. Измерьте качество получившейся сегментации с помощью метрики PSNR.
# Эту метрику нужно реализовать самостоятельно (см. определение)


def psnr(image1, image2):

    mse = np.mean((image1 - image2) ** 2)
    PIXEL_MAX = 1
    return 10 * math.log10(float(PIXEL_MAX ** 2) / mse)


# 5. Найдите минимальное количество кластеров, при котором значение PSNR выше 20 (можно рассмотреть не более
# 20 кластеров, но не забудьте рассмотреть оба способа заполнения пикселей одного кластера).
# Это число и будет ответом в данной задаче.

for n in range(1, 21):

    mean_image, median_image = cluster_pixels(df, n)
    mean_psrn, median_psrn = psnr(image, mean_image), psnr(image, median_image)

    if mean_psrn > 20 or median_psrn > 20:
        print(n)
        break
