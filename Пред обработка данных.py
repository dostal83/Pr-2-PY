
#Предварительная обработка данных

import numpy as np
from sklearn import preprocessing

input_data = np.array([[2.1, -1.9, 5.5],
                      [-1.5, 2.4, 3.5], 
                      [0.5, -7.9, 5.6], 
                      [5.9, 2.3, -5.8]])

# Применение методов предварительной обработки

# Бинар
data_binarized = preprocessing.Binarizer(threshold = 0.5).transform(input_data)
print("\tБинаризация\n", data_binarized)

# Мидл удаление
print("\n\tСреднее удаление\nmean = ", input_data.mean(axis = 0)) # Средние значение по сплюс массиву
print("Standart deviation = ", input_data.std(axis = 0)) # вычисляет среднеквадрат  отклонения

data_scaled = preprocessing.scale(input_data) # Метод preprocessing. scale() полезен при стандартизации точек данных. Разделить все точки данных на среднее значение и вычесть стандартное отклонение для каждой точки данных.
print("\nmean = ", data_scaled.mean(axis = 0))
print("standart deviation = ", data_scaled.std(axis = 0))

# Пересчет 
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1)) 
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data) # Вычисление -> преобразование
print ("\n\tПересчет\n", data_scaled_minmax)

data_normalized_l1 = preprocessing.normalize(input_data, norm = 'l1') # l1(норм)
print("\n\tL1 нормализация\n", data_normalized_l1)

data_normalized_l2 = preprocessing.normalize(input_data, norm = 'l2') # l2(норм)
print("\n\tL2 нормализация\n", data_normalized_l2)