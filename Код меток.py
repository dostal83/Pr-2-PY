# Кодирование меток

import numpy as np
from sklearn import preprocessing

input_labels = ['red','black','red','green','black','yellow','white'] 

encoder = preprocessing.LabelEncoder() 

print(encoder.fit(input_labels)) 

test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels) 
print("\nLabels =", test_labels)

print("Encoded values =", list(encoded_values))

#декод случайного набора числел
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values) 
print("\nEncoded values =", encoded_values)

print("Decoded labels =", list(decoded_list))
# декод значения