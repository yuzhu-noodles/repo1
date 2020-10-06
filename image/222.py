
import numpy as np
from sklearn.model_selection import train_test_split
data = np.load('CD.npz')
print(data.files)
image = data['image']
max_CD = data['max_CD']
print(image)
print(max_CD)
x_train, x_test, y_train, y_test = train_test_split(image, max_CD, test_size=0.2)  # 随机划分数据集
#print(x_train.shape, y_train.shape)
'''

import numpy as np
data = np.load('boston_housing.npz')
print(data.files )
y = data['y']
print(y)
x = data['x']
print(x)'''