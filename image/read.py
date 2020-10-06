import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import random
from sklearn.model_selection import train_test_split

train_idx = 0
npy_idx = 0
path = 'C:/Users/Administrator/Desktop/image/dest'
files = os.listdir(path)     #path指定的文件夹包含的文件或文件夹的名字的列表。
random.shuffle(files)        #用于将一个列表中的元素打乱
images = []
labels = []
for f in files: #目录下所有文件夹
    file_path = os.path.join(path, str(f)) + '//'    #连接两个或更多的路径名组件
    for root, dirs, files in os.walk(file_path):     #python遍历获取文件
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg':   #分离文件名与扩展名
                train_idx = train_idx + 1
                img_path = os.path.join(file_path, str(file))
                #print('img_path={}'.format(img_path))
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img)
                images.append(img_array)
                #labels.append(f)
images = np.array(images)   #（num, h, w, 3）
#labels = np.array(labels)   #(num, )
images /= 255
np.savez_compressed('sample.npy',images)

#x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)  #划分训练数据、训练标签、验证数据、验证标签
'''

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(img_h, img_h, 3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(n_class, activation='softmax')) #25分类

model.summary()
model.compile(loss=loss_func, optimizer=Adam(lr=0.0003), metrics=['accuracy'])

model.fit(x_train, y_train,
      batch_size=nbatch_size,
      epochs=nepochs,
      verbose=1,
      validation_data=(x_test, y_test))

yaml_string = model.to_yaml()
with open('./models/model_name.yaml', 'w') as outfile:
    outfile.write(yaml_string)
model.save_weights('./models/model_name.h5')
'''