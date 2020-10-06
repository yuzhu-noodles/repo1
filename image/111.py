import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image

max_CD = np.loadtxt("max_CD.txt")
train_idx = 0
path = 'C:/Users/Administrator/Desktop/dest'
files = os.listdir(path)
#random.shuffle(files)  # 打乱数据集图片顺序
images = []
for f in files:
    train_idx = train_idx + 1
    img_path = os.path.join(path, str(f))
    img = load_img(img_path, target_size=None)
    img_array = img_to_array(img)
    images.append(img_array)

image = np.array(images)
image /= 255
np.savez_compressed('CD',image=image,max_CD=max_CD)   #保存+压缩


'''
image=load_img('1.jpeg')
array=img_to_array(image)
np.savez_compressed('sample.npy',array)
#np.save('sample.npy',array)
print(array)'''