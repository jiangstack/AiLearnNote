from tensorflow.keras.preprocessing import image
import os, sys
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

filepath = sys.argv[1]
print(filepath)
model = keras.models.load_model('cats_and_dogs_small_2.h5')

#读取图像并调整大小
# PIL.Image.Image
img = image.load_img(filepath, target_size=(150, 150),)

# 将其转换为形状 (150, 150, 3) 的 Numpy 数组
x = image.img_to_array(img)

x = np.multiply(x,1./255)
x = x.reshape((1,) + x.shape)

# print(train_generator.class_indices)
# {'cats': 0, 'dogs': 1}

# 预测得到的值为每个类别的概率值，而不是类别
res = model.predict(x)
print("图片结果是",res)
# print(np.argmax(res, axis=1))
plt.imshow(img)
plt.text(10,10,'Dog Probability :'+str(res[0][0]*100)+'%',color='r')
plt.show()

# os.system('pause')