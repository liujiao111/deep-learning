
# coding: utf-8

# 1.加载MNIST数据集

# In[27]:

import numpy as np
f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']  
x_test, y_test = f['x_test'], f['y_test']  
f.close() 
print('训练数据集样本数： %d ,标签个数 %d ' % (len(x_train), len(y_train)))
print('测试数据集样本数： %d ,标签个数  %d ' % (len(x_test), len(y_test)))

print(x_train.shape)
print(y_train[0])


# 2.可视化前六张图片

# In[28]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib.cm as cm
import numpy as np

fig = plt.figure(figsize = (20, 20))
for i in range(6):
    ax = fig.add_subplot(1, 6,i + 1, xticks = [], yticks = [])
    ax.imshow(x_train[i], cmap = 'gray')
    ax.set_title(str(y_train[i]))


# 3.每张图片都是28*28像素组成的，我们可以查看一张图片的像素构成细节

# In[29]:

def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(x_train[0], ax)


# 4.特征值缩放：将每个像素除以255

# In[30]:

x_train = x_train.astype('float') / 255
x_test = x_test.astype('float') / 255


# 5.对输出标签进行One-hot编码

# In[31]:

from keras.utils import np_utils

print('Integer-valued labels:')
print(y_train[:10])

#标签进行one-hot编码
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print('One-hot labels:')
print(y_train[:10])


# 6.定义模型架构

# In[33]:

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten

model = Sequential()
model.add(Flatten(input_shape = x_train.shape[1:]))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 7.编译模型

# In[34]:

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# 8.训练模型之前在测试集上看分类精确度

# In[35]:

score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]
print('Test accuracy: %.4f%%' % accuracy)


# 9.训练模型

# In[ ]:

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath = 'mnist.model.best.hdf5',verbose=1, save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=128, epochs=10,
          validation_split=0.2, callbacks=[checkpointer],
          verbose=1, shuffle=True)


# 10.加载训练好的模型，并在测试集上进行测试准确率

# In[38]:

model.load_weights('mnist.model.best.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100 * score[1]
print('Test accuracy: %.4f%%' % accuracy)


# In[ ]:



