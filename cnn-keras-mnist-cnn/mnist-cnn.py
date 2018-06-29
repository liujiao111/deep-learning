'''
使用kears完成MNIST手写体识别任务
'''

import numpy as np

f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
print('训练数据集样本数： %d ,标签个数 %d ' % (len(x_train), len(y_train)))
print('测试数据集样本数： %d ,标签个数  %d ' % (len(x_test), len(y_test)))

print(x_train.shape)
print(x_test.shape)


#特征值缩放

x_train = x_train / 255
x_test = x_test / 255


#输出标签进行one-hot编码
from keras.utils import np_utils

print('Integer-valued labels:')
print(y_train[:10])

#标签进行one-hot编码
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print('One-hot labels:')
print(y_train[:10])


#定义模型架构：
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu',input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

#编译模型
model.compile(loss = 'categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#训练模型
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape)
print(x_test.shape)
model.fit(x_train, y_train, batch_size=128, epochs = 10, verbose=1, validation_data=(x_test, y_test))

#评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


