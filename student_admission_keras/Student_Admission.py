import numpy as np
import pandas as pd
import keras


data = pd.read_csv('student_data.csv')
#print(data.head())  #admit    gre   gpa  rank
#print(data.shape)  #(400, 4)
#print(data['gre'])

import matplotlib.pyplot as plt
def plot_points(data):
    X = np.array(data[['gre', 'gpa']])
    y = np.array(data['admit'])
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color = 'red', edgecolors = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grada (GPA)')
#plot_points(data)
#plt.show()

from keras.utils import np_utils

def procee_data(data) :
    '''
        数据预处理
    :param data:
    :return:
    '''

    #移除NAN的数据
    data = data.fillna(0)

    #对rank字段进行one-hot编码
    processed_data = pd.get_dummies(data, columns=['rank'])

    #归一化字段gre   gpa
    processed_data['gre'] = processed_data['gre'] / 800
    processed_data['gpa'] = processed_data['gpa'] / 4.0

    return processed_data

data = procee_data(data)
#print(data.head())

#将数据切分为特征和标签
X = np.array(data)[:, 1:]
X = X.astype('float32')
y = keras.utils.to_categorical(data['admit'], 2)

#print(X.shape)
#print(y.shape)
#print(X[:10])
#print(y[:10])

def split_train_test(X, y):
    '''
    数据切分为训练集和测试集
    :param X:
    :param y:
    :return:
    '''
    (X_train, X_test) = X[50:], X[:50]
    (y_train, y_test) = y[50:], y[:50]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test(X, y)
#print('x_train shape:', X_train.shape)

# print number of training, validation, and test images
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')



import keras
from keras.models import Sequential
from keras.layers.core import  Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

def train_model_define(X_train, y_train):
    '''
    定义训练模型
    :return:
    '''
    model = Sequential()
    model.add(Dense(128, input_dim=7))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    #编译模型
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()


    #训练数据
    model.fit(X_train, y_train, epochs=200, batch_size=100, verbose=0)

    #查看训练结果
    score = model.evaluate(X_train, y_train)
    print("\n Training Accuracy:", score[1])
    score = model.evaluate(X_test, y_test)
    print("\n Testing Accuracy:", score[1])


train_model_define(X_train, y_train)

'''
    可以尝试修改的参数：
        激活函数: relu and sigmoid
        损失函数选择: categorical_crossentropy, mean_squared_error
        优化器: rmsprop, adam, ada
'''