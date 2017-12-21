#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

session = tf.Session()
#加载数据
iris = datasets.load_iris()
x_vals = np.array([[x[0],x[3]] for x in iris.data]) #数据
y_vals = np.array([ 1 if y == 0 else -1 for y in iris.target])  #标签

#分离训练集和测试集
train_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)  #训练集索引
test_indices = np.array(list(set(range(len(x_vals)))-set(train_indices)))   #测试集索引

x_vals_train = x_vals(train_indices)
x_vals_test = x_vals(test_indices)
y_vals_train = y_vals(train_indices)
y_vals_test = y_vals(test_indices)

#定义模型
batch_size=100
#初始化feedin
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
