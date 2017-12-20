#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

session = tf.Session()
#加载数据
iris = datasets.load_iris()
x_vals = np.array([[x[0],x[3]] for x in iris.data])
y_vals = np.array([ 1 if y == 0 else -1 for y in iris.target])

#分离训练集和测试集
test_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)