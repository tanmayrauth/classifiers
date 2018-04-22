import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random


# Reading the dataset
def read_dataset():
    df = pd.read_csv("sonar.csv")
    X = df[df.columns[0:60]].values
    y1 = df[df.columns[60]]
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)

    # Return
    return (X, Y, y1)


# Define the encoder function.
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


X, Y, y1 = read_dataset()

model_path = "./model/"
learning_rate = 0.3
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = 60
n_class = 2

# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
y_ = tf.placeholder(tf.float32, [None, n_class])


# Define the model
def neural_network(x):
    layer_1 = tf.add(tf.matmul(x, hidden_1['weight']), hidden_1['bias'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2['weight']), hidden_2['bias'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3['weight']), hidden_3['bias'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, hidden_4['weight']), hidden_4['bias'])
    layer_4 = tf.nn.relu(layer_4)

    out = tf.add(tf.matmul(layer_4, output['weight']), output['bias'])

    return out


hidden_1 = {'weight' : tf.Variable(tf.random_normal([n_dim, n_hidden_1])),
          'bias' : tf.Variable(tf.random_normal([n_hidden_1]))}
hidden_2 = {'weight' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
          'bias' : tf.Variable(tf.random_normal([n_hidden_2]))}
hidden_3 = {'weight' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
          'bias' : tf.Variable(tf.random_normal([n_hidden_3]))}
hidden_4 = {'weight' : tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
          'bias' : tf.Variable(tf.random_normal([n_hidden_4]))}
output =  {'weight' : tf.Variable(tf.random_normal([n_hidden_4, n_class])),
          'bias' : tf.Variable(tf.random_normal([n_class]))}
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Call your model defined
y = neural_network(x)

# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, model_path)

prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print (accuracy_run)
print('******************************************************')
print(" 0 Stands for M i.e. Mine & 1 Stands for R i.e. Rock")
print('******************************************************')
for i in range(93, 101):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 60)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1, 60), y_: Y[i].reshape(1, 2)})
    print("Original Class : ", y1[i], " Predicted Values : ", prediction_run[0], " Accuracy : ", accuracy_run)


# print(sess.run(prediction, feed_dict={x: x_test}))
# print(sess.run(accuracy,  feed_dict={x: x_test, y_: y_test}))
