import numpy as np
import os
import tensorflow as tf

_D = os.path.dirname(__file__)
print(_D)

# データの読込

filenames = ["fujitou_happy_001.wav", "tsuchiya_happy_001.wav"]

fujitou = np.load('{}/mfcc/{}.npy'.format(_D, filenames[0]))
tsuchiya = np.load('{}/mfcc/{}.npy'.format(_D, filenames[1]))

X = np.vstack((fujitou, tsuchiya))
Y = np.vstack((np.zeros((fujitou.shape[0], 1)), np.ones((tsuchiya.shape[0], 1))))
print(X)
print(Y)

w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b) # matmul = 掛け算

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 計算開始
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(200):
  sess.run(train_step, feed_dict={
    x: X,
    t: Y
  })

# 分類
classified = correct_prediction.eval(session=sess, feed_dict={
  x: X,
  t: Y
})
print(classified)

prob = y.eval(session=sess, feed_dict={
  x: X,
  t: Y
})
print(prob)
print('w: ', sess.run(w))
print('b: ', sess.run(b))
