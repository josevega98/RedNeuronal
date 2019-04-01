import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
x_data = np.random.randn(2000,3)
w_real = [0.4, 0.6, 0.2]
b_real = -0.3 
noise = np.random.randn(1,2000)*0.1 
y_data = np.matmul(w_real, x_data.T)+b_real+noise
num_iters = 10 
g = tf.Graph()
wb = []

with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name='W')
        b = tf.Variable(0, dtype=tf.float32, name='b')
        y_pred = tf.matmul(w, tf.transpose(x))+b
        
    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))
        
    with tf.name_scope('training') as scope:
        lr = 0.5
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)
        
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for step in range(num_iters):
            sess.run(train, {x:x_data, y_true:y_data})
            if(step%5==0):
                print(step, sess.run([w,b]))
                wb.append(sess.run([w,b]))
                
        print(10, sess.run([w,b]))
def sigmoid(x):
    return 1/(1+np.exp(-x))

x_data = np.random.randn(20000,3)
w_real = [0.4, 0.6, 0.2]
b_real = -0.3
wb = np.matmul(w_real,x_data.T)+b_real

y_data_bef_noise = sigmoid(wb)
y_data = np.random.binomial(1, y_data_bef_noise)
num_iters = 50
g = tf.Graph()
wb = []

with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name='W')
        b = tf.Variable(0, dtype=tf.float32, name='b')
        y_pred = tf.matmul(w, tf.transpose(x))+b
        
    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss = tf.reduce_mean(loss)
        
    with tf.name_scope('training') as scope:
        lr = 0.5
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)
        
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for step in range(num_iters):
            sess.run(train, {x:x_data, y_true:y_data})
            if(step%5==0):
                print(step, sess.run([w,b]))
                wb.append(sess.run([w,b]))
                
        print(50, sess.run([w,b]))
datadir = '/data'
num_iters = 1000
minibatch_size = 100
data = input_data.read_data_sets(datadir, one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(num_iters):
        batch_xs, batch_ys = data.train.next_batch(minibatch_size)
        sess.run(optimizer, feed_dict={x:batch_xs, y_true:batch_ys})
        
    testing = sess.run(accuracy, feed_dict={x:data.test.images, y_true:data.test.labels})
    
print('Accuracy: {:.4}%'.format(testing*100))