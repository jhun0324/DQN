import tensorflow as tf
import numpy as np


class ConvDQN :
	def __init__(self, sess, input_size, output_size, learning_rate = 5e-2, name = 'main') :
		self.session = sess
		self.input_size = input_size
		self.output_size = output_size
		self.net_name = name
		self.learning_rate = learning_rate
		self._build_network()


	def _build_network(self) :
		with tf.variable_scope(self.net_name) :
			W_conv1 = weight_variable([8, 8, 4, 32])
			b_conv1 = bias_variable([32])

			W_conv2 = weight_variable([5, 5, 32, 64])
			b_conv2 = bias_variable([64])

			W_conv3 = weight_variable([2, 2, 64, 64])
			b_conv3 = bias_variable([64])

			W_fc1 = weight_variable([1600, 512])
			b_fc1 = bias_variable([512])

			W_fc2 = weight_variable(shape = [512, self.output_size])
				
			self._X = tf.placeholder(shape = [None, self.input_size, self.input_size, 4], dtype = tf.float32)
	    
			# hidden layers
			h_conv1 = tf.sigmoid(conv2d(self._X, W_conv1, 4) + b_conv1)
			h_pool1 = max_pool_2x2(h_conv1)


			h_conv2 = tf.sigmoid(conv2d(h_pool1, W_conv2, 2) + b_conv2)
			h_pool2 = max_pool_2x2(h_conv2)


			h_conv3 = tf.sigmoid(conv2d(h_pool2, W_conv3, 1) + b_conv3)

			h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
			h_fc1 = tf.sigmoid(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

			self._Qpred = tf.matmul(h_fc1, W_fc2)

		self._Y = tf.placeholder(shape = [None, self.output_size], dtype = tf.float32)
		self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
		self._train = tf.train.AdamOptimizer(
			learning_rate = self.learning_rate).minimize(self._loss)

	def predict(self, states) :
		x = np.reshape(states, [1, self.input_size, self.input_size, 4])
		return self.session.run(self._Qpred, feed_dict = {self._X : x})

	def update(self, x_stack, y_stack) :
		return self.session.run([self._loss, self._train],
			feed_dict = {self._X : x_stack, self._Y : y_stack})

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")











