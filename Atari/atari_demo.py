import gym
import cv2
import numpy as np
from gym import wrappers
import convDQN
import random
from collections import deque
import tensorflow as tf

# while True :
# 	action = env.action_space.sample()
# 	state, reward, done, info = env.step(action)
# 	# gray_state = np.asarray(cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)) / 255.
# 	# gray_state = cv2.resize(gray_state, (160, 160))
# 	# plt.imshow(gray_state, cmap = 'gray')
# 	plt.show()

env = gym.make('Pong-v0')
input_size = env.observation_space.shape
output_size = env.action_space.n

GAMMA = 0.99
REPLAY_MEMORY = 50000
REWARD_COUNT = 10

def replay_train(mainDQN, targetDQN, train_batch) :
	x_stack = np.empty(0).reshape(0, mainDQN.input_size)
	y_stack = np.empty(0).reshape(0, mainDQN.output_size)

	for state, action, reward, next_state, done in train_batch :
		Q = mainDQN.predict(state)

		if done :
			Q[0, action] = reward
		else :
			Q[0, action] = reward + GAMMA * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

		y_stack = np.vstack([y_stack, Q])
		x_stack = np.vstack([x_stack, state])

	return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops(*, dest_scope_name = "target", src_scope_name = "main") :
	op_holder = []
	src_vars = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope_name)
	dest_vars = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)

	for src_var, dest_var in zip(src_vars, dest_vars) :
		op_holder.append(dest_var.assign(src_var.value()))
	return op_holder

def preprocess(state) :
	gray_state = np.asarray(cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)) / 255.
	resized_gray_state = cv2.resize(gray_state, (160, 160))
	return resized_gray_state

def main() :
	max_episode = 2000
	replay_buffer = deque()
	rewards_list = []

	with tf.Session() as sess :
		mainDQN = convDQN.ConvDQN(input_size, output_size, name = 'main')
		targetDQN = convDQN.ConvDQN(input_size, output_size, name = 'target')
		tf.global_variables_initializer().run()

		copy_ops = get_copy_var_ops(dest_scope_name = "target", src_scope_name = "main")
		sess.run(copy_ops)

		for episode in xrange(max_episode) :
			e = 1. / ((i / 10) + 1)
			state = env.reset()
			preprocessed_state = preprocess(state) 
	 		total_reward = 0

			while True :
				if random.rand(1) < e :
					action = env.action_space.sample()
				else :
					action = np.argmax(mainDQN.predict(preprocessed_state))

				new_state, reward, done, _ = env.step(action)
				preprocessed_new_state = preprocess(new_state)

				if done :
					reward = -100
				
				replay_buffer.append((state, action, reward, done, preprocessed_new_state))
				if len(replay_buffer) > REPLAY_MEMORY :
					replay_buffer.popleft()
				state = new_state
				total_reward += reward

				if done :
					break

				elif total_reward > 50000 :
					break

			print("Episode : {} total reward : {}".format(episode, total_reward))

			if len(rewards_list) > REWARD_COUNT :
				rewards_list.pop(0)

			if sum(rewards_list) / float(len(rewards_list)) > 50000 :
				print("Training is done at {}".format(episode))
				break

			if total_reward > 50000 :
				pass

			if episode % 20 == 1 :
				for _ in range(50) :
					minibatch = random.sample(replay_buffer, 20)
					loss, _ = replay_train(mainDQN, targetDQN, mnibatch)
				print("Loss : {}".format(loss))
				sess.run()

		saver = tf.train.Saver()
		save_path = saver.save(sess, "./saved_networks/CartPole-v0_model2", global_step = episode)
		print("Model saved in file : %s" % save_path)

if __name__ == '__main__' :
	main()






