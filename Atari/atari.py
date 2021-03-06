import gym
import cv2
import numpy as np
from gym import wrappers
import convDQN
import random
from collections import deque
import tensorflow as tf
import pickle



env = gym.make('Pong-v0')
input_size = 80
output_size = 6

GAMMA = 0.98
REPLAY_MEMORY = 50000
REWARD_COUNT = 10
load = True

def replay_train(mainDQN, targetDQN, train_batch) :
	x_stack = np.empty(0).reshape(0, mainDQN.input_size, mainDQN.input_size, 4)
	y_stack = np.empty(0).reshape(0, mainDQN.output_size)
	for states, action, reward, done, next_states in train_batch :
		Q = mainDQN.predict(states)

		if done :
			Q[0, action] = reward
		else :
			Q[0, action] = reward + GAMMA * targetDQN.predict(next_states)[0, np.argmax(mainDQN.predict(next_states))]

		states = np.reshape(states, [1, mainDQN.input_size, mainDQN.input_size, 4])
		y_stack = np.vstack([y_stack, Q])
		x_stack = np.vstack([x_stack, states])

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
	gray_state = gray_state.astype(np.uint8)
	gray_state = cv2.threshold(gray_state, 0.40, 1., cv2.THRESH_BINARY)[1]
	resized_gray_state = cv2.resize(gray_state, (80, 105))
	modified_state = resized_gray_state[17:97, :]
	return modified_state

def update_e(training_number) :
	result = max(0.1, -(0.9 / 1000000) * training_number + 1)
	return 0.0

def main() :
	max_episode = 100000
	replay_buffer = deque()
	rewards_list = []
	if not load :
		training_number = 0
		pickle.dump(training_number, open("./saved_networks/training_number.p", "wb"))
	training_number = pickle.load(open("./saved_networks/training_number.p", "rb"))

	with tf.Session() as sess :
		try :
			mainDQN = convDQN.ConvDQN(sess, input_size, output_size, name = 'main')
			targetDQN = convDQN.ConvDQN(sess, input_size, output_size, name = 'target')
			
			saver = tf.train.Saver()
			tf.global_variables_initializer().run()
			checkpoint = tf.train.get_checkpoint_state("saved_networks")
			if load and checkpoint and checkpoint.model_checkpoint_path:
				saver.restore(sess, checkpoint.model_checkpoint_path)
				print("Successfully loaded:", checkpoint.model_checkpoint_path)
			else:
				print("Could not find old network weights")
			
			copy_ops = get_copy_var_ops(dest_scope_name = "target", src_scope_name = "main")
			sess.run(copy_ops)
			e = 1.0

			for episode in range(max_episode) :
				state = env.reset()
				state = preprocess(state) 
				states = np.stack((state, state, state, state), axis = 2)
				total_reward = 0
				num_random_move = 0.0
				num_game_played = 0

				while True :
					if np.random.rand(1) < e :
						action = env.action_space.sample()
						num_random_move += 1
					else :
						action = np.argmax(mainDQN.predict(states))

					new_state, reward, done, _ = env.step(action)
					new_state = np.reshape(preprocess(new_state), (80, 80, 1))
					new_states = np.append(new_state, states[:,:,:3], axis = 2)

					if reward != 0 :
						num_game_played += 1

					replay_buffer.append((states, action, reward, done, new_states))
					if len(replay_buffer) > REPLAY_MEMORY :
						replay_buffer.popleft()
					states = new_states
					total_reward += reward
					e = update_e(training_number)

					if done :
						break

					elif total_reward > 20 :
						break

				print("Episode : {} total reward : {} random move {}".format(episode, total_reward, num_random_move/num_game_played))
				training_number += 1
				rewards_list.append(total_reward)

				if len(rewards_list) > REWARD_COUNT :
					rewards_list.pop(0)

				if sum(rewards_list) / float(len(rewards_list)) > 20 :
					print("Training is done at {}".format(episode))
					break

				if total_reward > 20 :
					pass

				if training_number > REPLAY_MEMORY and episode % 20 == 0 :
					total_loss = 0
					for _ in range(50) :
						minibatch = random.sample(replay_buffer, 32)
						loss, _ = replay_train(mainDQN, targetDQN, minibatch)
						total_loss += loss
					print("Loss : {}".format(total_loss))
					sess.run(copy_ops)

				if episode != 0 and episode % 500 == 0 :
					save_path = saver.save(sess, "./saved_networks/Pong-v0")
					pickle.dump(training_number, open("./saved_networks/training_number.p", "wb"))
					print("Model saved in file : %s" % save_path)
					print("Total training number : %s" % training_number)

		except KeyboardInterrupt :
			print("KeyboardInterrupt occured. Saving the model...")
			save_path = saver.save(sess, "./saved_networks/Pong-v0")
			pickle.dump(training_number, open("./saved_networks/training_number.p", "wb"))
			print("Model saved in file : %s" % save_path)
			print("Total training number : %s" % training_number)
			env.close()

if __name__ == '__main__' :
	main()






