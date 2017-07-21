import gym
from gym.envs.registration import register
import numpy as np
import tensorflow as tf
import random 
import dqn
from collections import deque


# Register CartPole with user-defined max_episode_steps
register(
	id='CartPole-v2',
	entry_point='gym.envs.classic_control:CartPoleEnv',
	tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
	reward_threshold=10000.0
)
env = gym.make('CartPole-v2')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.98
REPLAY_MEMORY = 50000
count_size = 10

def replay_train(mainDQN, targetDQN, train_batch) :
	x_stack = np.empty(0).reshape(0, mainDQN.input_size)
	y_stack = np.empty(0).reshape(0, mainDQN.output_size)

	for state, action, reward, next_state, done in train_batch :
		Q = mainDQN.predict(state)

		if done :
			Q[0, action] = reward
		else :
			Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

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



def main() :
	max_episodes = 2000
	replay_buffer = deque()
	step_list = []

	with tf.Session() as sess :
		mainDQN = dqn.DQN(sess, input_size, output_size, name = "main")
		targetDQN = dqn.DQN(sess, input_size, output_size, name = "target")
		tf.global_variables_initializer().run()

		copy_ops = get_copy_var_ops(dest_scope_name = "target", src_scope_name = "main")
		sess.run(copy_ops)

		for episode in range(max_episodes) :
			e = 1. / ((episode / 10) + 1)
			done = False 
			step_count = 0
			state = env.reset()

			while not done :
				if np.random.rand(1) < e :
					action = env.action_space.sample()
				else :
					action = np.argmax(mainDQN.predict(state))

				next_state, reward, done, _ = env.step(action)

				if done :
					reward = -100

				replay_buffer.append((state, action, reward, next_state, done))
				if len(replay_buffer) > REPLAY_MEMORY :
					replay_buffer.popleft()

				state = next_state
				step_count += 1

				if step_count > 10000 :
					break

			step_list.append(step_count)

			if len(step_list) > count_size :
				step_list.pop(0)

			print("Episode : {} steps : {}".format(episode, step_count))

			if sum(step_list) / float(len(step_list)) >= 10000 :
				print("Training is done at {}".format(episode))
				break

			if step_count > 10000 :
				pass

			if episode % 10 == 1 :
				for _ in range(50) :
					minibatch = random.sample(replay_buffer, 10)
					loss, _  = replay_train(mainDQN, targetDQN, minibatch)
				print("Loss : ", loss)
				sess.run(copy_ops)

		saver = tf.train.Saver()
		save_path = saver.save(sess, "./saved_networks/CartPole-v0_model2", global_step = episode)
		print("Model saved in file : %s" % save_path)

if __name__ == "__main__" :
	main()	














