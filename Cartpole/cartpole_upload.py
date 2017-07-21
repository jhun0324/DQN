import dqn
import gym
import tensorflow as tf
from gym import wrappers
import numpy as np


def each_play(mainDQN, monitor) :
	state = monitor.reset()
	reward_sum = 0
	while True :
		monitor.render()
		action = np.argmax(mainDQN.predict(state))
		state, reward, done, _ = monitor.step(action)
		reward_sum += reward

		if done :
			print("Total score : {}".format(reward_sum))
			break
	return reward_sum

def play(submit) :
	env = gym.make('CartPole-v0')
	monitor = wrappers.Monitor(env, 'gym-results', force = True)
	input_size = env.observation_space.shape[0]
	output_size = env.action_space.n
	
	with tf.Session() as sess :
		mainDQN = dqn.DQN(sess, input_size, output_size, name = "main")

		saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")
		
		# #First let's load meta graph and restore weights
		# saver = tf.train.import_meta_graph('CartPole-v0_model.ckpt')
		# saver.restore(sess,tf.train.latest_checkpoint('./'))

		# saver = tf.train.Saver()
	 #    	sess.run(tf.initialize_all_variables())
		
		reward_list = []

		for i in range(200) :
			reward = each_play(mainDQN, monitor)
			reward_list.append(reward)

		reward_avg = sum(reward_list) / float(len(reward_list))

		print("Average reward is {}".format(reward_avg))
		monitor.close()
		
		if  reward_avg > 195 :
			if submit :
				gym.upload('gym-results', api_key = "sk_NsESxpGkRPqUt9hoDiHZag")
				print("The result is uploaded")
			else :
				print("Average reward is good enough but do not want to submit it")
		else :
			print("Average reward is too small. Cannot be uploaded")


if __name__ == '__main__' :
	submit = False
	play(submit)