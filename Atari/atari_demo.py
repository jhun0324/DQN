import gym
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('Pong-v0')
env.reset()

while True :
	action = env.action_space.sample()
	print(action)
	state, reward, done, info = env.step(action)
	gray_state = np.asarray(cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)) / 255.
	gray_state = cv2.resize(gray_state, (110, 84))
	gray_state = gray_state[:, 12:96]
	print(gray_state.shape)
	plt.imshow(gray_state, cmap = 'gray')
	plt.show()








