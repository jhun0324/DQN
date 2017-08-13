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
	(thresh, gray_state) = cv2.threshold(gray_state, 0.40, 1., cv2.THRESH_BINARY)
	print(gray_state)
	gray_state = cv2.resize(gray_state, (80, 105))
	gray_state = gray_state[17:97, :]
	print(gray_state.shape)
	plt.imshow(gray_state, cmap = 'gray')
	plt.show()








