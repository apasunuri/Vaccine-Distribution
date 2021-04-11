import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.table = deque(maxlen=4000)
        self.lr = 0.0001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_minimum = 0.01
        self.exploration_decay = 0.995
        self.model = tf.keras.Sequential()
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='relu'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))

    def action(self, state):
        if(np.random.rand() <= self.exploration_rate):
            return random.randrange(self.action_size)
        values = self.model.predict(state)
        return np.argmax(values[0])

    def add_as(self, state, action, reward, next_state, done):
        self.table.append((state, action, reward, next_state, done))

    def play(self, batch_size):
        if(len(self.table) < batch_size):
            return
        batch = random.sample(self.table, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            prediction = self.model.predict(state)
            prediction[0][action] = target
        if(self.exploration_rate > self.exploration_minimum):
            self.exploration_rate *= self.exploration_decay


class Environment:
    def __init__(self, batch_size, iterations):
        self.batch_size = batch_size
        self.iterations = iterations
        self.environment = gym.make('CartPole-v1')
        self.state_size = self.environment.observation_space.shape[0]
        self.action_size = self.environment.action_space.n
        self.agent = Agent(self.state_size, self.action_size)

    def run(self):
        for iteration in range(self.iterations):
            state = self.environment.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.environment.render()
                action = self.agent.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.agent.append_as(state, action, reward, next_state, done)
                state = next_state
                i += 1
            print('Episode {}# Score: {}'.format(iteration, i + 1))
            self.agent.replay(self.batch_size)


if __name__ == '__main__':
    environment = Environment(32, 100)
    # environment.run()
