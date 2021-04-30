import gym
import math
import copy
import random
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env


# Model is based on and modified from the VacSim Paper
# https://arxiv.org/pdf/2009.06602.pdf

class DistributionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, county_count, iterations, total_vaccines, susceptible, exposed, symptomatic_infection, asymptomatic_infection, recovered, pathogen, bucket):
        self.county_count = county_count
        self.observation_space = spaces.Box(
            np.zeros(6), np.ones(6), shape=(6,), dtype=np.float32)
        self.bucket = bucket
        self.action_space = spaces.Discrete(bucket)
        self.step_count = 1
        self.done = False
        self.values = np.zeros((self.county_count, 100))
        self.total_vaccines = total_vaccines
        self.iterations = iterations
        self.received_vaccines = [0 for i in range(self.county_count)]
        self.states = []
        self.actions = None
        self.susceptibles = [0 for i in range(self.county_count)]
        self.susceptible = susceptible
        self.exposed = exposed
        self.symptomatic_infection = symptomatic_infection
        self.asymptomatic_infection = asymptomatic_infection
        self.recovered = recovered
        self.pathogen = pathogen
        self.reset()

    def reset(self):
        self.step_count = 1
        self.done = False
        self.total_vaccines = 1000000
        self.states = np.array([self.susceptible, self.exposed,
                                self.symptomatic_infection, self.asymptomatic_infection, self.recovered, self.pathogen])
        return copy.deepcopy(self.states)

    def step(self, action):
        self.actions = action
        reward = self.reward()
        if(self.step_count == self.iterations):
            self.done = True
        else:
            self.done = False
            self.step_count += 1
        extra_info = {
            'actions': self.actions,
            'vaccine_distribution': (self.actions * (1 / self.bucket)) + (0.5 / self.bucket),
            'iteration_number': self.step_count
        }
        return self.states, self.done, reward, extra_info

    def reward(self):
        vaccine_proportion = self.actions * (1 / self.bucket)
        susceptible_proportion = self.states[4]
        reward = (
            100 * math.exp((-(vaccine_proportion - susceptible_proportion) ** 2) / 0.0001))
        return reward

    def close(self):
        pass


# Alachua County, FL, Mercer County, OH, Suffolk County, VA
if __name__ == '__main__':
    df = pd.read_csv('seir-data/combined-seir-1.csv')
    actions = []
    rewards = []
    for i in range(0, len(df), 3):
        day_actions = []
        day_rewards = []
        print(i)
        array = df[i:i + 3].to_numpy()
        print(array)
        total_susceptible = sum(array[:, 2])
        total_exposed = sum(array[:, 3])
        total_symptomatic_infection = sum(array[:, 5])
        total_asymptomatic_infection = sum(array[:, 4])
        total_recovered = sum(array[:, 6])
        total_pathogen = sum(array[:, 7])
        for j in range(3):
            print(j)
            susceptible = array[j][2] / total_susceptible
            exposed = array[j][3] / total_exposed
            symptomatic_infection = array[j][5] / total_symptomatic_infection
            asymptomatic_infection = array[j][4] / total_asymptomatic_infection
            recovered = array[j][6] / (total_recovered + 0.01)
            pathogen = array[j][7] / total_pathogen
            env = DistributionEnv(1, 1, 1000000, susceptible, exposed,
                                  symptomatic_infection, asymptomatic_infection, recovered, pathogen, 200)
            nn_model = DQN('MlpPolicy', env, learning_rate=1e-3,
                           prioritized_replay=True, verbose=1)
            nn_model.learn(total_timesteps=int(1e4), log_interval=10000)
            observation = env.reset()
            action, states = nn_model.predict(observation)
            observation, done, reward, info = env.step(action)
            day_actions.append(action)
            day_rewards.append(reward)
            print(action)
        actions.append(day_actions)
        rewards.append(day_rewards)
    distributions = []
    for action in actions:
        s = sum(action)
        l = []
        for a in action:
            l.append(a / s)
        distributions.append(l)
    print(actions)
    print(distributions)
    print(rewards)
