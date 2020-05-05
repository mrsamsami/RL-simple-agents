import math
import time
from collections import deque

import gym
import numpy as np
import pandas as pd
from src import utils


class Agent:
    def observation_to_state(self, observation):
        raise NotImplementedError()

    def render(self):
        self.frames.append(self.env.render(mode='rgb_array'))

    def sample_act(self, state, epsilon):
        return np.random.choice(np.arange(self.env.action_space.n), p=self.get_act_policy(state, epsilon))

    def get_act_policy(self, state, epsilon):
        pi = np.full((self.env.action_space.n,), epsilon / self.env.action_space.n)
        pi[self.get_best_action(state)] += 1.0 - epsilon
        return pi

    def get_best_action(self, state):
        raise NotImplementedError()

    def get_parameter(self, t, min_value):
        return max(min_value, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def display(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.frames = []
        observation = self.env.reset()
        state = self.observation_to_state(observation)
        done = False
        while not done:
            self.render()
            action = self.get_best_action(state)
            next_observation, reward, done, _ = self.env.step(action)
            next_state = self.observation_to_state(next_observation)
            state = next_state

        utils.save_gif(self.frames, 'assets/gif/{name}.gif'.format(name=name))
        utils.display_gif('assets/gif/{name}.gif'.format(name=name))
        self.frames = []

    def before_episode(self):
        raise NotImplementedError()

    def during_episode(self, state, action, reward, next_state, epsilon, alpha):
        raise NotImplementedError()

    def after_episode(self, alpha):
        raise NotImplementedError()

    def _train(self, print_stats=True, window_size=10, *args, **kwargs):
        scores = []
        start = time.time()
        stats = []
        sliding_window = deque()

        for episode in range(self.n_episodes):
            observation = self.env.reset()
            state = self.observation_to_state(observation)
            alpha = self.get_parameter(episode, self.min_alpha)
            epsilon = self.get_parameter(episode, self.min_epsilon)
            done = False
            total_reward = 0

            self.before_episode()
            while not done:
                action = self.sample_act(state, epsilon)
                next_observation, reward, done, _ = self.env.step(action)
                next_state = self.observation_to_state(next_observation)

                self.during_episode(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    epsilon=epsilon,
                    alpha=alpha
                )

                state = next_state
                total_reward += reward

            self.after_episode(alpha=alpha)
            sliding_window.append(total_reward)
            if len(sliding_window) > window_size:
                sliding_window.popleft()

            sliding_window_mean = sum(sliding_window) / len(sliding_window)
            scores.append(sliding_window_mean)
            stats.append({'Episode': episode, 'Score': sliding_window_mean})

            if episode > 0 and episode % 100 == 0:
                mean_score = sum(scores) / len(scores)
                scores = []
                end = time.time()
                if print_stats:
                    print('Episode {:0=4d}: Average score = {}, total time = {}'
                          .format(episode + 1, mean_score, end - start))
                start = time.time()

                if mean_score >= self.final_score:
                    if print_stats:
                        print("Finished!")
                    return pd.DataFrame(stats)

        return pd.DataFrame(stats)

    def train(self,
              env,
              n_episodes=1000,
              final_score=190,
              min_alpha=0.1,
              min_epsilon=0.1,
              gamma=1,
              ada_divisor=25,
              *args, **kwargs,
              ):
        self.env = env
        self.n_episodes = n_episodes
        self.final_score = final_score
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.ada_divisor = ada_divisor
        return self._train(*args, **kwargs)
