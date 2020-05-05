import numpy as np

from src.td_zero import TDZero


class QLearning(TDZero):
    def get_td_target(self, next_state, reward, epsilon):
        return reward + self.gamma * self.Q[next_state][np.argmax(self.Q[next_state])]
