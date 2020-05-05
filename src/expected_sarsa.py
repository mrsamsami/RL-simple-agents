import numpy as np

from src.td_zero import TDZero


class ExpectedSARSA(TDZero):
    def get_td_target(self, next_state, reward, epsilon):
        return reward + self.gamma * np.dot(self.get_act_policy(next_state, epsilon), self.Q[next_state])

