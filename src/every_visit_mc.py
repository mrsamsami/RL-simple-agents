import numpy as np

from src.tabular import Tabular


class EveryVisitMC(Tabular):
    def before_episode(self):
        self._G = np.zeros(self.Q.shape)
        self._replay_buffer = []

    def during_episode(self, state, action, reward, *args, **kwargs):
        self._replay_buffer.append((state, action))
        for i, (replay_state, replay_action) in enumerate(self._replay_buffer, start=1):
            self._G[replay_state][replay_action] += reward * self.gamma ** (len(self._replay_buffer) - i)

    def after_episode(self, alpha):
        self.Q += alpha * (self._G - self.Q)
