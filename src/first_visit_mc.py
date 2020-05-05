import numpy as np

from src.tabular import Tabular


class FirstVisitMC(Tabular):
    def before_episode(self):
        self._G = np.zeros(self.Q.shape)
        self._count = np.zeros(self.Q.shape)
        self._visited = np.zeros(self.Q.shape, bool)

    def during_episode(self, state, action, reward, *args, **kwargs):
        self._visited[state][action] = True
        self._G += self._visited * reward * self.gamma ** self._count
        self._count += self._visited

    def after_episode(self, alpha):
        self.Q += self._visited * alpha * (self._G - self.Q)
