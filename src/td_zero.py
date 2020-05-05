
from src.tabular import Tabular


class TDZero(Tabular):
    def get_td_target(self, next_state, reward, epsilon):
        raise NotImplementedError()

    def before_episode(self):
        pass

    def during_episode(self, state, action, reward, next_state, epsilon, alpha):
        td_target = self.get_td_target(next_state=next_state, reward=reward, epsilon=epsilon)
        self.Q[state][action] += alpha * (td_target - self.Q[state][action])

    def after_episode(self, alpha):
        pass
