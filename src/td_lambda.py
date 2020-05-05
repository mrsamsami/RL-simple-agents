import numpy as np

from src.tabular import Tabular


class TDLambda(Tabular):
    def get_td_target(self, next_state, reward, epsilon):
        raise NotImplementedError()

    def before_episode(self):
        pass

    def during_episode(self, state, action, reward, next_state, epsilon, alpha):
        self._E = self._lambda_val * self.gamma * self._E
        self._E[state][action] += 1
        td_target = self.get_td_target(next_state=next_state, reward=reward, epsilon=epsilon)
        td_delta = td_target - self.Q[state][action]
        self.Q += alpha * td_delta * self._E

    def after_episode(self, alpha):
        pass

    def train(self,
              env,
              lambda_val=0.5,
              n_X=1,
              n_X_dot=1,
              n_theta=6,
              n_theta_dot=12,
              *args, **kwargs):
        self._E = np.zeros((n_X, n_X_dot, n_theta, n_theta_dot) + (env.action_space.n,))
        self._lambda_val = lambda_val
        return super().train(
              env,
              n_X=n_X,
              n_X_dot=n_X_dot,
              n_theta=n_theta,
              n_theta_dot=n_theta_dot,
              *args, **kwargs)
