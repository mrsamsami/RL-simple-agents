from src.td_zero import TDZero


class SARSA(TDZero):
    def get_td_target(self, next_state, reward, epsilon):
        return reward + self.gamma * self.Q[next_state][self.sample_act(next_state, epsilon)]
