from src.td_lambda import TDLambda
from src.q_learning import QLearning


class QLearningLambda(TDLambda):
    def get_td_target(self, *args, **kwargs):
        return QLearning.get_td_target(self, *args, **kwargs)
