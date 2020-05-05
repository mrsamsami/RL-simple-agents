from src.td_lambda import TDLambda
from src.sarsa import SARSA


class SARSALambda(TDLambda):
    def get_td_target(self, *args, **kwargs):
        return SARSA.get_td_target(self, *args, **kwargs)
