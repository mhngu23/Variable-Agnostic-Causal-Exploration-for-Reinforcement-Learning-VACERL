from gymnasium.core import Env
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium import Env, spaces, utils

import numpy as np

class UpdatedFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, render_mode: str | None = None, desc=None, map_name="4x4", is_slippery=False, intrinsic_reward = {}):
        super().__init__(render_mode, desc, map_name, is_slippery)
        self.intrinsic_reward = intrinsic_reward
    
    def step(self, a):
        intrinsic_reward = self.intrinsic_reward.get((self.s, a), 0)
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        # if self.render_mode == "human":
        #     self.render()
        return (int(s), r + intrinsic_reward, t, False, {"prob": p})
    
    
