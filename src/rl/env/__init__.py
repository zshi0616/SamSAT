from .solve_env import solve_Env
from gym.envs.registration import register

register(id='SIMSAT-v0', entry_point='env.solve_env:solve_Env')