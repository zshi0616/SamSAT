import os 
import torch
import numpy as np 
import deepgate as dg 
import copy
import gym
from collections import deque
import shutil
import time

from config import get_parse_args
from rl.config import RL_Config
from rl.qnet import Q_Net
from rl.agent import Agent
from rl.trainer import Trainer
from rl.buffer import ReplayBuffer
from rl.env import solve_Env, env_map_solve

from utils.logger import Logger
from utils.aiger_utils import solve_aig
from utils.utils import run_command

PROBLEM_LIST = [
    'ad44', 'ab38'
    'ac3', 'ad14', 'ac1', 'ab18', 'h29'
]

# Baseline: AIG --> CNF --> Solver
# SAT07: AIG -->(abc) AIG -->(bMapper) LUT Netlist --> CNF --> Solver
# Our: AIG -->(abc+RL) AIG -->(cMapper) LUT Netlist --> CNF --> Solver 

def solve_sat07(arg, problem_path):
    tmp_path = os.path.join(arg.tmp_dir, 'tmp.aig')
    # cmd = 'abc -c \"&read {}; &syn2; &write {}\"'.format(problem_path, tmp_path)
    cmd = 'abc -c \"read_aiger {}; rewrite; balance; rewrite; write_aiger {}\"'.format(problem_path, tmp_path)
    _, syn_time = run_command(cmd)
    sat, map_time, solve_time, no_dec, no_var, no_clas = env_map_solve(arg, arg.baseline_mapper, tmp_path, arg.tmp_dir)
    os.remove(tmp_path)
    trans_time = syn_time + map_time
    
    return sat, trans_time, solve_time, no_dec, no_var, no_clas

if __name__ == '__main__':
    args = get_parse_args()
    config = RL_Config(args)
    # print('==> Using settings {}'.format(args))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test in single device: ', args.device)
    logger = Logger(args)
    
    # Create RL environments 
    net = Q_Net(args)
    model_last_path = os.path.join(args.save_dir, 'qnet_last.pth')
    net.load(model_last_path)
    print('==> Load model from {}'.format(model_last_path))
    
    rl_env = solve_Env(args=args, config=config, instance_list=PROBLEM_LIST, mode='test')
    agent = Agent(net, args, config)
    
    # Test 
    for problem_idx in range(len(PROBLEM_LIST)):
        rl_env.next_instance()
        obs = rl_env.reset()
        print(rl_env.problem_name)
        
        # Our
        done = False
        tot_reward = 0
        tot_q = 0
        no_act = 0
        model_time = 0 
        print('==> Testing: {:} / {:}, Problem: {}'.format(problem_idx, len(PROBLEM_LIST), rl_env.problem_name))
        while not done:
            if args.disable_rl:
                action = 999
                q_val = 0
            else:
                start_time = time.time()
                action, q_val = agent.act(obs, 0, mode='test')
                model_time += time.time() - start_time
            no_act += 1
            next_obs, reward, done, info = rl_env.step(action)
            tot_reward += reward
            tot_q += q_val
            print('Action: {}, Tot Q: {:.2f}, Tot Reward: {:.2f}'.format(action, tot_q, tot_reward))
            
        # Print
        info = rl_env.get_solve_info()
        logger.write('Circuit Name: {}'.format(rl_env.problem_name))
        logger.write('================== Our ==================')
        logger.write('Result: {}'.format(info['res']))
        md_time = info['md_mp'] + info['md_st'] + model_time + info['md_mt']
        logger.write('# Vars: {:}, # Clause: {:}'.format(info['md_nvars'], info['md_nclas']))
        logger.write('Model: {:.4f}, Transform: {:.4f}, Solve: {:.4f}, Overall: {:.4f}'.format(
            model_time + info['md_mt'], info['md_mp'], info['md_st'], md_time
        ))
        logger.write(' ')
        