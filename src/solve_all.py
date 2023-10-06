import os 
import torch
import numpy as np 
import deepgate as dg 
import copy
import gym
from collections import deque
import shutil

from config import get_parse_args
from rl.config import RL_Config
from rl.qnet import Q_Net
from rl.agent import Agent
from rl.trainer import Trainer
from rl.buffer import ReplayBuffer
from rl.env import solve_Env, env_map_solve

from utils.logger import Logger
from utils.aiger_utils import solve_aig

PROBLEM_LIST = [
    # 'ac3', 'ab18', 'h29', 
    # 'ad14', 'ac1', 
    # 'mult_op_DEMO1_11_11_TOP13', 'mult_op_DEMO1_11_11_TOP14', 
    # 'mult_op_DEMO1_12_12_TOP17', 'mult_op_DEMO1_11_11_TOP12', 
    # 'mult_op_DEMO1_12_12_TOP13'

    'mchess16-mixed-25percent-blocked', 'mchess16-mixed-35percent-blocked', 'mchess16-mixed-45percent-blocked'
]

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
    
    rl_env = solve_Env(args=args, instance_list=PROBLEM_LIST, mode='test')
    agent = Agent(net, args, config)
    
    # Test 
    for problem_idx in range(len(PROBLEM_LIST)):
        # 1 - Customized-Mapping
        obs = rl_env.reset()
        
        # 2 - Raw AIG
        _, _, ra_time, ra_nvars, ra_nclas = solve_aig(rl_env.origin_problem, args.tmp_dir, args='--time={:}'.format(args.max_solve_time))
        
        # 3 - Raw Mapping 
        _, rm_mt, rm_st, rm_dec, rm_nvars, rm_nclas = env_map_solve(args, args.baseline_mapper, rl_env.origin_problem, args.tmp_dir)
        
        # 4 - RL  
        done = False
        tot_reward = 0
        tot_q = 0
        no_act = 0
        print('==> Testing: {:} / {:}, Problem: {}'.format(problem_idx, len(PROBLEM_LIST), rl_env.problem_name))
        while not done:
            action, q_val = agent.act(obs, 0, mode='test')
            no_act += 1
            next_obs, reward, done, info = rl_env.step(action)
            tot_reward += reward
            tot_q += q_val
            
        # Print
        info = rl_env.get_solve_info()
        logger.write('Circuit Name: {}'.format(rl_env.problem_name))
        logger.write('# Vars: {:}, # Clause: {:}'.format(ra_nvars, ra_nclas))
        logger.write('Baseline: {:.4f}'.format(ra_time))
        logger.write('Timeout: {:}'.format(args.max_solve_time))
        logger.write('================== Baseline Map ==================')
        rm_all_time = rm_mt + rm_st
        logger.write('# Vars: {:}, # Clause: {:}'.format(rm_nvars, rm_nclas))
        logger.write('Transform: {:.4f}, Solve: {:.4f}, Overall: {:.4f}'.format(
            rm_mt, rm_st, rm_all_time
        ))
        if ra_time != 0:
            logger.write('Reduction: {:.2f}%'.format((ra_time - rm_all_time) / ra_time * 100))
        logger.write('================== Customized Map ==================')
        bl_time = info['bl_mp'] + info['bl_st']
        logger.write('# Vars: {:}, # Clause: {:}'.format(info['bl_nvars'], info['bl_nclas']))
        logger.write('Transform: {:.4f}, Solve: {:.4f}, Overall: {:.4f}'.format(
            info['bl_mp'], info['bl_st'], bl_time
        ))
        if ra_time != 0:
            logger.write('Reduction: {:.2f}%'.format((ra_time - bl_time) / ra_time * 100))
        logger.write('================== RL + Map ==================')
        md_time = info['md_mp'] + info['md_st']
        logger.write('# Vars: {:}, # Clause: {:}'.format(info['md_nvars'], info['md_nclas']))
        logger.write('Transform: {:.4f}, Solve: {:.4f}, Overall: {:.4f}'.format(
            info['md_mp'], info['md_st'], md_time
        ))
        if ra_time != 0:
            logger.write('Reduction: {:.2f}%'.format((ra_time - md_time) / ra_time * 100))
        logger.write(' ')
        