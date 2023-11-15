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
from rl.env import solve_Env

PROBLEM_LIST = [
    # 'ac3', 'ab18', 'h29', 
    # 'ad14', 'ac1', 
    # 'mult_op_DEMO1_11_11_TOP13', 'mult_op_DEMO1_11_11_TOP14', 
    # 'mult_op_DEMO1_12_12_TOP17', 'mult_op_DEMO1_11_11_TOP12', 
    # 'mult_op_DEMO1_12_12_TOP13'
    'mult_op_DEMO1_13_13_TOP11'
]

if __name__ == '__main__':
    args = get_parse_args()
    config = RL_Config(args)
    # print('==> Using settings {}'.format(args))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test in single device: ', args.device)
    
    # Create RL environments 
    net = Q_Net(args)
    model_last_path = os.path.join(args.save_dir, 'qnet_last.pth')
    net.load(model_last_path)
    print('==> Load model from {}'.format(model_last_path))
    rl_env = solve_Env(args=args, instance_list=PROBLEM_LIST, mode='test')
    agent = Agent(net, args, config)
    
    # Test 
    for problem_idx in range(len(PROBLEM_LIST)):
        obs = rl_env.reset()
        done = False
        tot_reward = 0
        tot_q = 0
        no_act = 0
        print('==> Testing: {:} / {:}, Problem: {}'.format(problem_idx, len(PROBLEM_LIST), rl_env.problem_name))
        while not done:
            action, q_val = agent.act(obs, 0, mode='test')
            print('Action: {}, Q_Val: {:.4f}'.format(action, q_val))
            no_act += 1
            next_obs, reward, done, info = rl_env.step(action)
            tot_reward += reward
            tot_q += q_val
            
        # Print
        info = rl_env.get_solve_info()
        print('==> Problem: {}, Reward: {:.4f}, Q: {:.4f}'.format(
            rl_env.problem_name, tot_reward, tot_q
        ))
        bl_time = info['bl_mp'] + info['bl_st']
        md_time = info['md_mp'] + info['md_st']
        print('Baseline: #Var: {:}, #Cls: {:}, Dec: {:}, Trans. {:.2f}s, Solve: {:.2f}s, All: {:.2f}s'.format(
            info['bl_nvars'], info['bl_nclas'], info['bl_dec'], info['bl_mp'], info['bl_st'], bl_time
        ))
        print('Agent: #Var: {:}, #Cls: {:}, Dec: {:}, Trans. {:.2f}s, Solve: {:.2f}s, All: {:.2f}s'.format(
            info['md_nvars'], info['md_nclas'], info['md_dec'], info['md_mp'], info['md_st'], md_time
        ))
        print('Decision Reduction: {:.2f} -> {:.2f} = {:.2f}%'.format(
            info['bl_dec'], info['md_dec'], (info['bl_dec'] - info['md_dec']) / info['bl_dec'] * 100
        ))
        print('Runtime Reduction: {:.2f} -> {:.2f} = {:.2f}%'.format(
            bl_time, md_time, (bl_time - md_time) / bl_time * 100
        ))
        print()
        