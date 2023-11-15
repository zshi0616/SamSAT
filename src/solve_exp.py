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
    # 'ac3', 'ad14', 'ac1', 'ab18', 'h29'
    # 'php17-mixed-35percent-blocked', 'sat05-2534', 'brent_9_0', 
    # 'CNF_to_alien_11', 'apx_2_DC-ST',
    #'46bits_11', 
    # 'velev-pipe-o-uns-1-7', 
    # 'mchess16-mixed-25percent-blocked', 'mchess16-mixed-35percent-blocked', 'mchess18-mixed-35percent-blocked', 
    
    # 'brent_9_0', 'php17-mixed-35percent-blocked', 'apx_2_DC-ST', 
    # 'brent_13_0_1', 'brent_15_0_25', 'brent_69_0_3', 'brent_13_0_1', 
    'php16-mixed-35percent-blocked', 'php17-mixed-15percent-blocked'
    'SE_apx_0', 'apx_0', 'apx_2_DC-AD', 'apx_2_DS-ST', 
    '138_apx_2_DS-ST', 
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
        
        # Baseline
        _, _, baseline_time, baseline_nvars, baseline_nclas = solve_aig(rl_env.origin_problem, args.tmp_dir, args='--time={:}'.format(1000))
        print('Baseline: {:.4f}'.format(baseline_time))
        
        # SAT07 
        _, s7_trans_time, s7_solve_time, s7_no_dec, s7_no_var, s7_no_clas = solve_sat07(args, rl_env.origin_problem)
        print('SAT07: {:.4f}'.format(s7_trans_time + s7_solve_time))
        
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
        logger.write('# Vars: {:}, # Clause: {:}'.format(baseline_nvars, baseline_nclas))
        logger.write('Baseline: {:.4f}'.format(baseline_time))
        logger.write('Timeout: {:}'.format(args.max_solve_time))
        if baseline_time <= 0:
            continue
        logger.write('================== SAT07 ==================')
        s7_time = s7_trans_time + s7_solve_time
        logger.write('# Vars: {:}, # Clause: {:}'.format(s7_no_var, s7_no_clas))
        logger.write('#Dec: {:}, Transform: {:.4f}, Solve: {:.4f}, Overall: {:.4f}'.format(
            s7_no_dec, s7_trans_time, s7_solve_time, s7_time
        ))
        logger.write('Reduction: {:.2f}%'.format((baseline_time - s7_time) / baseline_time * 100))
        logger.write('================== Our ==================')
        md_time = info['md_mp'] + info['md_st'] + model_time + info['md_mt']
        logger.write('# Vars: {:}, # Clause: {:}'.format(info['md_nvars'], info['md_nclas']))
        logger.write('Model: {:.4f}, Transform: {:.4f}, Solve: {:.4f}, Overall: {:.4f}'.format(
            model_time + info['md_mt'], info['md_mp'], info['md_st'], md_time
        ))
        if s7_no_dec > 0:
            logger.write('SAT07 Decision Reduction: {:} -> {:} = {:.2f}%'.format(
                s7_no_dec, info['md_dec'], (s7_no_dec - info['md_dec']) / s7_no_dec * 100
            ))
        if baseline_time > 0:
            logger.write('Reduction: {:.2f}%'.format((baseline_time - md_time) / baseline_time * 100))
        logger.write(' ')
        