import numpy as np
import gym
import random
import os 
import sys
import glob
import shutil
import deepgate as dg

from utils.utils import run_command
from utils.cnf_utils import kissat_solve
from utils.lut_utils import parse_bench_cnf
import utils.circuit_utils as circuit_utils
from utils.aiger_utils import aig_to_xdata, xdata_to_cnf

IGNORE_LIST = []
PROBLEM_LIST = ['a8', 'd28']

def env_map_solve(args, aig_filename, tmp_dir):
    # Map to LUTs
    ckt_name = aig_filename.split('/')[-1].split('.')[0]
    bench_filename = os.path.join(tmp_dir, '{}.bench'.format(ckt_name))
    cmd_maplut = '{} {} {}'.format(
        args.customized_mapper, aig_filename, bench_filename
    )
    _, maplut_runtime = run_command(cmd_maplut, args.max_solve_time)
    if maplut_runtime < 0:
        return -1, 0, 0, 0, 0
    
    # Parse LUTs
    cnf, no_var = parse_bench_cnf(bench_filename)
    os.remove(bench_filename)
    
    # Solve 
    if maplut_runtime > args.max_solve_time:
        sat_res = -1 
        solvetime = 0
    else:
        sat_res, asg, solvetime = kissat_solve(cnf, no_var, args='--time={:}'.format(int(args.max_solve_time - maplut_runtime)))
        if solvetime < args.min_solve_time: 
            sat_res = -1
    
    return sat_res, maplut_runtime, solvetime, no_var, len(cnf)

class solve_Env(gym.Env):
    def __init__(
        self,
        args 
    ):
        self.args = args
        self.problem_list = []
        self.step_ntk_filepath = self.args.step_ntk_filepath
        for aig_path in glob.glob(os.path.join(args.Problem_AIG_Dir, '*.aiger')):
            aig_name = os.path.basename(aig_path).split('.')[0]
            if aig_name in PROBLEM_LIST:
                self.problem_list.append(aig_path)
        self.parser = dg.AigParser()
                
    def reset(self):
        curr_problem = random.choice(self.problem_list)
        shutil.copyfile(curr_problem, self.step_ntk_filepath)
        
        # Initialize 
        init_cmd = 'abc -c \"read_aiger {}; rewrite -lz; balance; rewrite -lz; balance; rewrite -lz; balance; write_aiger {}; \"'.format(
            self.step_ntk_filepath, self.step_ntk_filepath
        )
        _, _ = run_command(init_cmd)
        self.graph = self.parser.read_aiger(curr_problem)
        
        # Baseline
        self.sat, maplut_time, solve_time, nvars, nclas = env_map_solve(self.args, self.step_ntk_filepath, self.args.tmp_dir)
        self.baseline_time = maplut_time + solve_time
        self.baseline_trans_time = maplut_time
        self.baseline_solve_time = solve_time
        self.baseline_nvars = nvars
        self.baseline_nclas = nclas
        self.step_cnt = 0
        self.action_time = 0
        return self.graph
        
    def step(self, action):
        if action < 5 and self.step_cnt < self.args.max_step:
            if action == 0:
                action_str = 'rewrite -lz'
            elif action == 1:
                action_str = 'balance'
            elif action == 2:
                action_str = 'renode; strash'
            elif action == 3:
                action_str = 'refactor'
            elif action == 4:
                action_str = 'multi; strash'
            elif action == 5:
                action_str = 'strash'
            # strash  
            # mockturtle XMG,... 
            action_cmd = 'abc -c \"read_aiger {}; {}; write_aiger {}; \"'.format(
                self.step_ntk_filepath, action_str, self.step_ntk_filepath
            )
            _, action_runtime = run_command(action_cmd)
            self.action_time += action_runtime
            self.graph = self.parser.read_aiger(self.step_ntk_filepath)
            
            reward = 0
            done = False
        
        else:
            sat_res, maplut_runtime, solvetime, nvars, nclas = env_map_solve(self.args, self.step_ntk_filepath, self.args.tmp_dir)
            self.action_time += maplut_runtime
            self.synmap_nvars = nvars
            self.synmap_nclas = nclas
            reward = self.baseline_time - (solvetime + self.action_time)
            done = True
            
        self.step_cnt += 1
        info = {}
        return self.graph, reward, done, info
            
        
            
