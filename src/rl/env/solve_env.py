import numpy as np
import gym
import random
import os 
import sys
import glob
import shutil
import deepgate as dg

from utils.utils import run_command
from utils.cnf_utils import kissat_solve, kissat_solve_dec
from utils.lut_utils import parse_bench_cnf
import utils.circuit_utils as circuit_utils
from utils.aiger_utils import aig_to_xdata, xdata_to_cnf

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
        no_dec = 0
    else:
        sat_res, asg, solvetime, no_dec = kissat_solve_dec(args.kissat_path, cnf, no_var, args='--time={:}'.format(int(args.max_solve_time - maplut_runtime)))
    
    return sat_res, maplut_runtime, solvetime, no_dec, no_var, len(cnf)

class solve_Env(gym.Env):
    def __init__(
        self,
        args 
    ):
        if args.debug:
            PROBLEM_LIST = ['b5', 'f6']
        else:
            PROBLEM_LIST = ["b27", "mult_op_DEMO1_11_11_TOP17", "aa31", "mult_op_DEMO1_12_12_TOP11", "ac21", "ad18", "h22", "d19", "aa23", "h17", "ad27", "ab19", "ac5", "h14", "aa18", "f2", "ad22", "mult_op_DEMO1_10_10_TOP13", "c1", "b31", "e21", "h2", "ac26", "ad28", "b29", "ad29", "mult_op_DEMO1_10_10_TOP12", "ac17", "f22", "ad5", "mult_op_DEMO1_13_13_TOP24", "d1", "h23", "ab24", "c8", "ab3", "f23", "ac4", "aa14", "aa3", "aa2", "aa24", "mult_op_DEMO1_10_10_TOP14", "d9", "ab26", "ac27", "e20", "mult_op_DEMO1_12_12_TOP21", "f1", "ad4", "ac16", "ab25", "e1", "h1", "ab14", "ab32", "c11", "h10", "d12", "h13", "ab11", "ac13", "e12", "mult_op_DEMO1_11_11_TOP18", "aa11", "aa25", "ad13", "b28", "f8", "ac28", "aa10", "e9", "d21", "aa28", "f9", "ad17", "h27", "mult_op_DEMO1_10_10_TOP11", "d24", "d23", "ac31", "e8", "f26", "f29", "e24", "d8", "d25", "h31", "b1", "h24", "ab29", "ab2", "e27", "h9", "c19", "f24", "f12", "c7", "c22", "ab10", "mult_op_DEMO1_10_10_TOP15", "ac12", "c16", "aa32", "ad32", "mult_op_DEMO1_12_12_TOP22", "c21", "h25", "ad12", "mult_op_DEMO1_13_13_TOP25", "e25", "d17", "mult_op_DEMO1_13_13_TOP10", "e22", "aa26", "ac35", "ab33", "c14", "ab27", "b8", "h28", "mult_op_DEMO1_11_11_TOP10", "f27", "c27", "e18", "mult_op_DEMO1_9_9_TOP12", "c23", "d29", "f19", "b11", "ac29", "mult_op_DEMO1_12_12_TOP10", "ac24", "ac32", "a9", "mult_op_DEMO1_10_10_TOP10", "ad36", "aa29", "ad33", "aa21", "a17", "b21", "ac22", "ab30", "h20", "b19", "mult_op_DEMO1_9_9_TOP11", "ad25", "mult_op_DEMO1_11_11_TOP19", "aa36"]

        self.args = args
        self.problem_list = []
        self.step_ntk_filepath = self.args.step_ntk_filepath
        for aig_path in glob.glob(os.path.join(args.Problem_AIG_Dir, '*.aiger')):
            aig_name = os.path.basename(aig_path).split('.')[0]
            if aig_name in PROBLEM_LIST:
                self.problem_list.append(aig_path)
        self.parser = dg.AigParser()
                
    def reset(self):
        while True:
            curr_problem = random.choice(self.problem_list)
            problem_name = curr_problem.split('/')[-1].split('.')[0]
            shutil.copyfile(curr_problem, self.step_ntk_filepath)
            # Initialize 
            init_cmd = 'abc -c \"read_aiger {}; rewrite -lz; balance; rewrite -lz; balance; rewrite -lz; balance; write_aiger {}; \"'.format(
                self.step_ntk_filepath, self.step_ntk_filepath
            )
            _, _ = run_command(init_cmd)
            self.graph = self.parser.read_aiger(curr_problem)
            
            # Baseline
            self.sat, map_time, solve_time, no_dec, nvars, nclas = env_map_solve(self.args, self.step_ntk_filepath, self.args.tmp_dir)
            if self.sat != -1 and len(self.graph.POs) == 1:
                break

        self.baseline_dec = no_dec
        self.baseline_nvars = nvars
        self.baseline_nclas = nclas
        self.step_cnt = 0
        self.action_time = 0
        self.problem_name = problem_name
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
            sat_res, map_time, solve_time, no_dec, nvars, nclas = env_map_solve(self.args, self.step_ntk_filepath, self.args.tmp_dir)
            self.synmap_nvars = nvars
            self.synmap_nclas = nclas
            reward = (self.baseline_dec - no_dec) / self.baseline_dec
            done = True
            
        self.step_cnt += 1
        info = {}
        return self.graph, reward, done, info
            
        
            
