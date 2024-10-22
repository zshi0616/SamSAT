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

def env_map_solve(args, mapper, aig_filename, tmp_dir):
    # Map to LUTs
    ckt_name = aig_filename.split('/')[-1].split('.')[0]
    bench_filename = os.path.join(tmp_dir, '{}.bench'.format(ckt_name))
    cmd_maplut = '{} {} {}'.format(
        mapper, aig_filename, bench_filename
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
        sat_res, asg, solvetime, no_dec = kissat_solve_dec(args.kissat_path, cnf, no_var, tmp_dir, args='--time={:}'.format(int(args.max_solve_time - maplut_runtime)))
    
    return sat_res, maplut_runtime, solvetime, no_dec, no_var, len(cnf)

class solve_Env(gym.Env):
    def __init__(
        self,
        args, 
        instance_list = [], 
        mode='train'
    ):
        if len(instance_list) == 0:
            if args.debug:
                self.instance_list = ['b5', 'f6']
            else:
                # 10s - 50s
                # self.instance_list = ["b27", "mult_op_DEMO1_11_11_TOP17", "aa31", "mult_op_DEMO1_12_12_TOP11", "ac21", "ad18", "h22", "d19", "aa23", "h17", "ad27", "ab19", "ac5", "h14", "aa18", "f2", "ad22", "mult_op_DEMO1_10_10_TOP13", "c1", "b31", "e21", "h2", "ac26", "ad28", "b29", "ad29", "mult_op_DEMO1_10_10_TOP12", "ac17", "f22", "ad5", "mult_op_DEMO1_13_13_TOP24", "d1", "h23", "ab24", "c8", "ab3", "f23", "ac4", "aa14", "aa3", "aa2", "aa24", "mult_op_DEMO1_10_10_TOP14", "d9", "ab26", "ac27", "e20", "mult_op_DEMO1_12_12_TOP21", "f1", "ad4", "ac16", "ab25", "e1", "h1", "ab14", "ab32", "c11", "h10", "d12", "h13", "ab11", "ac13", "e12", "mult_op_DEMO1_11_11_TOP18", "aa11", "aa25", "ad13", "b28", "f8", "ac28", "aa10", "e9", "d21", "aa28", "f9", "ad17", "h27", "mult_op_DEMO1_10_10_TOP11", "d24", "d23", "ac31", "e8", "f26", "f29", "e24", "d8", "d25", "h31", "b1", "h24", "ab29", "ab2", "e27", "h9", "c19", "f24", "f12", "c7", "c22", "ab10", "mult_op_DEMO1_10_10_TOP15", "ac12", "c16", "aa32", "ad32", "mult_op_DEMO1_12_12_TOP22", "c21", "h25", "ad12", "mult_op_DEMO1_13_13_TOP25", "e25", "d17", "mult_op_DEMO1_13_13_TOP10", "e22", "aa26", "ac35", "ab33", "c14", "ab27", "b8", "h28", "mult_op_DEMO1_11_11_TOP10", "f27", "c27", "e18", "mult_op_DEMO1_9_9_TOP12", "c23", "d29", "f19", "b11", "ac29", "mult_op_DEMO1_12_12_TOP10", "ac24", "ac32", "a9", "mult_op_DEMO1_10_10_TOP10", "ad36", "aa29", "ad33", "aa21", "a17", "b21", "ac22", "ab30", "h20", "b19", "mult_op_DEMO1_9_9_TOP11", "ad25", "mult_op_DEMO1_11_11_TOP19", "aa36"]
                # self.instance_list = ["b27", "aa31", "ac21", "ad18", "h22", "d19", "aa23", "h17", "ad27", "ab19", "ac5", "h14", "aa18", "f2", "ad22", "c1", "b31", "e21", "h2", "ac26", "ad28", "b29", "ad29", "ac17", "f22", "ad5", "d1", "h23", "ab24", "c8", "ab3", "f23", "ac4", "aa14", "aa3", "aa2", "aa24", "d9", "ab26", "ac27", "e20", "f1", "ad4", "ac16", "ab25", "e1", "h1", "ab14", "ab32", "c11", "h10", "d12", "h13", "ab11", "ac13", "e12", "aa11", "aa25", "ad13", "b28", "f8", "ac28", "aa10", "e9", "d21", "aa28", "f9", "ad17", "h27", "d24", "d23", "ac31", "e8", "f26", "f29", "e24", "d8", "d25", "h31", "b1", "h24", "ab29", "ab2", "e27", "h9", "c19", "f24", "f12", "c7", "c22", "ab10", "ac12", "c16", "aa32", "ad32", "c21", "h25", "ad12", "e25", "d17", "e22", "aa26", "ac35", "ab33", "c14", "ab27", "b8", "h28", "f27", "c27", "e18", "c23", "d29", "f19", "b11", "ac29", "ac24", "ac32", "a9", "ad36", "aa29", "ad33", "aa21", "a17", "b21", "ac22", "ab30", "h20", "b19", "ad25", "aa36"]
                
                # <3s
                self.instance_list = ["aa13", "c26", "ad40", "ac30", "h34", "h12", "ac38", "mult_op_DEMO1_9_9_TOP14", "mult_op_DEMO1_13_13_TOP9", "mult_op_DEMO1_11_11_TOP21", "mult_op_DEMO1_11_11_TOP9", "ad39", "f32", "f25", "ab36", "aa35", "c9", "ab28", "mult_op_DEMO1_10_10_TOP18", "ad15", "d22", "a24", "b4", "b26", "e23", "e10", "c20", "h3", "d10", "ac15", "aa12", "e11", "ab12", "mult_op_DEMO1_12_12_TOP24", "aa27", "a8", "d28", "mult_op_DEMO1_9_9_TOP15", "e3", "ac19", "f10", "ab4", "mult_op_DEMO1_8_8_TOP10", "ad31", "a29", "ad20", "h26", "mult_op_DEMO1_10_10_TOP9", "h11", "mult_op_DEMO1_8_8_TOP11", "ad6", "a25", "b10", "c2", "aa4", "ac14", "f3", "e14", "a3", "mult_op_DEMO1_8_8_TOP12", "ac6", "mult_op_DEMO1_10_10_TOP19", "h16", "a22", "mult_op_DEMO1_9_9_TOP16", "d14", "b24", "c13", "mult_op_DEMO1_8_8_TOP13", "e7", "ab17", "a27", "mult_op_DEMO1_13_13_TOP8", "f7", "d3", "mult_op_DEMO1_8_8_TOP9", "f15", "ac11", "mult_op_DEMO1_9_9_TOP9", "a16", "mult_op_DEMO1_12_12_TOP9", "mult_op_DEMO1_11_11_TOP8", "b18", "mult_op_DEMO1_11_11_TOP22", "h8", "a7", "c6", "mult_op_DEMO1_9_9_TOP17", "b9", "ad11", "aa17", "mult_op_DEMO1_8_8_TOP14", "c24", "d7", "e28", "mult_op_DEMO1_10_10_TOP20", "mult_op_DEMO1_8_8_TOP15", "mult_op_DEMO1_10_10_TOP8", "h32", "ac36", "mult_op_DEMO1_7_7_TOP10", "mult_op_DEMO1_6_6_TOP6", "mult_op_DEMO1_7_7_TOP11", "mult_op_DEMO1_12_12_TOP8", "ab9", "mult_op_DEMO1_7_7_TOP12", "aa9", "ab34", "ad37", "mult_op_DEMO1_7_7_TOP8", "mult_op_DEMO1_8_8_TOP8", "d26", "c5", "mult_op_DEMO1_9_9_TOP18", "mult_op_DEMO1_7_7_TOP7", "b2", "a1", "mult_op_DEMO1_13_13_TOP7", "mult_op_DEMO1_9_9_TOP8", "mult_op_DEMO1_9_9_TOP7", "ac10", "mult_op_DEMO1_7_7_TOP9", "h7", "mult_op_DEMO1_8_8_TOP16", "a11", "ad10", "mult_op_DEMO1_11_11_TOP7", "aa33", "d6", "mult_op_DEMO1_6_6_TOP9", "b13", "aa8", "f30", "mult_op_DEMO1_12_12_TOP7", "b6", "a5", "b22", "mult_op_DEMO1_6_6_TOP7", "mult_op_DEMO1_6_6_TOP10", "ab8", "h5", "mult_op_DEMO1_7_7_TOP14", "b32", "a30", "a20", "b5", "f6", "a26", "mult_op_DEMO1_6_6_TOP8", "mult_op_DEMO1_8_8_TOP7", "mult_op_DEMO1_10_10_TOP7", "mult_op_DEMO1_13_13_TOP6", "aa6", "mult_op_DEMO1_7_7_TOP13", "a4", "mult_op_DEMO1_11_11_TOP6", "e6", "mult_op_DEMO1_8_8_TOP6", "c3", "a2", "mult_op_DEMO1_12_12_TOP6", "a28", "f4", "d4", "mult_op_DEMO1_5_5_TOP8", "mult_op_DEMO1_6_6_TOP11", "ab6", "e17", "mult_op_DEMO1_5_5_TOP3", "mult_op_DEMO1_11_11_TOP5", "mult_op_DEMO1_13_13_TOP5", "ad24", "e4", "mult_op_DEMO1_5_5_TOP9", "ac8", "aa20"]
        else:
            self.instance_list = instance_list
            
        self.args = args
        self.problem_list = []
        self.mode = mode
        self.no_instance = 0
        self.step_ntk_filepath = self.args.step_ntk_filepath
        for aig_path in glob.glob(os.path.join(args.Problem_AIG_Dir, '*.aiger')):
            aig_name = os.path.basename(aig_path).split('.')[0]
            if aig_name in self.instance_list:
                self.problem_list.append(aig_path)
        self.parser = dg.AigParser()
                
    def reset(self):
        while True:
            if self.mode == 'train':
                curr_problem = random.choice(self.problem_list)
            else:
                curr_problem = self.problem_list[self.no_instance]
                self.no_instance += 1
            problem_name = curr_problem.split('/')[-1].split('.')[0]
            shutil.copyfile(curr_problem, self.step_ntk_filepath)
            # Initialize 
            init_cmd = 'abc -c \"read_aiger {}; rewrite -lz; balance; rewrite -lz; balance; rewrite -lz; balance; write_aiger {}; \"'.format(
                self.step_ntk_filepath, self.step_ntk_filepath
            )
            _, _ = run_command(init_cmd)
            self.origin_problem = curr_problem
            status, self.graph = self.parser.read_aiger(self.step_ntk_filepath)
            
            # Baseline
            self.sat, map_time, solve_time, no_dec, nvars, nclas = env_map_solve(self.args, self.args.customized_mapper, self.step_ntk_filepath, self.args.tmp_dir)
            
            if status and self.sat != -1 and len(self.graph.POs) == 1:
                break
            elif self.mode != 'train':
                raise "There is illegal instance in test mode!"

        self.bl_dec = no_dec
        self.bl_nvars = nvars
        self.bl_nclas = nclas
        self.step_cnt = 0
        self.action_time = 0
        self.problem_name = problem_name
        self.bl_st = solve_time
        self.bl_mp = map_time
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
            _, self.graph = self.parser.read_aiger(self.step_ntk_filepath)
            reward = 0
            done = False
        
        else:
            sat_res, map_time, solve_time, no_dec, nvars, nclas = env_map_solve(self.args, self.args.customized_mapper, self.step_ntk_filepath, self.args.tmp_dir)
            self.synmap_nvars = nvars
            self.synmap_nclas = nclas
            reward = (self.bl_dec - no_dec) / self.bl_dec
            done = True
            self.md_st = solve_time
            self.md_mp = self.action_time + map_time
            self.md_dec = no_dec
            self.md_nvars = nvars
            self.md_nclas = nclas
            
        self.step_cnt += 1
        info = {}
        return self.graph, reward, done, info
        
    def get_solve_info(self):
        res_dict = {
            'bl_st': self.bl_st,
            'bl_mp': self.bl_mp, 
            'bl_dec': self.bl_dec,
            'bl_nvars': self.bl_nvars,
            'bl_nclas': self.bl_nclas,
            'md_st': self.md_st,
            'md_mp': self.md_mp,
            'md_dec': self.md_dec,
            'md_nvars': self.md_nvars,
            'md_nclas': self.md_nclas,
        }
        return res_dict
        
