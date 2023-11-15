import numpy as np
import gym
import random
import os
import argparse
import sys
import glob
import shutil
import deepgate as dg
import torch
import time

from utils.utils import run_command
from utils.cnf_utils import kissat_solve, kissat_solve_dec
from utils.lut_utils import parse_bench_cnf
import utils.circuit_utils as circuit_utils
from utils.aiger_utils import aig_to_xdata, xdata_to_cnf


def get_parse_args():
    parser = argparse.ArgumentParser(description='Solve')
    parser.add_argument('--path', default='./debug/138.bench')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parse_args()
    kissat_path = './src/kissat/build/kissat'
    cnf, no_var = parse_bench_cnf(args.path)
    print('==> Start Solving')
    print('# Var: {:}, # Clause: {:}'.format(no_var, len(cnf)))
    sat_res, asg, solvetime, no_dec = kissat_solve_dec(kissat_path, cnf, no_var, './tmp', args='--time={:}'.format(1000))
    
    print('Result: {}, Time: {:.2f}'.format(sat_res, solvetime))
    