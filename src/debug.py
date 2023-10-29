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

MAX_SOLVE_TIME = 100
customized_mapper = './mockturtle/build/examples/my_mapper'

def env_map_solve(mapper, aig_filename, tmp_dir):
    # Map to LUTs
    ckt_name = aig_filename.split('/')[-1].split('.')[0]
    bench_filename = os.path.join(tmp_dir, '{}.bench'.format(ckt_name))
    cmd_maplut = '{} {} {}'.format(
        mapper, aig_filename, bench_filename
    )
    _, maplut_runtime = run_command(cmd_maplut, MAX_SOLVE_TIME)
    if maplut_runtime < 0:
        return -1, 0, 0, 0, 0
    
    # Parse LUTs
    cnf, no_var = parse_bench_cnf(bench_filename)
    os.remove(bench_filename)
    
    # Solve 
    if maplut_runtime > MAX_SOLVE_TIME:
        sat_res = -1 
        no_dec = 0
    else:
        sat_res, asg, solvetime, no_dec = kissat_solve_dec('./kissat/build/kissat', cnf, no_var, args='--time={:}'.format(int(MAX_SOLVE_TIME - maplut_runtime)))
    
    return sat_res, maplut_runtime, solvetime, no_dec, no_var, len(cnf)


if __name__ == '__main__':
    dst_dir = '/uac/gds/zyshi21/studio/dataset/SamSAT'
    instance_list = ["aa13", "c26", "ad40", "ac30", "h34", "h12", "ac38", "mult_op_DEMO1_9_9_TOP14", "mult_op_DEMO1_13_13_TOP9", "mult_op_DEMO1_11_11_TOP21", "mult_op_DEMO1_11_11_TOP9", "ad39", "f32", "f25", "ab36", "aa35", "c9", "ab28", "mult_op_DEMO1_10_10_TOP18", "ad15", "d22", "a24", "b4", "b26", "e23", "e10", "c20", "h3", "d10", "ac15", "aa12", "e11", "ab12", "mult_op_DEMO1_12_12_TOP24", "aa27", "a8", "d28", "mult_op_DEMO1_9_9_TOP15", "e3", "ac19", "f10", "ab4", "mult_op_DEMO1_8_8_TOP10", "ad31", "a29", "ad20", "h26", "mult_op_DEMO1_10_10_TOP9", "h11", "mult_op_DEMO1_8_8_TOP11", "ad6", "a25", "b10", "c2", "aa4", "ac14", "f3", "e14", "a3", "mult_op_DEMO1_8_8_TOP12", "ac6", "mult_op_DEMO1_10_10_TOP19", "h16", "a22", "mult_op_DEMO1_9_9_TOP16", "d14", "b24", "c13", "mult_op_DEMO1_8_8_TOP13", "e7", "ab17", "a27", "mult_op_DEMO1_13_13_TOP8", "f7", "d3", "mult_op_DEMO1_8_8_TOP9", "f15", "ac11", "mult_op_DEMO1_9_9_TOP9", "a16", "mult_op_DEMO1_12_12_TOP9", "mult_op_DEMO1_11_11_TOP8", "b18", "mult_op_DEMO1_11_11_TOP22", "h8", "a7", "c6", "mult_op_DEMO1_9_9_TOP17", "b9", "ad11", "aa17", "mult_op_DEMO1_8_8_TOP14", "c24", "d7", "e28", "mult_op_DEMO1_10_10_TOP20", "mult_op_DEMO1_8_8_TOP15", "mult_op_DEMO1_10_10_TOP8", "h32", "ac36", "mult_op_DEMO1_7_7_TOP10", "mult_op_DEMO1_6_6_TOP6", "mult_op_DEMO1_7_7_TOP11", "mult_op_DEMO1_12_12_TOP8", "ab9", "mult_op_DEMO1_7_7_TOP12", "aa9", "ab34", "ad37", "mult_op_DEMO1_7_7_TOP8", "mult_op_DEMO1_8_8_TOP8", "d26", "c5", "mult_op_DEMO1_9_9_TOP18", "mult_op_DEMO1_7_7_TOP7", "b2", "a1", "mult_op_DEMO1_13_13_TOP7", "mult_op_DEMO1_9_9_TOP8", "mult_op_DEMO1_9_9_TOP7", "ac10", "mult_op_DEMO1_7_7_TOP9", "h7", "mult_op_DEMO1_8_8_TOP16", "a11", "ad10", "mult_op_DEMO1_11_11_TOP7", "aa33", "d6", "mult_op_DEMO1_6_6_TOP9", "b13", "aa8", "f30", "mult_op_DEMO1_12_12_TOP7", "b6", "a5", "b22", "mult_op_DEMO1_6_6_TOP7", "mult_op_DEMO1_6_6_TOP10", "ab8", "h5", "mult_op_DEMO1_7_7_TOP14", "b32", "a30", "a20", "b5", "f6", "a26", "mult_op_DEMO1_6_6_TOP8", "mult_op_DEMO1_8_8_TOP7", "mult_op_DEMO1_10_10_TOP7", "mult_op_DEMO1_13_13_TOP6", "aa6", "mult_op_DEMO1_7_7_TOP13", "a4", "mult_op_DEMO1_11_11_TOP6", "e6", "mult_op_DEMO1_8_8_TOP6", "c3", "a2", "mult_op_DEMO1_12_12_TOP6", "a28", "f4", "d4", "mult_op_DEMO1_5_5_TOP8", "mult_op_DEMO1_6_6_TOP11", "ab6", "e17", "mult_op_DEMO1_5_5_TOP3", "mult_op_DEMO1_11_11_TOP5", "mult_op_DEMO1_13_13_TOP5", "ad24", "e4", "mult_op_DEMO1_5_5_TOP9", "ac8", "aa20"]
    aig_dir = '/uac/gds/zyshi21/studio/dataset/LEC/all_case'
    for aig_name in instance_list:
        aig_filepath = os.path.join(aig_dir, aig_name + '.aiger')
        if os.path.exists(aig_filepath):
            dst_filepath = os.path.join(dst_dir, aig_name + '.aiger')
            shutil.copyfile(aig_filepath, dst_filepath)
            print(dst_filepath)

    instance_list = ["mult_op_DEMO1_9_9_TOP14", "mult_op_DEMO1_13_13_TOP9", "mult_op_DEMO1_11_11_TOP21", "mult_op_DEMO1_11_11_TOP9", "ad39", "f32", "f25", "ab36", "aa35", "c9", "ab28", "mult_op_DEMO1_10_10_TOP18", "ad15", "d22", "a24", "b4", "b26", "e23", "e10", "c20", "h3", "d10", "ac15", "aa12", "e11", "ab12", "mult_op_DEMO1_12_12_TOP24", "aa27", "a8", "d28", "mult_op_DEMO1_9_9_TOP15", "e3", "ac19", "f10", "ab4", "mult_op_DEMO1_8_8_TOP10", "ad31", "a29", "ad20", "h26", "mult_op_DEMO1_10_10_TOP9", "h11", "mult_op_DEMO1_8_8_TOP11", "ad6", "a25", "b10", "c2", "aa4", "ac14", "f3", "e14", "a3", "mult_op_DEMO1_8_8_TOP12", "ac6", "mult_op_DEMO1_10_10_TOP19", "h16", "a22", "mult_op_DEMO1_9_9_TOP16", "d14", "b24", "c13", "mult_op_DEMO1_8_8_TOP13", "e7", "ab17", "a27", "mult_op_DEMO1_13_13_TOP8", "f7", "d3", "mult_op_DEMO1_8_8_TOP9", "f15", "ac11", "mult_op_DEMO1_9_9_TOP9", "a16", "mult_op_DEMO1_12_12_TOP9", "mult_op_DEMO1_11_11_TOP8", "b18", "mult_op_DEMO1_11_11_TOP22", "h8", "a7", "c6", "mult_op_DEMO1_9_9_TOP17", "b9", "ad11", "aa17", "mult_op_DEMO1_8_8_TOP14", "c24", "d7", "e28", "mult_op_DEMO1_10_10_TOP20", "mult_op_DEMO1_8_8_TOP15", "mult_op_DEMO1_10_10_TOP8", "h32", "ac36", "mult_op_DEMO1_7_7_TOP10", "mult_op_DEMO1_6_6_TOP6", "mult_op_DEMO1_7_7_TOP11", "mult_op_DEMO1_12_12_TOP8", "ab9", "mult_op_DEMO1_7_7_TOP12", "aa9", "ab34", "ad37", "mult_op_DEMO1_7_7_TOP8", "mult_op_DEMO1_8_8_TOP8", "d26", "c5", "mult_op_DEMO1_9_9_TOP18", "mult_op_DEMO1_7_7_TOP7", "b2", "a1", "mult_op_DEMO1_13_13_TOP7", "mult_op_DEMO1_9_9_TOP8", "mult_op_DEMO1_9_9_TOP7", "ac10", "mult_op_DEMO1_7_7_TOP9", "h7", "mult_op_DEMO1_8_8_TOP16", "a11", "ad10", "mult_op_DEMO1_11_11_TOP7", "aa33", "d6", "mult_op_DEMO1_6_6_TOP9", "b13", "aa8", "f30", "mult_op_DEMO1_12_12_TOP7", "b6", "a5", "b22", "mult_op_DEMO1_6_6_TOP7", "mult_op_DEMO1_6_6_TOP10", "ab8", "h5", "mult_op_DEMO1_7_7_TOP14", "b32", "a30", "a20", "b5", "f6", "a26", "mult_op_DEMO1_6_6_TOP8", "mult_op_DEMO1_8_8_TOP7", "mult_op_DEMO1_10_10_TOP7", "mult_op_DEMO1_13_13_TOP6", "aa6", "mult_op_DEMO1_7_7_TOP13"]
    aig_dir = '/uac/gds/zyshi21/studio/dataset/LEC/all_case_recoveraig'
    for aig_name in instance_list:
        aig_filepath = os.path.join(aig_dir, aig_name + '.aiger')
        if os.path.exists(aig_filepath):
            dst_filepath = os.path.join(dst_dir, 'r_{}.aiger'.format(aig_name))
            shutil.copyfile(aig_filepath, dst_filepath)
            print(dst_filepath)
            