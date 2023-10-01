import os 
import torch
import numpy as np 
import deepgate as dg 
import copy
import gym
from collections import deque

from config import get_parse_args
from rl.config import RL_Config
from rl.qnet import Q_Net
from rl.agent import Agent
from rl.trainer import Trainer
from rl.buffer import ReplayBuffer
from rl.env import solve_Env

if __name__ == '__main__':
    args = get_parse_args()
    config = RL_Config()
    print('==> Using settings {}'.format(args))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training in single device: ', args.device)
    
    # Create RL environments 
    net = Q_Net(args)
    target_net = copy.deepcopy(net)
    rl_env = solve_Env(args=args)
    agent = Agent(net, args, config)
    buffer = ReplayBuffer(args, config)
    trainer = Trainer(args, config, net, target_net, buffer)
    
    # Train 
    eps = 0
    for train_times in range(args.train_times):
        obs = rl_env.reset()
        done = False
        tot_reward = 0
        print('==> Training: {:} / {:}, Problem: {}'.format(train_times, args.train_times, rl_env.problem_name))
        
        while not done:
            action = agent.act(obs, eps)
            next_obs, reward, done, info = rl_env.step(action)
            eps += 1
            buffer.add_transition(obs, action, reward, done)
            obs = next_obs
            tot_reward += reward
            
            if buffer.ctr >= config.OBSERVE:
                train_info = trainer.step()
                print('==> Step: {:}, Loss: {:.4f}, Average Q: {:.4f}'.format(trainer.step_ctr, train_info['loss'], train_info['average_q']))
                if trainer.step_ctr % args.save_epoch == 0:
                    net.save(os.path.join(args.save_dir, 'qnet.pth'))
                    print('==> Save model to {}'.format(os.path.join(args.save_dir, 'qnet.pth')))
    