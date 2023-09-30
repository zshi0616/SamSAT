import os 
import numpy as np 
import torch

class Agent:
    def __init__(self, net, args, config) -> None:
        self.net = net
        self.args = args
        self.config = config
        self.device = args.device 
        
    def forward(self, hist_buffer):
        self.net.eval()
        with torch.no_grad():
            y_pred = self.net(hist_buffer)
            return y_pred
        
    def act(self, hist_buffer, eps):
        if eps < self.config.RANDOM_ACTION:
            return np.random.randint(0, self.args.n_action)
        else:
            y_pred = self.forward(hist_buffer)
            return torch.argmax(y_pred).item()