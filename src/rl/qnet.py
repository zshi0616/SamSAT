import torch 
import deepgate as dg
import torch.nn as nn 
from progress.bar import Bar
from torch.nn import LSTM, GRU
from models.mlp import MLP

class Q_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ckt_model = dg.Model(dim_hidden=args.ckt_dim)
        self.mlp = MLP(args.ckt_dim*2, args.mlp_dim, args.n_action, \
            num_layer=args.mlp_layers, p_drop=0.2, act_layer='relu')
        self.graph_emb = None 
        
    def forward_one(self, obs):
        hs, hf = self.ckt_model(obs)
        self.graph_emb = torch.cat([hs[obs.POs], hf[obs.POs]], dim=1)
        y_pred = self.mlp(self.graph_emb)
        
        return y_pred
    
    def forward_batch(self, obs):
        batch_size = len(obs)

        # Merge 
        x = torch.cat([obs[i].x for i in range(batch_size)], dim=0)
        edge_index = torch.cat([obs[i].edge_index for i in range(batch_size)], dim=1)
        forward_level = torch.cat([obs[i].forward_level for i in range(batch_size)], dim=0)
        backward_level = torch.cat([obs[i].backward_level for i in range(batch_size)], dim=0)
        forward_index = torch.cat([obs[i].forward_index for i in range(batch_size)], dim=0)
        backward_index = torch.cat([obs[i].backward_index for i in range(batch_size)], dim=0)
        gate = torch.cat([obs[i].gate for i in range(batch_size)], dim=0)
        PIs = torch.cat([obs[i].PIs for i in range(batch_size)], dim=0)
        POs = torch.cat([obs[i].POs for i in range(batch_size)], dim=0)
        graph = dg.OrderedData(edge_index=edge_index, x=x, forward_level=forward_level, backward_level=backward_level, \
            forward_index=forward_index, backward_index=backward_index)
        graph.gate = gate
        graph.PIs = PIs
        graph.POs = POs
        
        # forward
        hs, hf = self.ckt_model(graph)
        self.graph_emb = torch.cat([hs[graph.POs], hf[graph.POs]], dim=1)
        y_pred = self.mlp(self.graph_emb)
        
        return y_pred
        
    def forward(self, obs):
        if len(obs) == 1:
            return self.forward_one(obs[0])
        else:
            return self.forward_batch(obs)
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
        
        