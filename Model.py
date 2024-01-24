from GCN import GraphConvolution
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class GCNnet(nn.Module):
    """
    Define a two-layers GCN model
    """
    def __init__(self, input_dim=1433):
        super(GCNnet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim=input_dim, output_dim=16)
        self.gcn2 = GraphConvolution(16, 7)
    
    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits
    
    def normalization(adjacency):
        """ L = D^(-0.5) * (A + I) * D^(-0.5)"""
        adjacency += sp.eye(adjacency.shape[0])
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()