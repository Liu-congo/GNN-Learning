from GCN import GraphConvolution

import torch.nn as nn
import torch.nn.functional as F 


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
    
    