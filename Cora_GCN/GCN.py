import torch
import torch.nn as nn
import torch.nn.init as init

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()
        return
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """
        adjacency_matrix is a sparse matrix 
        therefore using torch.sparse.mm when calculating
        args:
        adjacency: torch.sparse.FloatTensor
        input_feature: torch.Tensor
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output