import torch
import torch.nn as nn
from src.shape_utils import load_shape_with_lbo



class CSE(nn.Module):
    def __init__(self, class_name, num_basis=64, skip_first=True, dim=16, num_vert=None, barebones=False, device=torch.device('cuda'), rand_init=False):
        super(CSE, self).__init__()
        
        self.shape = load_shape_with_lbo(class_name, num_basis, skip_first)

        if barebones:
            return
        self.functional_basis=None
        
        # Create a parameter tensor for the D x Q matrix, initialized randomly
        if not rand_init:
            self.weight_matrix = nn.Parameter(torch.zeros(num_basis, dim, requires_grad=True))
        else:
            self.weight_matrix = nn.Parameter(torch.randn(num_basis, dim, requires_grad=True))

        self.to(device)
        self.nns = None
        self.num_vert = num_vert
    
    def forward(self):
        output = torch.matmul(self.functional_basis, self.weight_matrix)

        if self.num_vert is not None:
            output_tmp = torch.zeros((self.num_vert, output.shape[1])).to(output.device)
            output_tmp[:output.shape[0], :] = output
            output = output_tmp
        return output
    
