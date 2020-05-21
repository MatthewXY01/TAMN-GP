import torch
import math
from torch import nn
from torch.nn import functional as F

class MAtt(nn.Module):
    def __init__(self, N=5, K=1):
        super(MAtt, self).__init__()
        self.N = N
        self.K = K

    def forward(self, fspt, fqry):
        # fspt: [B, N*K, C, H, W]
        # fqry: [B, N*num_query, C, H, W]
        
        b, nk, c, h, w = fspt.size()
        nq = fqry.size(1)
        n = self.N

        prototype = fspt.view(b, self.N, self.K, c, h, w).mean(dim=2) # [B, N, C, H, W]
        prototype = prototype.view(b, n, c, -1).unsqueeze(1) # [B, 1, N, C, H*W]
        fqry = fqry.view(b, nq, c, -1).unsqueeze(2) # [B, N*num_query, 1, C, H*W]

        prototype_norm = F.normalize(prototype, p=2, dim=3).transpose(3, 4) # [B, 1, N, H*W, C]
        fqry_norm = F.normalize(fqry, p=2, dim=3) # [B, N*num_query, 1, C, H*W]

        correlation_mat = torch.matmul(prototype_norm, fqry_norm) # [B, N*num_query, N, H*W, H*W]

        ap = correlation_mat.mean(dim=4, keepdim=True).transpose(3, 4) # [B, N*num_query, N, 1, H*W]
        ap = F.softmax(ap/0.25, dim=4) + 1
        aq = correlation_mat.mean(dim=3, keepdim=True) # [B, N*num_query, N, 1, H*W]
        aq = F.softmax(aq/0.25, dim=4) + 1

        prototype_adp = (prototype*ap).view(b, nq, n, c, h, w)
        fqry_adp = (fqry*aq).view(b, nq, n, c, h, w)

        return prototype_adp, fqry_adp

