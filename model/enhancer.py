import torch
import math
from torch import nn
from torch.nn import functional as F

class Concentrator(nn.Module):
    def __init__(self, in_c, out_c):
        super(Concentrator, self).__init__()
        self.conv0 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(out_c)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # x: [B, N*K, C, H, W]
        x = x.reshape(-1, *x.size()[2:])
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out + x

class Enhancer(nn.Module):
    def __init__(self, in_c, out_c, N, K, args=None):
        super(Enhancer, self).__init__()
        self.args = args
        self.N = N
        self.K = K
        self.fc = nn.Sequential(
                        nn.Linear(in_c, in_c//4),
                        nn.BatchNorm1d(in_c//4),
                        nn.LeakyReLU(0.2, True),
                        nn.Linear(in_c//4, in_c//16),
                        nn.BatchNorm1d(in_c//16),
                        nn.LeakyReLU(0.2, True),
                        nn.Linear(in_c//16, 1)
        )

        self.concentrator = Concentrator(in_c, out_c)

    def weight_sum(self, prototype):
        # prototype: [B, N, C, H, W]
        b, n, c, h, w = prototype.size()
        centroid = prototype.mean(dim=[3, 4]) # [B, N, C]
        centroid = centroid.view(-1, c) # [B*N, C]
        att_score = self.fc(centroid)
        att_score = F.softmax(att_score.view(b, n, 1, 1, 1), dim=1) # [B, N]
        return torch.sum(prototype*att_score, dim=1) # [B, C, H, W]

    def forward(self, embeddings):
        # embeddings: [B*N*(K+num_query), C, H, W]
        batch_size=0
        if self.training:
            batch_size = self.args.train_batch
        else:
            batch_size = self.args.test_batch
        embeddings = embeddings.view(batch_size, -1, *embeddings.size()[1:])
        b, nkq, c, h, w = embeddings.size()
        nk = self.N*self.K
        fspt = embeddings[:, :nk] # [B, N*K, C, H, W]
        fspt_conc = self.concentrator(fspt)
        prototype = fspt_conc.reshape([b, self.N, self.K, c, h, w]).mean(dim=2) # [B, N, C, H, W]
        # prototype = prototype.mean(dim=1) # [B, C, H, W]
        prototype = self.weight_sum(prototype) # [B, C, H, W]

        mask = F.softmax(prototype, dim=1)+1
        enhanced_embeddings = mask.unsqueeze(1)*embeddings
        enhanced_embeddings = enhanced_embeddings.view(-1, *enhanced_embeddings.size()[2:])

        return enhanced_embeddings