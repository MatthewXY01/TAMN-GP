import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet12 import ResNet, BasicBlock
from .matt import MAtt

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.layer_sizes = [64, 64, 64, 64]
        self.conv0 = nn.Conv2d(3, self.layer_sizes[0], 3, 1, 1)
        self.norm0 = nn.BatchNorm2d(self.layer_sizes[0], momentum=0.01)
        self.maxpool0 = nn.MaxPool2d(2, 2, padding=0)
        self.drop0 = nn.Dropout2d(0.2) # leave it
        self.conv1 = nn.Conv2d(self.layer_sizes[0], self.layer_sizes[1], 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(self.layer_sizes[1], momentum=0.01)
        self.maxpool1 = nn.MaxPool2d(2, 2, padding=0)
        self.conv2 = nn.Conv2d(self.layer_sizes[1], self.layer_sizes[2], 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(self.layer_sizes[2], momentum=0.01)
        self.maxpool2 = nn.MaxPool2d(2, 2, padding=1)
        self.conv3 = nn.Conv2d(self.layer_sizes[2], self.layer_sizes[3], 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(self.layer_sizes[3], momentum=0.01)   

    def forward(self, X):
        b, num_img, c, h, w = X.size()
        X = X.view(b*num_img, c, h, w)
        #84x84
        X = self.maxpool0(F.relu(self.norm0(self.conv0(X))))
        #42x42
        X = self.maxpool1(F.relu(self.norm1(self.conv1(X))))
        #21x21
        X = self.maxpool2(F.relu(self.norm2(self.conv2(X))))
        #11x11
        X = F.relu(self.norm3(self.conv3(X)))

        bnum_img, ce, he, we = X.size()
        embeddings = X.view(b, num_img, ce, he, we)
        return embeddings

class ResExtractor(nn.Module):
    def __init__(self, N=5, K=1, args=None):
        super(ResExtractor, self).__init__()
        self.resnet12 = ResNet(BasicBlock, [1, 1, 1, 1], N, K, args)
    def forward(self, X):
        b, num_img, c, h, w = X.size()
        X = X.view(b*num_img, c, h, w)
        X = self.resnet12(X)
        bnum_img, ce, he, we = X.size()
        embeddings = X.view(b, num_img, ce, he, we)
        return embeddings

class TaskAwareMetricNetwork(nn.Module):
    def __init__(self, args):
        super(TaskAwareMetricNetwork, self).__init__()
        self.args = args
        self.N = args.way
        self.K = args.shot
        self.extractor = ResExtractor(self.N, self.K, args)
        self.scale_cls = args.scale_cls
        self.use_matt = args.use_matt
        if self.use_matt:
            self.matt = MAtt(self.N, self.K)
        self.use_global = args.use_global
        if self.use_global:
            self.global_classifier = nn.Conv2d(512, args.train_categories, kernel_size=1)

    def tamn_eval(self, prototype, fqry):
        # prototype: [B, N*num_query or 1, N, C, H, W]
        # fqry: [B, N*num_query, 1, C, H, W]
        prototype = prototype.mean(dim=[4, 5])
        fqry = fqry.mean(dim=[4, 5])
        prototype_norm = F.normalize(prototype, p=2, dim=prototype.dim()-1, eps=1e-12)
        fqry_norm = F.normalize(fqry, p=2, dim=fqry.dim()-1, eps=1e-12)
        scores = torch.sum(fqry_norm*prototype_norm, dim=-1) # [B, N*num_query, N]
        val_logits = F.softmax(scores, 2)
        scores = self.scale_cls*scores
        return val_logits, scores

    def forward(self, Xspt, Yspt, Xqry, Yqry, Zqry=None):
        num_spt = Xspt.size()[1]
        num_qry = Xqry.size()[1]
        # merge support set and query set in order to share the feature extractor
        X = torch.cat([Xspt, Xqry], dim=1)
        embeddings = self.extractor(X) # [B, N*(K+num_query), 64, 11, 11] [4, 35, 64, 11, 11]
        batch_size, nkq, c, h, w = embeddings.size()

        fspt = embeddings[:, :num_spt] # [B, N*K, C, H, W]
        fqry = embeddings[:, num_spt:]
        if self.use_matt:
            prototype, fqry_adp = self.matt(fspt, fqry)  #[B, N*num_query, N, C, H, W]
        else:
            fqry_adp = fqry.unsqueeze(2) # [B, N*num_query, 1, C, H, W]
            prototype = fspt.view(batch_size, self.N, self.K, *fspt.size()[2:]).mean(dim=2) # [B, N, C, H, W]
            prototype = prototype.unsqueeze(1) # [B, 1, N, C, H, W]
        
        if not self.training:
            return self.tamn_eval(prototype, fqry_adp) # [B, N*num_query, N]

        prototype = prototype.mean(dim=[4, 5], keepdim=True) # [B, N*num_query or 1, N, C, 1, 1]
        fqry_norm = F.normalize(fqry_adp, p=2, dim=3, eps=1e-12)
        prototype_norm = F.normalize(prototype, p=2, dim=3, eps=1e-12)
        cls_scores = self.scale_cls* torch.sum(fqry_norm*prototype_norm, dim=3)
        cls_scores = cls_scores.view(*cls_scores.size()[:3], -1) # [B, N*num_query, N, H*W]

        if not self.use_global:
            return cls_scores
        else:
            fqry = fqry.reshape(batch_size*num_qry, c, h, w)
            global_preds = self.global_classifier(fqry)
            global_preds = global_preds.view(batch_size, num_qry, self.args.train_categories, -1)
            return (cls_scores, global_preds)

if __name__ == '__main__':
    test_model = TaskAwareMetricNetwork()
    # test_model.eval()
    for i, j in test_model.named_parameters():
        if j.requires_grad:
            print(i)
    # print(test_model)