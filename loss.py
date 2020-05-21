import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropyLoss(nn.Module):
    """
    Because of the attentional idea of matching network which includes
    a probability attention kernel to compute probablity attention(attention score)
    and finally to obtain probablity scores of the test(query) samples
    (weighted sum of the groundtruth of the train(support) samples),
    the 'preds' is actually kind of probability distribution.
    """
    def __init__(self, args):
        super(CrossEntropyLoss, self).__init__()
        self.args = args

    def forward(self, preds, one_hot_label, one_hot_label_global=None):
        global_loss = 0.0
        loss = 0.0
        global_preds = None
        alpha = 1.0
        if self.args.use_global:
            global_preds = preds[1]
            preds = preds[0]
            alpha = 0.5
        preds = -F.log_softmax(preds, dim=2).mean(3)
        loss = torch.sum(preds*one_hot_label.float(), dim=[0, 1, 2])
        if self.args.use_global:
            # global_preds: [B, N8num_query, train_categoties, H*W]
            global_preds = -F.log_softmax(global_preds, dim=2).mean(3) # [4, 30, train_categoties]
            global_loss = torch.sum(global_preds*one_hot_label_global.float(), dim=[0, 1, 2])

        return (alpha*loss+global_loss)/(preds.size(0)*preds.size(1))

