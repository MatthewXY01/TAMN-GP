import os
import os.path as osp
import sys
import time
import random
import datetime
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.tamn import TaskAwareMetricNetwork
from datamanager import DataManager
from loss import CrossEntropyLoss
from args.args_tiredImageNet import argument_parser
from utils.logger import AverageMeter, calculate_acc, Logger, save_checkpoint
# import multiprocessing
# multiprocessing.set_start_method('spawn',True)

parser = argument_parser()
args = parser.parse_args()
# import wandb
# api = wandb.Api()
# run = api.run(args.run_path)
def main():
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print('You deserve a gpu!')
        return
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print('Initializing data manager')
    dm = DataManager(args)

    test_loader = dm.create_dataloader(phase='test', mode='few-shot')
    model = TaskAwareMetricNetwork(args=args)

    init_params = torch.load(osp.join(args.save_dir, 'best_model.pth.tar'))['state_dict']
    dict2load = {k:v for k, v in init_params.items() if k in model.state_dict()}
    model_dict = model.state_dict()
    model_dict.update(dict2load)

    model = model.cuda()
    model.load_state_dict(model_dict, strict = False)

    start_time = time.time()
    test_time = 0
    print("==> Start testing")
    test(test_loader, model, args)


def test(dataloader, model, args):

    accs = AverageMeter()
    losses = AverageMeter()
    test_accuracies = []
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images_spt, labels_spt, images_qry, labels_qry, cids_qry) in enumerate(dataloader):
            batch_size = images_spt.size(0)
            images_spt, labels_spt = images_spt.cuda(), labels_spt.cuda()
            images_qry, labels_qry = images_qry.cuda(), labels_qry.cuda()
            labels_spt_1hot = F.one_hot(labels_spt).cuda()
            labels_qry_1hot = F.one_hot(labels_qry).cuda()

            test_logits, preds = model(images_spt, labels_spt_1hot, images_qry, labels_qry_1hot)
            test_loss = criterion(test_logits.view(-1, test_logits.size(-1)), labels_qry.view(-1))
            losses.update(test_loss.item(), batch_size*labels_qry.size()[1])
            acc, acc_per_task = calculate_acc(preds, labels_qry)
            accs.update(acc.item(), batch_size*labels_qry.size(1))
            acc_per_task = np.reshape(acc_per_task.numpy(), (batch_size))
            test_accuracies.append(acc_per_task)
    
    accuracy = accs.avg
    loss = losses.avg
    test_accuracies = np.reshape(np.array(test_accuracies), -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.num_episode_test)
    print('Accuracy: {:.2%}, std: {:.2%}, Loss: {:.4f}'.format(accuracy, ci95, loss))
    run.summary['test acc'] = accuracy
    run.summary.update()
    return accuracy
    

if __name__=='__main__':
    main()
