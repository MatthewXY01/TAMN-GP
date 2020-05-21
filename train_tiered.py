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
import torch.optim as optim
from model.tamn import TaskAwareMetricNetwork
from datamanager import DataManager
from loss import CrossEntropyLoss
from optimizer import init_optimizer
from args.args_tiredImageNet import argument_parser
from utils.logger import AverageMeter, calculate_acc, Logger, save_checkpoint
# import multiprocessing
# multiprocessing.set_start_method('spawn',True)
# import wandb
# os.environ["WANDB_MODE"] = "dryrun"
# os.environ["WANDB_API_KEY"] = 'b186daff8b1df891791510b544e35c66c8731ac7'
parser = argument_parser()
args = parser.parse_args()

def main():
    # wandb.init(project="tamn")
    # wandb.config.update(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print('You deserve a gpu!')
        return
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print('Initializing data manager')
    dm = DataManager(args)
    train_loader = dm.create_dataloader(phase='train',mode='few-shot')
    val_loader = dm.create_dataloader(phase='val', mode='few-shot')
    model = TaskAwareMetricNetwork(args=args)
    # wandb.watch(model)
    criterion = CrossEntropyLoss(args)
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, 5e-04)

    model = model.cuda()

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")
    learning_rate = args.lr
    for epoch in range(args.max_epoch):
        if args.optim=='Adam':
            if epoch >49:
                learning_rate = max(1e-6, learning_rate*0.9)
                for params in optimizer.param_groups:
                    params['lr'] = learning_rate
        elif args.optim=='SGD':
            for (stepvalue, base_lr) in args.LUT_lr:
                if epoch < stepvalue:
                    lr = base_lr
                    break
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        start_train_time = time.time()
        train(epoch, train_loader, model, criterion, optimizer, args)
        train_time += round(time.time()-start_train_time)

        if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
            # When epoch<60, evaluate the performance on the val set every 10 epochs. When epoch>60, do evaluations after each epoch
            acc = evaluate(epoch, val_loader, model, args)
            is_best = acc > best_acc

            if is_best:
                best_acc = acc
                # wandb.run.summary["best val accuracy"] = best_acc
                best_epoch = epoch+1
            
            save_checkpoint({
                'state_dict': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'epoch.'+str(epoch+1) + 'pth.tar'))
            print("==> Test {}-way Best accuracy {:.2%}, achieved at epoch {}".format(args.way, best_acc, best_epoch))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))

def train(epoch, dataloader, model, criterion, optimizer, args):
    accs = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    end = time.time()
    for batch_idx, (images_spt, labels_spt, images_qry, labels_qry, cids_qry) in enumerate(dataloader):
        data_time.update(time.time()-end)

        images_spt, labels_spt = images_spt.cuda(), labels_spt.cuda() # [4, 5, 3, 84, 84] [4, 5]
        images_qry, labels_qry = images_qry.cuda(), labels_qry.cuda() # [4, 30, 3, 84, 84] [4, 30]
        cids_qry = cids_qry.cuda() # [4, 30]

        labels_spt_1hot = F.one_hot(labels_spt).cuda()
        labels_qry_1hot = F.one_hot(labels_qry).cuda()
        cids_qry_1hot = F.one_hot(cids_qry, args.train_categories).cuda()

        preds = model(images_spt, labels_spt_1hot, images_qry, labels_qry_1hot, cids_qry_1hot)
        loss = criterion(preds, labels_qry_1hot, cids_qry_1hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), cids_qry.size()[0]*cids_qry.size()[1])
        batch_time.update(time.time() - end)
        end = time.time()
    # wandb.log({'epoch': epoch+1, 'train loss': losses.avg}, step=epoch+1)
    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
           epoch+1, batch_time=batch_time, 
           data_time=data_time, loss=losses))

def evaluate(epoch, dataloader, model, args):
    accs = AverageMeter()
    losses = AverageMeter()
    test_accuracies = []
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images_spt, labels_spt, images_qry, labels_qry, cids_qry) in enumerate(dataloader):
            batch_size = images_spt.size()[0]
            images_spt, labels_spt = images_spt.cuda(), labels_spt.cuda()
            images_qry, labels_qry = images_qry.cuda(), labels_qry.cuda()

            labels_spt_1hot = F.one_hot(labels_spt).cuda()
            labels_qry_1hot = F.one_hot(labels_qry).cuda()

            val_logits, preds = model(images_spt, labels_spt_1hot, images_qry, labels_qry_1hot)
            val_loss = criterion(val_logits.view(-1, val_logits.size(-1)), labels_qry.view(-1))
            losses.update(val_loss.item(), batch_size*labels_qry.size()[1])
            acc, acc_per_task = calculate_acc(preds, labels_qry)
            accs.update(acc.item(), batch_size*labels_qry.size()[1])
            acc_per_task = np.reshape(acc_per_task.numpy(), (batch_size))
            test_accuracies.append(acc_per_task)

    accuracy = accs.avg
    loss = losses.avg
    test_accuracies = np.reshape(np.array(test_accuracies), -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.num_episode_val)
    print('Accuracy: {:.2%}, std: {:.2%}'.format(accuracy, ci95))
    # wandb.log({'val acc': accuracy, 'val loss': loss}, step=epoch+1)
    return accuracy


if __name__ == '__main__':
    main()


