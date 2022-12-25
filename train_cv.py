import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import math
import random
import time
import mymodel.model as model
import numpy as np
import glob
import shutil
import pandas as pd
from PIL import Image
import mydataset.mydataset as mydataset
import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='cam16')
parser.add_argument('--model', default='', type=str,
                    help='path to the model')
parser.add_argument('--arch', default='attmil', type=str,
                    help='architecture')
parser.add_argument('--data', default='cam16_curcos', type=str,
                    help='dataset')                    
parser.add_argument('--split', default=42, type=int,
                    help='split random seed')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='batch size')          
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs')   
parser.add_argument('--inputd', default=1024, type=int,
                    help='input dim')                                      
parser.add_argument('--code', default='test', type=str,
                    help='exp code')                        
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='accuracy threshold') 
parser.add_argument('--lr', default=2e-4, type=float, 
                    help='init learning rate')   
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--freq', default=100, type=int, 
                    help='training log display frequency')                              
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')  
parser.add_argument('--pretrained', default='', type=str, 
                    help='pretrained model for validate')    

parser.add_argument('--patience', default=10, type=int, 
                    help='early stopping patience') 
parser.add_argument('--stop_epoch', default=30, type=int, 
                    help='start epoch to activate early stopping counting') 
parser.add_argument('--monitor', default='loss', type=str, 
                    help='value to monitor for early stopping')                                                                                             

parser.add_argument('-r', default=0.3, type=float, help='rate of selected query patches')
parser.add_argument('--keys', default='', type=str, help='rate of selected query patches')
parser.add_argument('--sd', default=None, type=int, help='use sepctrum decoupling')

def run(args):
    

    net = getattr(model, args.arch)(inputd=args.inputd)

    # BCE loss, Adam opt
    criterion = nn.BCEWithLogitsLoss().cuda('cuda')


    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))


    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.reg)

    train_dataset = getattr(mydataset, args.data)(train='train', r=args.r, keys=args.keys, split=args.split)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=32, pin_memory=True)

    val_dataset = getattr(mydataset, args.data)(train='val', r=args.r, keys=args.keys, split=args.split)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)

    test_dataset = getattr(mydataset, args.data)(train='test', r=args.r, keys=args.keys, split=args.split)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)


    net.cuda()
    writer = SummaryWriter(os.path.join(args.save, time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())))
    
    # set-up early stopping
    EarlyStopping = utils.EarlyStopping(save_dir=args.save, args=args)
    monitor_values = {'acc':0, 'auc':1, 'loss':4}
    monitor_idx = monitor_values[args.monitor]
    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set (acc, auc, sen, spe, loss)
        metrics = validate(val_loader, net, epoch, criterion, args, writer, 'val')

        # early stopping
        EarlyStopping(epoch, metrics[monitor_idx], net, optimizer)

        # evaluate on testing set
        if EarlyStopping.early_stop:
            _ = validate(test_loader, net, epoch, criterion, args, writer, 'test')
            print('****Early stop at epoch:{}'.format(epoch-args.patience))
            break
        else:
            if EarlyStopping.counter == 0:
                best_metrics = validate(test_loader, net, epoch, criterion, args, writer, 'test')
                best_epoch = epoch
            else:
                _ = validate(test_loader, net, epoch, criterion, args, writer, 'test')

    print('****best testing result: epoch: {}, acc: {}, auc: {}, sen: {}, spe: {}, loss: {}'.format(best_epoch, best_metrics[0], best_metrics[1], \
    best_metrics[2], best_metrics[3], best_metrics[4]))

def train(train_loader, model, criterion,  optimizer, epoch, args, writer):
    losses = utils.AverageMeter('Loss', ':.4e')

    progress = utils.ProgressMeter(
        len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda().float()

        output, _ = model(images)
        output = output.view(-1).float()
        if i == 0:
            outputs = output
            targets = target
        else:
            outputs = torch.cat((outputs, output), 0)
            targets = torch.cat((targets, target), 0)

        loss = criterion(output, target)
        if args.sd:
            loss += (args.sd/2) * torch.norm(output).pow(2)
        
        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.freq == 0:
            progress.display(i)

    acc, sen, spe = utils.accuracy(outputs, targets, args.threshold, False)

    if writer:
        writer.add_scalar("Loss/train", losses.avg, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        writer.add_scalar("sen/train", sen, epoch)
        writer.add_scalar("spe/train", spe, epoch)


def validate(val_loader, model, epoch, criterion, args, writer, val='val'):
    losses = utils.AverageMeter('Loss', ':.4e')

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):    
            images = images.cuda()
            target = target.cuda().float()

            output, _ = model(images)
            output = output.view(-1).float()
            if i == 0:
                outputs = output
                targets = target
            else:
                outputs = torch.cat((outputs, output), 0)
                targets = torch.cat((targets, target), 0)
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))

    acc, sen, spe, auc = utils.accuracy(outputs, targets, args.threshold)

    if val == 'val':
        print(' **Validation Acc {acc:.3f} sen {sen:.3f} spe {spe:.3f} AUC {auc:.3f} LOSS {loss:.3f}'
            .format(acc=acc, sen=sen, spe=spe, auc=auc, loss=losses.avg))
    else:
        print(' ***Testing Acc {acc:.3f} sen {sen:.3f} spe {spe:.3f} AUC {auc:.3f} LOSS {loss:.3f}'
            .format(acc=acc, sen=sen, spe=spe, auc=auc, loss=losses.avg))

    if writer:
        writer.add_scalar("Loss/"+val, losses.avg, epoch)
        writer.add_scalar("Accuracy/"+val, acc, epoch)
        writer.add_scalar("sen/"+val, sen, epoch)
        writer.add_scalar("spe/"+val, spe, epoch)
        writer.add_scalar("auc/"+val, auc, epoch)

    return acc, auc, sen, spe, losses.avg




if __name__ == '__main__':
    args = parser.parse_args()
    if args.sd:
        print('spectrum decoupling')
        
    save_dir = './runs/slidelevel/'+args.code
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    

    for i in range(42, 47):       
        args.split = i       
        args.save = os.path.join(save_dir, str(args.split))                  
        run(args)