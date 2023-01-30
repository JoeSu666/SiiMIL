import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import argparse
import os
from os.path import join
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
parser.add_argument('--inputd', default=1024, type=int,
                    help='input dim')                                      
parser.add_argument('--code', default='siimil', type=str,
                    help='exp code')                        
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='accuracy threshold') 
parser.add_argument('--pretrained', default='model_best.pth.tar', type=str, 
                    help='pretrained model for validate')   
parser.add_argument('--seed', default=7, type=int, 
                    help='random seed')  
parser.add_argument('--folds', default=1, type=int, 
                    help='number of cv folds')   
                                                                                        
parser.add_argument('-r', default=0.3, type=float, help='rate of selected query patches')
parser.add_argument('--keys', default='sm_sort.npy', type=str, help='rate of selected query patches')

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run(args):

    net = getattr(model, args.arch)(inputd=args.inputd)
    criterion = nn.BCEWithLogitsLoss().cuda('cuda')

    val_dataset = getattr(mydataset, args.data)(train='val', r=args.r, keys=args.keys, split=args.split)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)

    test_dataset = getattr(mydataset, args.data)(train='test', r=args.r, keys=args.keys, split=args.split)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)


    print('load model from: ', join(args.save, args.pretrained))
    checkpoint = torch.load(join(args.save, args.pretrained), map_location="cpu")

    state_dict = checkpoint['state_dict']
    msg = net.load_state_dict(state_dict, strict=True)
    print(msg.missing_keys)
    net.cuda()
    
    val_metrics = validate(val_loader, net, criterion, args.threshold, test=False)
    test_metrics = validate(test_loader, net, criterion, args.threshold, test=True)

    return test_metrics, val_metrics


def validate(val_loader, model, criterion, threshold, test):
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

    auc, acc, precision, recall, f1 = utils.eval_accuracy(outputs, targets, threshold)

    if not test:
        print(' ***Validation AUC {:.3f} ACC {:.3f} precision {:.3f} recall {:.3f} f1 {:.3f}'
            .format(auc, acc, precision, recall, f1))
    else:
        print(' ***Testing AUC {:.3f} ACC {:.3f} precision {:.3f} recall {:.3f} f1 {:.3f}'
            .format(auc, acc, precision, recall, f1))

    return auc, acc, precision, recall, f1, losses.avg, torch.sigmoid(outputs).cpu().numpy(), targets.cpu().numpy()


if __name__ == '__main__':
    args = parser.parse_args()
    save_dir = './runs/slidelevel/'+args.code   
    seed_torch(args.seed)
    val_aucs = []
    aucs, accs, precisions, recalls, f1s = [], [], [], [], []
    results = {}
    testnamelist = glob.glob(join('./data/feats/cam16res', 'test', '*', '*.npy'))
    testnamelist = [name.split('/')[-1].split('.')[0] for name in testnamelist]

    for i in range(42, 42+args.folds):       
        args.split = i       
        args.save = os.path.join(save_dir, str(args.split))                  
        test_metrics, val_metrics = run(args)
        aucs.append(test_metrics[0])
        accs.append(test_metrics[1])
        precisions.append(test_metrics[2])
        recalls.append(test_metrics[3])
        f1s.append(test_metrics[4])
        val_aucs.append(val_metrics[0])
        if i == 42:
            results['gts'] = test_metrics[-1]
        results[str(i)+'outputs'] = test_metrics[-2]
        
    results = pd.DataFrame(results, index=testnamelist)
    results.to_csv(join(save_dir, 'results.csv'))
    print('')
    print('========================================================')
    print('*Val AUC {:.3f} +- {:.3f}'.format(np.mean(val_aucs), np.std(val_aucs)))

    print('**Testing AUC {:.3f} +- {:.3f}, ACC {:.3f} +- {:.3f}, PRECISION {:.3f} +- {:.3f}, RECALL {:.3f} +- {:.3f}, F1 {:.3F} +- {:.3F}'
        .format(np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs), np.mean(precisions), np.std(precisions), 
        np.mean(recalls), np.std(recalls), np.mean(f1s), np.std(f1s)))

    print('Saved in ', join(save_dir, 'results.csv'))