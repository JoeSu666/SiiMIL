import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import torch
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def plotmap(mask, coords, values, ps, mlevel, plevel):
    shape = mask.shape
    img = np.zeros(shape)

    instclist = []
    distlist = []
    
    r = 2 ** (mlevel - plevel)

    for i in range(values.shape[0]):
        value = values[i]
        coord = coords[i]
        img[coord[0]:coord[0]+round(ps/r), coord[1]:coord[1]+round(ps/r)] = value
        
    return img

def assign(labeldf, name):
    if name.split('/')[-1].split('_')[0] == 'normal':
        train = 'train'
        label = 'normal'
    elif name.split('/')[-1].split('_')[0] == 'tumor':
        train = 'train'
        label = 'tumor'
    elif name.split('/')[-1].split('_')[0] == 'test':
        train = 'test'
        if labeldf.loc[name.split('/')[-1].split('.')[0]][1] == 'Normal':
            label = 'normal'
        elif labeldf.loc[name.split('/')[-1].split('.')[0]][1] == 'Tumor':
            label = 'tumor'

    return train, label

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, verbose=False, save_dir='', args=None):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 20
#             stop_epoch (int): Earliest epoch possible for stopping
#             verbose (bool): If True, prints a message for each validation loss improvement. 
#                             Default: False
#         """
#         self.patience = args.patience
#         self.args = args
#         self.stop_epoch = args.stop_epoch
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.monitor = args.monitor
#         self.save_dir = save_dir

#     def __call__(self, epoch, value, model, optimizer):
#         if self.monitor == 'loss':
#             score = -value
#         else:
#             score = value

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint({
#                 'epoch': epoch,
#                 'arch': self.args.arch,
#                 'state_dict': model.state_dict(),
#                 'best_score': self.best_score,
#                 'optimizer' : optimizer.state_dict(),
#             }, True, filename=os.path.join(self.save_dir, str(epoch) + 'ep_' + 'checkpoint.pth.tar'))
#         elif score < self.best_score:
#                 self.counter += 1
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#                 if self.counter >= self.patience and epoch > self.stop_epoch:
#                     self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint({
#                 'epoch': epoch,
#                 'arch': self.args.arch,
#                 'state_dict': model.state_dict(),
#                 'best_score': self.best_score,
#                 'optimizer' : optimizer.state_dict(),
#             }, True, filename=os.path.join(self.save_dir, str(epoch) + 'ep_' + 'checkpoint.pth.tar'))

#             self.counter = 0

#     def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
#         '''Saves model when validation loss decrease.'''
#         torch.save(state, filename)
#         if is_best:
#             shutil.copyfile(filename, os.path.join(self.save_dir, 'model_best.pth.tar'))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, verbose=False, save_dir='', args=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = args.patience
        self.args = args
        self.stop_epoch = args.stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor = args.monitor
        self.save_dir = save_dir

    def __call__(self, epoch, value, model, optimizer):
        if self.monitor == 'loss':
            score = -value
        else:
            score = value

        if epoch == self.stop_epoch:
            self.best_score = score
            self.save_checkpoint({
                'epoch': epoch,
                'arch': self.args.arch,
                'state_dict': model.state_dict(),
                'best_score': self.best_score,
                'optimizer' : optimizer.state_dict(),
            }, True, filename=os.path.join(self.save_dir, str(epoch) + 'ep_' + 'checkpoint.pth.tar'))
        elif epoch > self.stop_epoch:
            if score <= self.best_score:           
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.save_checkpoint({
                'epoch': epoch,
                'arch': self.args.arch,
                'state_dict': model.state_dict(),
                'best_score': self.best_score,
                'optimizer' : optimizer.state_dict(),
            }, False, filename=os.path.join(self.save_dir, str(epoch) + 'ep_' + 'checkpoint.pth.tar'))
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint({
                    'epoch': epoch,
                    'arch': self.args.arch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score,
                    'optimizer' : optimizer.state_dict(),
                }, True, filename=os.path.join(self.save_dir, str(epoch) + 'ep_' + 'checkpoint.pth.tar'))

                self.counter = 0

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
        '''Saves model when validation loss decrease.'''
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.save_dir, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, threshold, test=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        scores = torch.sigmoid(output).cpu().numpy()
        pred = (torch.sigmoid(output) > threshold).float().view(-1, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))

        if test:
            auc = roc_auc_score(target.view(-1).cpu().numpy(), scores)

        tp = torch.sum(torch.logical_and(correct, pred == 1)).float()
        tn = torch.sum(torch.logical_and(correct, pred == 0)).float()

        res = []

        correct = correct.reshape(-1).float().sum(0, keepdim=True)
        acc = correct.mul_(100.0 / batch_size)
        res.append(acc.view(-1).cpu().numpy()[0])
        recall = tp / torch.sum(target == 1).float()

        spe = tn / torch.sum(target == 0).float()

        res.append(recall.cpu().numpy()*100)
        res.append(spe.cpu().numpy()*100)
        if test:
            res.append(auc*100)
        return res

def eval_accuracy(output, target, threshold, test=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        scores = torch.sigmoid(output).cpu().numpy()
        pred = (torch.sigmoid(output) > threshold).float().view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()

        auc = roc_auc_score(target, scores)
        acc = accuracy_score(target, pred)
        precision = precision_score(target, pred)
        recall = recall_score(target, pred)
        f1 = f1_score(target, pred)

        res = [auc*100, acc*100, precision*100, recall*100, f1*100]

        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr                

def get_attention(feats, model):

    model.eval()
    with torch.no_grad():
        feats = np.expand_dims(feats, 0)
        feats = torch.Tensor(feats).cuda()
        _, A = model(feats)

        A = A.view(-1).cpu().numpy()

    return A