import numpy as np
import glob
import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import openslide

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import utils
import mymodel.resnet_custom as resnet

parser = argparse.ArgumentParser(description='outliers data preprocessing')
parser.add_argument('--save', default='data/feats/cam16res2', type=str, help='Saving directory')
parser.add_argument('--pts', default='data/pts/cam16l1p224s224/', type=str, help='Data directory')
parser.add_argument('--arch', default='resnet50_baseline', type=str, help='model')
# parser.add_argument('--pretrain', default='runs/cam16Pn_IN/checkpoint_0019.pth.tar', type=str, help='path to pretrained model')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

args = parser.parse_args()
if not os.path.exists(args.save):
    os.mkdir(args.save)
    os.mkdir(join(args.save, 'train'))
    os.mkdir(join(args.save, 'train', 'normal'))
    os.mkdir(join(args.save, 'train', 'tumor'))
    os.mkdir(join(args.save, 'test'))
    os.mkdir(join(args.save, 'test', 'normal'))
    os.mkdir(join(args.save, 'test', 'tumor'))

labeldf = pd.read_csv('./reference.csv', index_col=0, header=None)

def main(args=args):
    namelist = sorted(glob.glob(args.pts+'*.npy'))
    model = load_resnet50()
    model.eval()
    with torch.no_grad():
        for name in namelist:

            # load files
            train, label = utils.assign(labeldf, name)
            pid = name.split('/')[-1].split('.')[0]
            print('**********')
            print(pid)
            pts = np.load(name)
            print('{} patches'.format(pts.shape[0]))
            if train == 'train':
                slidename = join('/isilon/datalake/cialab/original/cialab/image_database/d00142', train+'ing', label, pid+'.tif')
            elif train == 'test':
                slidename = join('/isilon/datalake/cialab/original/cialab/image_database/d00142', train+'ing', 'images', pid+'.tif')

            with openslide.OpenSlide(slidename) as fp:
                dataloader = load_dataset(fp, pts)
                feats = np.empty((0,1024), dtype='float32')
                for images in dataloader:                    
                    images = images.cuda()

                    z1 = model(images)
                    z1 = z1.cpu().numpy()

                    feats.resize((feats.shape[0]+z1.shape[0], 1024))
                    feats[-z1.shape[0]:] = z1

            save_dir = join(args.save, train, label, pid+'.npy')
            np.save(save_dir, feats)
            print('{}x{} bag saved in '.format(feats.shape[0], feats.shape[1]), save_dir)



class cam16P(Dataset):
    def __init__(self, fp, pts, transform=None):
        
        self.pts = pts
        self.fp = fp
        self.transform = transform

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        pt = self.pts[idx]
        image = self.fp.read_region((pt[1], pt[0]), 1, (224, 224)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image

def load_model(pretrain, arch='resnet50'):
    pretrain = pretrain
    checkpoint = torch.load(pretrain, map_location="cpu")

    model = getattr(resnet, arch)(pretrained=True)

    # model = simsiam.builder.SimSiam(
    #     getattr(resnet, args.arch)(pretrained=True),
    #     1024, 512)

    # model = simsiam.builder.SimSiam(
    #     models.__dict__[arch],
    #     2048, 512)

    for name, param in model.named_parameters():
        param.requires_grad = False

    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder.0.'):
            # remove prefix
            state_dict[k[len("module.encoder.0."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg.missing_keys)
    model = model.cuda()

    return model

def load_resnet50(pretrained=True):
    model = resnet.resnet50_baseline(pretrained=True)
    model.cuda()

    return model

def load_dataset(fp, pts):
    test_dataset = cam16P(fp=fp, pts=pts, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=50, shuffle=False,
            num_workers=16, pin_memory=True)
    
    return test_loader


if __name__ == '__main__':
    main()