import numpy as np
import glob
import os
from os.path import join
import random
import h5py
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class cam16_sii(Dataset):
    def __init__(self, train='train', transform=None, r=None, keys='', split=42):

        self.img_dir = './data/feats/cam16res'

        self.split = split
        self.r = r
        self.keys = keys
        
        postrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'tumor', '*.npy'))
        negtrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'normal', '*.npy'))

        postrainlist, posvallist = train_test_split(postrainlist, test_size=0.10, random_state=self.split)
        negtrainlist, negvallist = train_test_split(negtrainlist, test_size=0.10, random_state=self.split)
        testnamelist = glob.glob(os.path.join(self.img_dir, 'test', '*', '*.npy'))

        if train == 'train':
            self.img_names = postrainlist + negtrainlist
        elif train == 'test':
            self.img_names = testnamelist
        elif train == 'val':
            self.img_names = posvallist + negvallist
            
        self.transform = transform
        self.sortdict = np.load(join('./data/keys', self.keys), allow_pickle=True).item()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = np.load(img_path)
        sortidx = self.sortdict[img_path.split('/')[-1].split('.')[0]]

        label = img_path.split('/')[-2]
        if label == 'tumor':
            label = 1
        elif label == 'normal':
            label = 0
            
        n = int(self.r * sortidx.shape[0])
        return torch.Tensor(image[sortidx[:n]]), label
