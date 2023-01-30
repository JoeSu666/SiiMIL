import argparse
import os
from os.path import join
import numpy as np
import json
import glob

parser = argparse.ArgumentParser(description='Q-K compare and sort')
parser.add_argument('--qdir', default='./data/feats/cam16res', type=str, help='Query data (feats from all WSIs)')
parser.add_argument('--kdir', default='./data/keys/cur-100.npy', type=str, help='Key data')
parser.add_argument('-k', default=150, type=int, help="top-k cosine values across the keys dimension in the cosine similarity matrix")
parser.add_argument('--name', default='sm_sort.npy')

args = parser.parse_args()

def cossim(Q, K):
    '''
    build cosine similarity matrix
    '''
    K_norm = np.linalg.norm(K, axis=1, keepdims=True)
    Q_norm = np.linalg.norm(Q, axis=1, keepdims=True)
    QK_norm = Q_norm @ K_norm.T
    QK = Q @ K.T

    return QK / QK_norm

def main(args=args):
    keys = np.load(args.kdir)
    namelist = sorted(glob.glob(join(args.qdir, '*/*/*.npy')))
    sortdict = {}
    for name in namelist:
        print(name)
        query = np.load(name)
        cos_SA = cossim(query, keys)
        cos_SA = np.sort(cos_SA, axis=1)
        cos_SA = np.mean(cos_SA[:, -args.k:], axis=1)
        sortidx = np.argsort(cos_SA)
        sortdict[name.split('/')[-1].split('.')[0]] = sortidx

    np.save(join('./data/keys', args.name), sortdict)

if __name__ == '__main__':
    main(args)
          
