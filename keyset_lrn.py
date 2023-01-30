import argparse
import os
from os.path import join
import glob
import numpy as np
import h5py

parser = argparse.ArgumentParser(description='cam16')
parser.add_argument('--featdir', default='./data/feats/cam16res/train/normal', type=str,
                    help='path to the feats of normal WSIs in trainng set')
parser.add_argument('--savedir', default='./data/keys/cur-', type=str,
                    help='path to save learned key set')
parser.add_argument('-t', default=100, type=int,
                    help='maximum number of keys from each normal WSI')       


def extractTopKColumns(matrix):
    '''
    Learn representative negative instances from each normal WSI
    '''
    score  = {}
    rank = np.linalg.matrix_rank(matrix)
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    
    for j in range(0, matrix.shape[1]):
        cscore = sum(np.square(vh[0:rank,j]))
        cscore /= rank
        score[j] = min(1, rank*cscore)
        
    prominentColumns = sorted(score, key=score.get, reverse=True)[:rank]
    #Removal of extra dimension\n",
    C = np.squeeze(matrix[:, [prominentColumns]])
    
    return ({"columns": prominentColumns, "matrix": C, "scores": sorted(score.values(), reverse = True)[:rank]})

def run(args):
    namelist = sorted(glob.glob(join(args.featdir, '*.npy')))
    keyset = np.empty((0, 1024))

    for name in namelist[:5]:
        print(name)
        feats = np.load(name).T
        res = extractTopKColumns(feats)
        keys = np.transpose(res["matrix"])
        cols = np.transpose(res["columns"])
        scores = np.transpose(res["scores"])
        
        length = keys.shape[0]
        if length <= args.t:
            keyset.resize([keyset.shape[0]+length, 1024])
            keyset[-length:] = keys
        else:
            keyset.resize([keyset.shape[0]+args.t, 1024])
            keyset[-args.t:] = keys[:args.t]
        # np.save(args + name.split('/')[-1].split(".")[0] + ".npy", CTrans)
        # np.save(curStorageDir + name.split('/')[-1].split(".")[0] + "_cols.npy", cols)
        # np.save(curStorageDir + name.split('/')[-1].split(".")[0] + "_scores.npy", scores)
    
    np.save(args.savedir + str(args.t) + '.npy', keyset)

if __name__ == '__main__':
    args = parser.parse_args()

    run(args)
