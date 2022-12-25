import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from os.path import join
import argparse
from tqdm import tqdm

import utils

parser = argparse.ArgumentParser(description='save patch coordinates into npy. Coords under certain level 5. \
                                                (IN (row, col) INDEXING, NOT (x, y))')
parser.add_argument('-p', default=224, type=int, help='Patch size')
parser.add_argument('-s', default=224, type=int, help='Stride')
parser.add_argument('-l', default=1, type=int, help='Magnification level')
parser.add_argument('--save', default='data', type=str, help='Saving directory')
parser.add_argument('--data', default='/isilon/datalake/cialab/original/cialab/image_database/d00142/testing/images/*.tif', 
                    type=str, help='Data directory')
parser.add_argument('--mask', default='../data/mask_level5', 
                    type=str, help='mask directory')  
parser.add_argument('--type', default=None, 
                    type=str, help='None (default, tissue region), normal or tumor')                     
parser.add_argument('--code', default='cam16', type=str, help='code')                                      


args = parser.parse_args()

def main(args=args):
    namelist = sorted(glob.glob(args.data))

    save_dir = join(args.save, 'pts', args.code+'l'+str(args.l)+'p'+str(args.p)+'s'+str(args.s))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for name in namelist:
        
        # read mask
        pid = name.split('/')[-1].split('.')[0]
        if args.type == 'tumor':
            pid = pid + 't'
        elif args.type == 'normal':
            pid = pid + 'n'
        else:
            pass
        mask = cv2.imread(join(args.mask, pid+'.png'), 0)

        
        if args.type == None:
            if os.path.exists(join(args.mask, pid+'t.png')):
                maskt = cv2.imread(join(args.mask, pid+'t.png'), 0)
            else:
                maskt = np.zeros(mask.shape)
        elif args.type == 'tumor':
            maskt = cv2.imread(join(args.mask, pid+'t.png'), 0)
        elif args.type == 'normal':
            maskt = np.zeros(mask.shape)

        print('***********')
        print(pid)

        # load wsi
        with openslide.OpenSlide(name) as fp:
            w, h = fp.level_dimensions[args.l]
            w0, h0 = fp.dimensions

            #extract coords
            pts, labels = extract(fp, w, h, w0, h0, args.p, args.s, mask, maskt, args.l)

        # save pts
        np.save(join(save_dir, pid+'.npy'), pts)
        np.save(join(save_dir, pid+'labels.npy'), labels)



def extract(fp, w, h, w0, h0, ps, stride, mask, maskt, level):

    boundaries = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    boundaries = boundaries[0]
    minw, minh, maxw, maxh = mask.shape[1], mask.shape[0], 0, 0
    for i in range(len(boundaries)):
        b = np.squeeze(boundaries[i], 1)
        if np.min(b[:, 0]) < minw:
            minw = np.min(b[:, 0])
        if np.min(b[:, 1]) < minh:
            minh = np.min(b[:, 1])
        if np.max(b[:, 0]) > maxw:
            maxw = np.max(b[:, 0])
        if np.max(b[:, 1]) > maxh:
            maxh = np.max(b[:, 1])

    psy = ps * mask.shape[0] / float(h)
    psx = ps * mask.shape[1] / float(w)

    stride = stride * mask.shape[0] / float(h)

    # Grid of points
    ys = np.arange(minh, maxh, stride)
    xs = np.arange(minw, maxw, stride)

    [ys, xs] = np.meshgrid(ys, xs, indexing='ij')
    ys = ys.reshape((-1, 1))
    xs = xs.reshape((-1, 1))
    pts = np.concatenate([ys, xs], 1)

    # Here's where we put things
    bag = np.zeros((ps, ps, 3, pts.shape[0]), 'uint8')
    keep = np.zeros((pts.shape[0],), dtype=bool)
    labels = np.zeros((pts.shape[0],), dtype='int')

    for p in range(pts.shape[0]):
        # Query pts
        rx = pts[p, 1]
        ry = pts[p, 0]

        # Pass first one
        if p == 0:
            continue
        
        # Checks if inside the mask image
        if mask.shape[1] > rx + psx and mask.shape[0] > ry + psy and \
        mask[round(ry+psy/2), round(rx+psx/2)] == 255 and \
        mask[round(ry), round(rx)] == 255 and \
        mask[round(ry+psy), round(rx)] == 255 and \
        mask[round(ry), round(rx+psx)] == 255:
            
            # im = np.array(fp.read_region((round(rx*float(w0)/mask.shape[1]), round(ry*float(h0)/mask.shape[0])), level, (ps, ps)))
            # plt.imsave('data/'+str(p)+'.jpg', im)
            pts[p] = np.array([round(ry*float(h0)/mask.shape[0]), round(rx*float(w0)/mask.shape[1])])

            keep[p] = True

            # Create patch level labels by checking tumor masks
            if maskt[round(ry+psy/2), round(rx+psx/2)] == 255:
                labels[p] = 1

    print('{} patches in bounding box'.format(pts.shape[0]))
    pts = pts[keep]
    labels = labels[keep]
    print('Found {} tissue patches'.format(pts.shape[0]))

    return np.around(pts).astype('int'), labels

if __name__ == '__main__':
    main()
