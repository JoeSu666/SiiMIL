# Salient instance inference Multiple Instance Learning (SiiMIL)
This is the offical implementation of our paper: Attention2Minority: A salient instance inference-based multiple instance learning for classifying small lesions in breast cancer whole slide images. [Paper](https://arxiv.org/abs/2301.07700)

## Requirements
Camelyon16 dataset, torch, torchvision, tensorboard, openslide, PIL, pandas, numpy, scikit-learn, tqdm, opencv

## Extract foreground patches coordinates
Extract from CAM16 raw slides:
```$ python extraction.py --slidedir <>```
Or use your own:
```
data
    ├── pts
          ├── cam16l1p224s224
                            ├── slide_1.npy
                            ├── slide_2.npy
                            └── ...
```

## Patch encoding using Resnet50
Encoding from CAM16 raw slides:
```$ python encoding_pts.py --slidedir <>```
Or use your own:
```
data
   ├── feats
           ├── cam16res
                      ├── train
                              ├── normal
                                       ├── slide_1.npy
                                       ├── slide_2.npy
                                       └── ...
                              └── tumor
                                      └── ...
                      └── test
                             ├── normal
                                      └── ...
                             └── tumor
                                     └── ...
```      

## Representation learning from negative instances
Learn representative negative instances (i.e., Key set)
```$ python keyset_lrn.py -t 100```

Or [download](https://drive.google.com/file/d/1jfNuKoPyWypryKbcWOKzODoIougw1byy/view?usp=share_link) the learned key set.


## Salient instance inference
```$ python sii.py -k 150```

## Train
```$ python train_cv.py -r 0.3 --keys sm_sort.npy --code cam16res_siimil --data cam16_sii```

[Download](https://drive.google.com/file/d/1SqsOrj2vO0MEQycSKm_sh3Y32FnDhGbt/view?usp=share_link) pretrained model.

## Evaluation
```$ python eval_cv.py -r 0.3 --keys sm_sort.npy --code cam16res_siimil --data cam16_sii```
