# Salient instance inference Multiple Instance Learning (SiiMIL)
This is the offical implementation of our paper: Attention2Minority: A salient instance inference-based multiple instance learning for classifying small lesions in breast cancer whole slide images.

## Extract foreground patches coordinates
```$ python extraction.py --slidedir <>```

## Patch encoding using Resnet50
```$ python encoding_pts.py --slidedir <>```

## Representation learning from negative instances
```$ python keyset_lrn.py -t 100```

## Salient instance inference
```$ python sii.py -k 150```

## Train
```$ python train_cv.py -r 0.3 --keys sm_sort.npy --code cam16res_siimil --data cam16_sii```

## Evaluation
```$ python eval_cv.py -r 0.3 --keys sm_sort.npy --code cam16res_siimil --data cam16_sii```
