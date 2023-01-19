# Salient instance inference Multiple Instance Learning (SiiMIL)
This is the offical implementation of our paper: Attention2Minority: A salient instance inference-based multiple instance learning for classifying small lesions in breast cancer whole slide images.

## Representation learn ing from negative instances


## Salient instance inference

## Train
```$ python train_cv.py -r 0.3 --keys sm_cur100_k150.npy --code cam16res_cur100_k150_cos_r0.3 --data cam16_curcos```

## Evaluation
```$ python eval_cv.py -r 0.3 --keys sm_cur100_k150.npy --code cam16res_cur100_k150_cos_r0.3 --data cam16_clucos```
