Progressive Growing GAN-PyTorch
============================
A Pytorch implementation of Progressive Growing GAN based on the paper [Progressive Growing of GANs for Improved Quality, Stability, and Variation
](https://arxiv.org/abs/1710.10196).

Requirement
----------------------------
* Argparse
* Numpy
* Matplotlib
* Pillow
* Python 3.7
* PyTorch
* TorchVision
* tqdm


Usage
----------------------------

### Training

Run the script train.py to train the network with CelebA dataset.
```
$ python train.py --h    

usage: train.py [-h] [--root ROOT] [--epochs EPOCHS] [--out_res OUT_RES]
                [--resume RESUME] [--cuda]

optional arguments:
  -h, --help         show this help message and exit
  --root ROOT        directory contrains the data and outputs
  --epochs EPOCHS    training epoch number
  --out_res OUT_RES  The resolution of final output image
  --resume RESUME    continues from epoch number
  --cuda             Using GPU to train
```

### Testing

Download the [weight](https://drive.google.com/file/d/1iZO8IGLXOQmAvkcUPvNGSjE_01tRnUN-/view?usp=sharing) to generate 128x128 faces.

Run the script generate_sample.py

```
$ python generate_sample.py -h               

usage: generate_sample.py [-h] [--seed SEED] [--out_dir OUT_DIR]
                          [--num_imgs NUM_IMGS] [--weight WEIGHT]
                          [--out_res OUT_RES] [--cuda]

optional arguments:
  -h, --help           show this help message and exit
  --seed SEED          Seed for generate images
  --out_dir OUT_DIR    Directory for the output images
  --num_imgs NUM_IMGS  Number of images to generate
  --weight WEIGHT      Generator weight
  --out_res OUT_RES    The resolution of final output image
  --cuda               Using GPU to train
```


Sample Results
----------------------------



