# Intro
This is final project code for CMU's course 16-726 Learning-Based Image Synthesis, Spring 2025.

## Dataset
To create a model that takes input images and attempts to change the hairstyle, one needs to first download the dataset from https://psh01087.github.io/K-Hairstyle/

Have the dataset in the directory outside of the code

## Training
### Style Identifier
First training needs to be done on the style identifier. We've attempted style identification training on a fresh model ([style_identifier.py](style_identifier.py)) as well as a pretrained resnet model ([pretrained_style_identifier.py](pretrained_style_identifier.py)):
```
python pretrained_style_identifier.py
```
### Style Generator

Training on the GAN model ([style_GAN.py](style_GAN.py)) can be specified by inputting which identifier to use, were we have the options of "ours", "pretrained", and "clip". 
```
python style_GAN.py

python style_GAN.py --iden_checkpoint_dir checkpoints_pretrained_style_id --iden pretrained

python style_GAN.py  --iden cycle
```
Generated images at each checkpoint step can be found under the outputs folder.