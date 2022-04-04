# Multilayer-Perceptron-MNIST-classification-NumPy
Class project： Realizing image classification of MNIST by MLP on the basis of NumPy.

## Preparations

### Requirements 
python = 3.6.13  
NumPy = 1.19.2  

**for visualization:** matplotlib = 3.3.4, PIL = 8.2.0, sklearn = 0.24.2  

### Dataset 
The MNIST can be downloaded from [official webset](http://yann.lecun.com/exdb/mnist/) or from [https://pan.baidu.com/s/1s8HoOEW_cBaVtq8_rQRHyA](https://pan.baidu.com/s/1s8HoOEW_cBaVtq8_rQRHyA), (pwd: rxor). After downloading the dataset, put the four files (train images and labels, t10k images and labels) in `./mnist/`. (If you use the baidu netdesk url, directly unzip the file at root dir is also acceptable.)

### Pre-trained weights
I have pretrained models with different sizes, the number of neurons in the hidden layer range from 100 to 900. You can choose the pre-trained weights you need to make inference or train from scratch.  
The pretrained-models can be downloaded from [https://pan.baidu.com/s/1ynDyD2Aa1QuzJqxCTOqNYg](https://pan.baidu.com/s/1ynDyD2Aa1QuzJqxCTOqNYg), (pwd: dkfb). The downloaded weights should be placed in `./save_model/`.

## Get Started

### Train a MLP
```
python train --hidden_nodes 900 --lr 0.01 --lambda_w 0.01 --vis_train True --vis_feature True
```
The visualization of training process and hidden features can be found in `./results/` after training.

