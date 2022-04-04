# Multilayer-Perceptron-MNIST-classification-NumPy
Class project： Realizing image classification of MNIST by MLP on the basis of NumPy.

## Preparations

### Requirements 
python = 3.6.13  
NumPy = 1.19.2  

**for visualization:** matplotlib = 3.3.4, PIL = 8.2.0, sklearn = 0.24.2  

### Dataset 
The MNIST can be downloaded from [official webset](http://yann.lecun.com/exdb/mnist/) or from [https://pan.baidu.com/s/1s8HoOEW_cBaVtq8_rQRHyA](https://pan.baidu.com/s/1s8HoOEW_cBaVtq8_rQRHyA), (pwd: rxor). If you use the baidu netdesk url, directly unzip the file at root dir is acceptable. Else, put the four files (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte) in `./mnist/`. 

### Pre-trained weights
I have pretrained models with different sizes, the number of neurons in the hidden layer range from 100 to 900. You can choose the pre-trained weights you need to make inference or train from scratch.  
The pretrained-models can be downloaded from [https://pan.baidu.com/s/1ynDyD2Aa1QuzJqxCTOqNYg](https://pan.baidu.com/s/1ynDyD2Aa1QuzJqxCTOqNYg), (pwd: dkfb). The downloaded weights should be placed in `./save_model/`.

## Get Started

### Train a MLP
```
python main.py train --hidden_nodes 900 --lr 0.01 --lambda_w 0.01 --vis_train True --vis_feature True
```
The visualization of training process and hidden features can be found in `./results/model_h-nodes${hidden_nodes}_lr${lr}_w${lambda_w}` after training.

### Inference
```
python main.py inference --hidden_nodes 900 --lr 0.01 --lambda_w 0.01
```

### Grid Search
```
python main.py search
```
The searching process including all of the hyper-parameter combinations from:
hidden nodes: {100, 200, 300, 400, 500, 600, 700, 800, 900}
learning rate: {0.01, 0.003, 0.001, 0.0003, 0.0001}
regularization weight: {0.1, 0.03, 0.01, 0.003, 0}

The searching process may take a long period, a full result list can be found in 

