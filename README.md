# SimilarNet
This repository contains TensorFlow Keras codes for SimilarNet

# Requirements
Python >= 3.9.9
TensorFlow >= 2.8.1

# Data
Except miniImageNet and MNIST-CIFAR, It will download required dataset automatically.
For miniImageNet and MNIST-CIFAR, please download the dataset from [this link](https://drive.google.com/file/d/17XYEKAWQIsMej9LjPJlW78XPrYwSEbyW/view?usp=sharing) and unzip the file into the ./data directory.

# Run
```
python SimilarNet_exp_(dataset).py (algorithm) (normalize)
```

Each parameter should have one of the following values. Parameters follow the order.

dataset: MNIST, FMNIST, Omniglot, CIFAR, miniImageNet, MNIST-CIFAR

algorithm: concat, SimilarNet, SimilarNetParametric

normalize: True, False


For example, This script will run the model using SimilarNet with L2 normalize on Omniglot dataset.
```
python SimilarNet_exp_Omniglot.py SimilarNet True
```

This script will run the model using concat without L2 normalize on CIFAR-10 dataset.
```
python SimilarNet_exp_CIFAR.py concat False
```

The result will be stored in ./models/(dataset) directory.

# Success Rate Evaluation
Run Check__result_Ratio-(dataset).ipynb on Jupyter notebook.
It will check the models and make .tag files containing positive/negative ratio and output values of models.
