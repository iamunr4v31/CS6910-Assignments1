# Neural Network from Scratch

### LINK TO THE REPORT IS IN THE LAST PAGE OF THE REPORT.PDF FILE

## Introduction

This is Assignment 1 of the course Fundamentals of Deep Learning (CS6910) at IIT Madras. The assignment is to construct a neural network from scratch using numpy for classification of the Fashion-MNIST dataset. Use wandb to plot the graphs and for hyperparameter tuning.

## Dataset

The dataset used is the Fashion-MNIST dataset. It is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. The dataset is available at https://www.kaggle.com/zalando-research/fashionmnist.

## Instructions

The code is written in Python 3.10.9. The following libraries are used:
- Numpy
- Matplotlib
- Scikit-learn
- Wandb

It is advised to run the code in a virtual environment. The requirements.txt file contains the list of libraries used. Use the following command to install the libraries.

```

pip install -r requirements.txt

```

The repository implements a module torchy which closely imitates the behaviour of PyTorch. The module is used to implement the neural network. Use train.py to train the model. The hyperparameters can be changed though command line arguments. The accuracy is printed on the console.

|           Name           | Description                                                                   |
| :----------------------: | :-----------------------------------------------------------------------------|
| `-wp`, `--wandb_project` | Project name used to track experiments in Weights & Biases dashboard          |
|  `-we`, `--wandb_entity` | Wandb Entity used to track experiments in the Weights & Biases dashboard.     |
|     `-d`, `--dataset`    | choices:  ["mnist", "fashion_mnist", "cifar10"]                               |
|     `-e`, `--epochs`     | Number of epochs to train neural network.                                     |
|   `-b`, `--batch_size`   | Batch size used to train neural network.                                      |
|      `-l`, `--loss`      | choices:  ["mse", "cross_entropy"]                                            |
|    `-o`, `--optimizer`   | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]              |
| `-lr`, `--learning_rate` | Learning rate used to optimize model parameters                               |
|    `-beta1`, `--beta1`   | Beta1 used by adam and nadam optimizers. Used as momentum for other optimizers|
|    `-beta2`, `--beta2`   | Beta2 used by adam and nadam optimizers.                                      |
|    `-eps`, `--epsilon`   | Epsilon used by optimizers.                                                   |
| `-a`, `--alpha`          | Weight decay used by optimizers.                                              |
|  `-w_i`, `--weight_init` | choices:  ["random", "xavier", "he", "normal"]                                |
|  `-nhl`, `--num_layers`  | Number of hidden layers used in feedforward neural network.                   |
|  `-sz`, `--hidden_size`  | Number of hidden neurons in a feedforward layer.                              |
|   `-ac`, `--activation`  | choices:  ["identity", "sigmoid", "tanh", "ReLU"]                             |
| `--use-wandb`            | Use wandb to track experiments.                                               |

## Results

The results are discussed in a detailed manner in the report. You may find them in the report.pdf file.


## References

- https://www.kaggle.com/zalando-research/fashionmnist
- ChatGPT for verification of code and understanding of the concepts. It also helped in rephrasing certain sentences in the report.
- Lecture Slides
- Previous year assignments for the bias correction problem (overflow error).
- Towards Data Science blog for the awesome explaination of the softmax diff function.
- PyTorch for inspiration and understanding of the concepts.
- Numpy documentation.
- Wandb documentation.

## Acknowledgements

I sincerely thank my course instructor Prof. Mitesh Khapra for the in-depth lectures and the vast amount of concepts that he covered. I also thank the course TAs for the help in understanding the concepts and debugging the code from the tutorials. I understand that I may have been inspired by the code of the previous year's assignments. I have tried to understand the concepts and implement them from scratch. I apologise for not citing the code in the report.
