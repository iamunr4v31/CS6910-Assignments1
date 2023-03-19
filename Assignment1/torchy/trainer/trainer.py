import numpy as np
import wandb
import json
import matplotlib.pyplot as plt
from typing import List, TypedDict, Tuple
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, cifar10, fashion_mnist

from torchy.activations import ReLU, Sigmoid, Tanh, Softmax
from torchy.loss import CrossEntropy, MSE
from torchy.optimizers import GradientDescent, Adam, Nadam, RMSProp, SGDM, Nesterov
from torchy.network import NeuralNetwork
from torchy.dataloader import DataLoader

class HyperParameters(TypedDict):
    hidden_activations: str
    init_strategy: str
    loss_fn: str
    optimizer: str
    learning_rate: float
    n_epochs: int
    batch_size: int
    alpha: float
    beta_1: float
    beta_2: float
    regularizer: str
    hidden_sizes: List[int]
    epsilon: float

default_hyperparameters: HyperParameters = {
    'hidden_activations': 'tanh',
    'init_strategy': 'he',
    'loss_fn': 'cross_entropy',
    'optimizer': 'nadam',
    'learning_rate': 1e-4,
    'n_epochs': 10,
    'batch_size': 16,
    'alpha': 1e-4,
    'beta_1': 0.9,
    'beta_2': 0.995,
    'regularizer': 'l1',
    'hidden_sizes': [128, 64, 32],
    'epsilon': 1e-8,
}

class Trainer:
    def __init__(self, dataset: str, wandb_entity: str, wandb_project_id: str="CS6910-Assignment1", use_wandb: bool=True, do_sweep: bool=False, conf_path: str="sweep_config.json", **kwargs: HyperParameters) -> None:
        self.wandb_entity = wandb_entity
        self.use_wandb = use_wandb
        self.wandb_project_id = wandb_project_id
        self.do_sweep = do_sweep
        self.conf_path = conf_path
        self.hyperparameters = {**default_hyperparameters, **kwargs}
        # if self.use_wandb:
        #     wandb.init(project=self.wandb_project_id, entity=wandb_entity, config=self.hyperparameters)
        self.train_loader, self.val_loader, self.test_loader = self.get_data(dataset, self.hyperparameters['batch_size'])
        self.n_features = self.train_loader.x.shape[1]
        self.n_classes = self.train_loader.y.shape[1]
        self.set_model(self.hyperparameters['hidden_sizes'], self.hyperparameters['hidden_activations'])
        self.set_loss_fn(self.hyperparameters['loss_fn'])
        self.set_optimizer(self.hyperparameters['optimizer'])
        
    def set_model(self, hidden_sizes: List[int], hidden_activation: str) -> None:
        self.model = NeuralNetwork(self.n_features, self.n_classes, hidden_sizes, hidden_activation, self.hyperparameters['init_strategy'])
    
    def set_loss_fn(self, loss_fn: str) -> None:
        self.loss_fn = {
            'cross_entropy': CrossEntropy(self.model, self.hyperparameters['regularizer'], self.hyperparameters['alpha']),
            'mse': MSE(self.model, self.hyperparameters['regularizer'], self.hyperparameters['alpha']),
        }.get(loss_fn, CrossEntropy(self.model, self.hyperparameters['regularizer'], self.hyperparameters['alpha']))

    def set_optimizer(self, optimizer: str) -> None:
        self.optimizer = {
            'sgd': GradientDescent(self.model.parameters, self.hyperparameters['learning_rate']),
            'momentum': SGDM(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters["epsilon"]),
            'nag': Nesterov(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters["epsilon"]),
            'rmsprop': RMSProp(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters["epsilon"]),
            'adam': Adam(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters['beta_2'], self.hyperparameters["epsilon"]),
            'nadam': Nadam(self.model.parameters, self.hyperparameters['learning_rate'], self.hyperparameters['beta_1'], self.hyperparameters['beta_2'], self.hyperparameters["epsilon"]),
        }.get(optimizer, GradientDescent(self.model.parameters, self.hyperparameters['learning_rate']))
    
    def get_data(self, dataset: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
            class_labels = [str(i) for i in range(10)]
        elif dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
            class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset == "fashion_mnist":
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
            class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        else:
            raise NotImplementedError("Dataset not implemented")
        
        self.plot_images(x_train, y_train, class_labels, use_wandb=self.use_wandb)
        train_loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(x_val, y_val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(x_test, y_test, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def train(self) -> None:
        for epoch in range(self.hyperparameters["n_epochs"]):
            self.train_epoch()
            self.val_epoch()
            if not self.use_wandb:
                print(f"[{epoch+1}] Train Loss: {self.train_loss:.4f} | Train Acc: {self.train_acc * 100:.4f} | Val Loss: {self.val_loss:.4f} | Val Acc: {self.val_acc * 100:.4f}")
            if self.use_wandb:
                wandb.log({
                    "train_loss": self.train_loss,
                    "val_loss": self.val_loss,
                    "train_acc": self.train_acc,
                    "val_acc": self.val_acc,
                })
    
    def train_epoch(self) -> None:
        train_loss = []
        train_acc = []
        for i, (images, labels) in enumerate(self.train_loader, start=1):
            # self.plot_samples(images, labels)
            preds = self.model(images)
            self.loss_fn(preds, labels)
            self.optimizer.zero_grad()
            self.loss_fn.backward()
            self.optimizer.step(i)
            train_loss.append(self.loss_fn.value)
            train_acc.append(np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)))
        self.train_loss = np.mean(train_loss)
        self.train_acc = np.mean(train_acc)
    
    def val_epoch(self) -> None:
        val_loss = []
        val_acc = []
        for i, (images, labels) in enumerate(self.val_loader):
            preds = self.model(images)
            self.loss_fn(preds, labels)
            val_loss.append(self.loss_fn.value)
            val_acc.append(np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1)))
        self.val_loss = np.mean(val_loss)
        self.val_acc = np.mean(val_acc)
    
    def plot_images(self, data, labels, class_names=None, flatten=False, use_wandb=False):

        uniq_labels = np.unique(labels)

        fig, ax = plt.subplots(2,5, figsize=(15, 6))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        ax = ax.reshape(-1)

        for i, label in enumerate(uniq_labels):
            img = data[np.where(labels == label)[0][0]]
            if class_names:
                ax[i].set_title(class_names[label])
            if flatten:
                img = img.reshape(28, 28)
            ax[i].imshow(img, cmap='gray')
            ax[i].axis('off')
        if use_wandb:
            wandb.log({"Class Images": fig})
        plt.close(fig)

    def sweep(self) -> None:
        config = wandb.config
        self.hyperparameters = {**self.hyperparameters, **config}
        self.set_model(config.hidden_sizes, config.hidden_activations)
        self.set_optimizer(config.optimizer)
        self.set_loss_fn(config.loss_fn)
        wandb.run_name = f"{len(config.hidden_sizes)}Layer-{config.hidden_activations}Activated-{config.optimizer}Optimized-{config.loss_fn}Loss"
        self.train()

    def run(self) -> None:
        if self.do_sweep:
            self.perform_sweep()
        else:
            self.train()

    def predict(self, x) -> np.ndarray:
        return np.argmax(self.model(x), axis=1)
    
    def perform_sweep(self) -> None:
        with open(self.conf_path, 'r') as f:
            sweep_config = json.load(f)
        sweep_id = wandb.sweep(sweep_config, project=self.wandb_project_id, entity=self.wandb_entity)
        wandb.agent(sweep_id, function=self.sweep)
        