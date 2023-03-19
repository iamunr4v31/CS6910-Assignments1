from argparse import ArgumentParser
from torchy.trainer import Trainer

def main():
    parser = ArgumentParser()

    parser.add_argument("-wp", "--wandb-project", type=str, default="cs6910-assignment1", help="Wandb project to use for logging")
    parser.add_argument("-we", "--wandb-entity", type=str, default="cs6910", help="Wandb entity to use for logging")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", help="Dataset to use for training")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("-l", "--loss", type=str, default="mse", help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam", help="Optimizer to use")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning Rate for Optimizers")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Use as momentum for SGDM, RMSProp, and Nesterov")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.995, help="Beta 2 for Adam and Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Numerical Stability Constant")
    parser.add_argument("-a", "--alpha", type=float, default=1e-4, help="Use as weight decay (regularization coefficient)")
    parser.add_argument("-w_i", "--weight-init", type=str, default="he", help="Weight initialization strategy")
    parser.add_argument("-sz", "--hidden-sizes", type=int, nargs="+", default=[128, 64, 32], help="Hidden layer sizes, Pass a list separated by spaces")
    parser.add_argument("-ac", "--activation", type=str, default="tanh", help="Activation function for hidden layers")
    parser.add_argument("-r", "--regularizer", type=str, default="l1", help="Regularizer to use")
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb for logging")

    args = parser.parse_args()
    hyperparameters = {
    'hidden_activations': args.activation,
    'init_strategy': args.weight_init,
    'loss_fn': args.loss,
    'optimizer': args.optimizer,
    'learning_rate': args.learning_rate,
    'n_epochs': args.epochs,
    'batch_size': args.batch_size,
    'alpha': args.alpha,
    'beta_1': args.beta1,
    'beta_2': args.beta2,
    'regularizer': args.regularizer,
    'hidden_sizes': args.hidden_sizes,
    'epsilon': args.epsilon,
    }
    trainer = Trainer(args.dataset, args.wandb_entity, args.wandb_project, use_wandb=False, do_sweep=False, conf_path="", **hyperparameters)
    trainer.run()

if __name__ == "__main__":
    main()