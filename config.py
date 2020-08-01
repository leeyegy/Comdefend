import argparse

parser = argparse.ArgumentParser()
# attack
parser.add_argument("--attack_method", default="PGD", choices=["PGD", "FGSM", "Momentum", "STA","NONE","DeepFool","CW","BIM"])
parser.add_argument("--epsilon", type=float, default=8 / 255)

# dataset
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')

# net
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--num_classes', default=10, type=int)


parser.add_argument('--set_size', default=1000, type=int)

args = parser.parse_args()
