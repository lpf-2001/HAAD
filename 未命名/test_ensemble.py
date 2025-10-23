from train_utils import *
from configobj import ConfigObj
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--model1', '-m1', default='varcnn', choices=['cnn', 'lstm', 'sdae', 'ensemble', 'df', 'varcnn','awf'])

parser.add_argument('--model2', '-m2', default='lstm', choices=['cnn', 'lstm', 'sdae', 'ensemble', 'df', 'varcnn','awf'])

parser.add_argument('--model3', '-m3', default='df', choices=['cnn', 'lstm', 'sdae', 'ensemble', 'df', 'varcnn','awf'])
parser.add_argument('--data', '-d', default='rimmer100', choices=['rimmer100', 'rimmer200', 'sirinam95'])

m1 = parser.parse_args().model1
m2 = parser.parse_args().model2
m3 = parser.parse_args().model3
dataset = parser.parse_args().data

dataname, num_classes = split_alpha_number(dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


learn_param = ConfigObj("My_tor.conf")["ensemble"]

train_loader, val_loader, test_loader = load_datasets(dataname, num_classes, learn_param)


evaluate_ensemble_vs_single( m1, m2, m3, dataname, num_classes, test_loader)