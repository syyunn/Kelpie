import argparse
import os
import torch

from config import MODEL_PATH
from dataset import ALL_DATASET_NAMES, Dataset
from evaluation import Evaluator
from models.distmult.model import DistMult

ALL_MODEL_NAMES = ["DistMult"]

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument('--dataset',
                    choices=ALL_DATASET_NAMES,
                    help="Dataset in {}".format(ALL_DATASET_NAMES)
)

parser.add_argument('--model',
                    choices=ALL_MODEL_NAMES,
                    help="Model in {}".format(ALL_MODEL_NAMES)
)

regularizers = ['N3', 'N2']
parser.add_argument('--regularizer',
                    choices=regularizers,
                    default='N3',
                    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adagrad',
                    help="Optimizer in {}".format(optimizers)
)

parser.add_argument('--max_epochs',
                    default=50,
                    type=int,
                    help="Number of epochs."
)

parser.add_argument('--valid',
                    default=-1,
                    type=float,
                    help="Number of epochs before valid."
)

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Embedding dimension"
)

parser.add_argument('--batch_size',
                    default=1000,
                    type=int,
                    help="Number of samples in each mini-batch in SGD, Adagrad and Adam optimization"
)

parser.add_argument('--reg',
                    default=0,
                    type=float,
                    help="Regularization weight"
)

parser.add_argument('--init',
                    default=1e-3,
                    type=float,
                    help="Initial scale"
)

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate"
)

parser.add_argument('--decay1',
                    default=0.9,
                    type=float,
                    help="Decay rate for the first moment estimate in Adam"
)
parser.add_argument('--decay2',
                    default=0.999,
                    type=float,
                    help="Decay rate for second moment estimate in Adam"
)

parser.add_argument('--load',
                    help="path to the model to load")

args = parser.parse_args()

if args.load is not None:
    model_path = args.load
else:
    model_path = os.path.join(MODEL_PATH, "_".join(["DistMult", args.dataset]) + ".pt")
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = DistMult(dataset=dataset, dimension=args.dimension, init_random=True, init_size=args.init) # type: DistMult
model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()

print("\nEvaluating model...")
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=True)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)