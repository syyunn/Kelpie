import argparse
import os

import numpy
import torch

from config import MODEL_PATH
from dataset import ALL_DATASET_NAMES, Dataset
from evaluation import Evaluator
from models.complex.model import ComplEx
from optimization.multiclass_nll_optimizer import MultiClassNLLptimizer

# todo: when we add more models, we should move these variables to another location
ALL_MODEL_NAMES = ["ComplEx"]

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument('--dataset',
                    choices=ALL_DATASET_NAMES,
                    help="Dataset in {}".format(ALL_DATASET_NAMES)
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
                    help="path to the model to load",
                    required=False)

args = parser.parse_args()

#deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

if args.load is not None:
    model_path = args.load
else:
    model_path = os.path.join(MODEL_PATH, "_".join(["ComplEx", args.dataset]) + ".pt")
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Initializing model...")
model = ComplEx(dataset=dataset, dimension=args.dimension, init_random=True, init_size=args.init)   # type: ComplEx
model.to('cuda')
if args.load is not None:
    model.load_state_dict(torch.load(model_path))

print("Training model...")
optimizer = MultiClassNLLptimizer(model=model,
                                  optimizer_name=args.optimizer,
                                  batch_size=args.batch_size,
                                  learning_rate=args.learning_rate,
                                  decay1=args.decay1,
                                  decay2=args.decay2,
                                  regularizer_name='N3',
                                  regularizer_weight=args.reg)

optimizer.train(train_samples=dataset.train_samples,
                max_epochs=args.max_epochs,
                save_path=model_path,
                evaluate_every=args.valid,
                valid_samples=dataset.valid_samples)

print("Evaluating model...")
mrr, h1 = Evaluator(model=model).eval(samples=dataset.test_samples, write_output=False)
print("\tTest Hits@1: %f" % h1)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
