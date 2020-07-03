import torch
from torch import optim
import torch.nn.functional as F

# from kelpie.evaluation import Evaluator
from torch.utils.data import DataLoader

from kelpie.models.hake.data import BatchType, TrainDataset, BidirectionalOneShotIterator
from kelpie.models.hake.model import Hake

from kelpie.dataset import Dataset as KelpieDataset


class HakeOptimizer:

    def __init__(self,
                 model: Hake,
                 optimizer_name: str = "Adam",
                 learning_rate: float = 0.0001,
                 no_decay: bool = False,
                 max_steps: int = 100000,
                 adversarial_temperature: float = 1.0,
                 save_path: str = None):

        self.model = model
        self.learning_rate = learning_rate
        self.no_decay = no_decay
        self.max_steps = max_steps
        self.adversarial_temperature = adversarial_temperature
        self.save_path = save_path

        # build all the supported optimizers using the passed params (learning rate and decays if Adam)
        supported_optimizers = {
            'Adagrad': optim.Adagrad(params=self.model.parameters(), lr=learning_rate),
            'Adam': optim.Adam(params=self.model.parameters(), lr=learning_rate),
            'SGD': optim.SGD(params=self.model.parameters(), lr=learning_rate)
        }

        # choose the Torch Optimizer object to use, based on the passed name
        self.optimizer = supported_optimizers[optimizer_name]

        # create the evaluator to use between epochs
        # self.evaluator = Evaluator(self.model)


    '''
    def _train(self,
              train_samples: np.array,
              max_epochs: int,
              save_path: str = None,
              evaluate_every:int =-1,
              valid_samples:np.array = None):

        # extract the direct and inverse train facts
        training_samples = np.vstack((train_samples,
                                      self.model.dataset.invert_samples(train_samples)))

        # batch size must be the minimum between the passed value and the number of Kelpie training facts
        batch_size = min(self.batch_size, len(training_samples))

        cur_loss = 0
        for e in range(max_epochs):
            cur_loss = self.epoch(batch_size, training_samples)

            if evaluate_every > 0 and valid_samples is not None and \
                    (e + 1) % evaluate_every == 0:
                mrr, h1 = self.evaluator.eval(samples=valid_samples, write_output=False)

                print("\tValidation Hits@1: %f" % h1)
                print("\tValidation Mean Reciprocal Rank': %f" % mrr)

                if save_path is not None:
                    print("\t saving model...")
                    torch.save(self.model.state_dict(), save_path)
                print("\t done.")

        if save_path is not None:
            print("\t saving model...")
            torch.save(self.model.state_dict(), save_path)
            print("\t done.")
    '''

    def get_train_iterator_from_dataset(self,
                                        kelpieDataset: KelpieDataset,
                                        cpu_num: int = 10,
                                        batch_size: int = 1024,
                                        negative_sample_size: int = 128,
                                        ):
        train_dataloader_head = DataLoader(
            TrainDataset(kelpieDataset, negative_sample_size, BatchType.HEAD_BATCH),
            batch_size=batch_size,
            shuffle=True,
            num_workers=max(1, cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(kelpieDataset, negative_sample_size, BatchType.TAIL_BATCH),
            batch_size=batch_size,
            shuffle=True,
            num_workers=max(1, cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        return BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)


    def train(self,
            train_iterator,
            init_step: int = 0):

        warm_up_steps = self.max_steps // 2
        current_learning_rate = self.learning_rate

        # Training Loop
        for step in range(init_step, self.max_steps):

            self.train_step(self.model, train_iterator)

            if step >= warm_up_steps:
                if not self.no_decay:
                    current_learning_rate = current_learning_rate / 10
                self.optimizer(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

        if self.save_path is not None:
            print("\t saving model...")
            torch.save(self.model.state_dict(), self.save_path)
            print("\t done.")


    def train_step(self, model: torch.nn.modules.module, train_iterator):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        self.optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
                        # gamma - dr(h, t)
        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

                        # p(h,r,t)                      # alpha
        negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                            # E log sigma (dr(h, t) - gamma)
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        self.optimizer.step()