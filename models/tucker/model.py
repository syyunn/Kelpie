import copy
from typing import Tuple, Any

import torch
from torch import nn
import numpy as np
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from dataset import Dataset, KelpieDataset
from model import Model


class TuckER(Model, nn.Module):
    """
        The TuckER class provides a Model implementation in PyTorch for the TuckER system.
        This implementation adheres to paper "TuckER: Tensor Factorization for Knowledge Graph Completion",
        and is largely inspired by the official implementation provided by the authors
        Ivana Balažević, Carl Allen, and Timothy M. Hospedales at https://github.com/ibalazevic/TuckER.

        In training or evaluation, our TuckER class requires samples to be passed as 2-dimensional np.arrays.
        Each row corresponds to a sample and contains the integer ids of its head, relation and tail.
        Only *direct* samples should be passed to the model.
    """


    def __init__(self,
                 dataset: Dataset,
                 entity_dimension: int,
                 relation_dimension:int,
                 input_dropout : float = 0.0,
                 hidden_dropout_1: float = 0.0,
                 hidden_dropout_2: float = 0.0,
                 init_random = True):
        """
            Constructor for TuckER model.

            :param dataset: the Dataset on which to train and evaluate the model
            :param entity_dimension: entity embedding dimension
            :param relation_dimension: relation embedding dimension
            :param input_dropout: input layer dropout rate
            :param hidden_dropout_1: dropout rate after the first hidden layer
            :param hidden_dropout_2: dropout rate after the second hidden layer
        """

        # note: the init_random parameter is important because when initializing a KelpieTuckER,
        #       self.entity_embeddings and self.relation_embeddings must not be initialized as Parameters!

        # initialize this object both as a Model and as a nn.Module
        Model.__init__(self, dataset)
        nn.Module.__init__(self)

        self.dataset = dataset
        self.num_entities = dataset.num_entities     # number of entities in dataset
        self.num_relations = dataset.num_relations   # number of relations in dataset
        self.entity_dimension = entity_dimension     # entity embedding dimension
        self.relation_dimension = relation_dimension # relation embedding dimension

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_dropout_1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_dropout_2)
        self.batch_norm_1 = torch.nn.BatchNorm1d(entity_dimension)
        self.batch_norm_2 = torch.nn.BatchNorm1d(entity_dimension)

        # create the embeddings for entities and relations as Parameters.
        # We do not use the torch.Embeddings module here in order to keep the code uniform to the KelpieTuckER model,
        # (on which torch.Embeddings can not be used as they do not allow the post-training mechanism).
        # We have verified that this does not affect performances in any way.
        if init_random:
            self.entity_embeddings = Parameter(torch.empty(self.num_entities, self.entity_dimension).cuda(), requires_grad=True)
            self.relation_embeddings = Parameter(torch.empty(self.num_relations, self.relation_dimension).cuda(), requires_grad=True)
            self.core_tensor = Parameter(torch.tensor(np.random.uniform(-1, 1, (self.relation_dimension, self.entity_dimension, self.entity_dimension)), dtype=torch.float).cuda(), requires_grad=True)

            # initialize only the entity_embeddings and relation embeddings wih xavier method
            xavier_normal_(self.entity_embeddings)
            xavier_normal_(self.relation_embeddings)

    def score(self, samples: np.array) -> np.array:
        """
            Compute scores for the passed samples
            :param samples: a 2-dimensional numpy array containing the samples to score, one per row
            :return: a monodimensional numpy array containing the score for each passed sample
        """
        # compute scores for each possible tail
        all_scores = self.all_scores(samples)

        # extract from all_scores the specific scores for the initial samples
        samples_scores = []
        for i, (head_index, relation_index, tail_index) in enumerate(samples):
            samples_scores.append(all_scores[i][tail_index])

        return np.array(samples_scores)

    def forward(self, samples: np.array):
        """
            Perform forward propagation on the passed samples.

            In the specific case of TuckER, this method just returns the scores
            that each sample obtains with each possible tail.
            This is because TuckER does not require any external Regularizer,
            so only the scores (for all possible tails) are required in training.
            So, in this specific case, the forward method corresponds to the all_scores method.

            :param samples: a 2-dimensional numpy array containing the samples to run forward propagation on, one per row
            :return: a 2-dimensional numpy array that, for each sample, contains a row with the for each passed sample
        """
        return self.all_scores(samples)

    def all_scores(self, samples: np.array):
        """
            For each of the passed samples, compute scores for all possible entities.
            :param samples: a 2-dimensional numpy array containing the samples to score, one per row
            :return: a 2-dimensional numpy array that, for each sample, contains a row with the for each passed sample
        """

        head_indexes, relation_indexes = samples[:, 0], samples[:, 1]
        head_embeddings = self.entity_embeddings[head_indexes]
        relation_embeddings = self.relation_embeddings[relation_indexes]
        tail_embeddings = self.entity_embeddings  # all possible tails

        # batch normalization and reshape of head embeddings

        head_embeddings = self.batch_norm_1(head_embeddings)
        head_embeddings = self.input_dropout(head_embeddings)
        head_embeddings_reshaped = head_embeddings.view(-1, 1, self.entity_dimension)

        # first multiplication with reshape and dropout
        first_multiplication = torch.mm(relation_embeddings, self.core_tensor.view(self.relation_dimension, -1))
        first_multiplication_reshaped = first_multiplication.view(-1, self.entity_dimension, self.entity_dimension)
        first_multiplication_reshaped = self.hidden_dropout1(first_multiplication_reshaped)

        # second multiplication with reshape, batch norm and dropout
        second_multiplication = torch.bmm(head_embeddings_reshaped, first_multiplication_reshaped)
        second_multiplication_reshaped = second_multiplication.view(-1, self.entity_dimension)
        second_multiplication_reshaped = self.batch_norm_2(second_multiplication_reshaped)
        second_multiplication_reshaped = self.hidden_dropout2(second_multiplication_reshaped)

        # third multiplication with sigmoid activation
        result = torch.mm(second_multiplication_reshaped, tail_embeddings.transpose(1, 0))

        scores = torch.sigmoid(result)

        return scores

    def predict_samples(self, samples: np.array) -> Tuple[Any, Any, Any]:
        """
            This method performs prediction on a collection of samples, and returns the corresponding
            scores, ranks and prediction lists.

            All the passed samples must be DIRECT samples in the original dataset.
            (if the Model supports inverse samples as well,
            it should invert the passed samples while running this method)

            :param samples: the direct samples to predict, in numpy array format
            :return: this method returns three lists:
                        - the list of scores for the passed samples,
                                    OR IF THE MODEL SUPPORTS INVERSE FACTS
                            the list of couples <direct sample score, inverse sample score>,
                            where the i-th score refers to the i-th sample in the input samples.

                        - the list of couples (head rank, tail rank)
                            where the i-th couple refers to the i-th sample in the input samples.

                        - the list of couples (head_predictions, tail_predictions)
                            where the i-th couple refers to the i-th sample in the input samples.
                            The head_predictions and tail_predictions for each sample
                            are numpy arrays containing all the predicted heads and tails respectively for that sample.
        """
        scores, ranks, predictions = [], [], []

        direct_samples = samples

        # assert all samples are direct
        assert (samples[:, 1] < self.dataset.num_direct_relations).all()

        inverse_samples = self.dataset.invert_samples(direct_samples)

        inverse_scores, head_ranks, head_predictions = self.predict_tails(inverse_samples)
        direct_scores, tail_ranks, tail_predictions = self.predict_tails(direct_samples)

        for i in range(direct_samples.shape[0]):
            scores += [(direct_scores[i], inverse_scores[i])]
            ranks += [(head_ranks[i], tail_ranks[i])]
            predictions += [(head_predictions[i], tail_predictions[i])]

        return scores, ranks, predictions

    # TODO: in the future, this may need to be moved to model.py
    def predict_tails(self, samples):
        """
        This method receives in input a batch of samples
        and returns the corresponding tail scores, tail ranks and tail predictions
        :param samples: the direct samples to predict, in numpy array format
        :return: this method returns three lists:
                        - the list of tails scores for the passed samples,
                            where the i-th score refers to the i-th sample in the input samples.

                        - the list of tail ranks
                            where the i-th rank refers to the i-th sample in the input samples.

                        - the list of tail predictions
                            where the i-th prediction refers to the i-th sample in the input samples.
                            The tail_predictions for each sample
                            are numpy arrays containing all the predicted tails respectively for that sample.
        """

        scores, ranks, pred_out = [], [], []

        batch_size = 128
        for i in range(0, samples.shape[0], batch_size):
            batch = samples[i : min(i + batch_size, len(samples))]

            all_scores = self.all_scores(batch)

            tail_indexes = torch.tensor(batch[:, 2]).cuda()  # tails of all passed samples

            # for every triple in the samples
            for sample_number, (head_id, relation_id, tail_id) in enumerate(batch):
                tails_to_filter = self.dataset.to_filter[(head_id, relation_id)]

                # predicted value for the correct tail of that triple
                target_tail_score = all_scores[sample_number, tail_id].item()
                scores.append(target_tail_score)

                # set to 0.0 all the predicted values for all the correct tails for that Head-Rel couple
                all_scores[sample_number, tails_to_filter] = 0.0
                # re-set the predicted value for that tail to the original value
                all_scores[sample_number, tail_id] = target_tail_score

            # this amounts to using ORDINAL policy
            sorted_values, sorted_indexes = torch.sort(all_scores, dim=1, descending=True)
            sorted_indexes = sorted_indexes.cpu().numpy()

            for row in sorted_indexes:
                pred_out.append(row)

            for row in range(batch.shape[0]):
                # rank of the correct target
                rank = np.where(sorted_indexes[row] == tail_indexes[row].item())[0][0]
                ranks.append(rank + 1)

        return scores, ranks, pred_out


################

class KelpieTuckER(TuckER):
    def __init__(
            self,
            dataset: KelpieDataset,
            model: TuckER,
            entity_dimension: int,
            relation_dimension: int,
            input_dropout: float,
            hidden_dropout_1: float,
            hidden_dropout_2: float
            ):
        TuckER.__init__(self,
                        dataset=dataset,
                        entity_dimension=entity_dimension,
                        relation_dimension=relation_dimension,
                        input_dropout=input_dropout,
                        hidden_dropout_1=hidden_dropout_1,
                        hidden_dropout_2=hidden_dropout_2,
                        init_random=False)  # NOTE: this is important! if it is set to True,
                                            # self.entity_embeddings and self.relation_embeddings will be initialized as Parameters
                                            # and it will not be possible to overwrite them with mere Tensors
                                            # such as the one resulting from torch.cat(...) and as frozen_relation_embeddings

        self.model = model
        self.original_entity_id = dataset.original_entity_id
        self.kelpie_entity_id = dataset.kelpie_entity_id

        # extract the values of the trained embeddings for entities, relations and core, and freeze them.
        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        frozen_relation_embeddings = model.relation_embeddings.clone().detach()
        frozen_core = model.core_tensor.clone().detach()

        # It is *extremely* important that kelpie_entity_embedding is both a Parameter and an instance variable
        # because the whole approach of the project is to obtain the parameters model params with parameters() method
        # and to pass them to the Optimizer for optimization.
        #
        # If I used .cuda() outside the Parameter, like
        #       self.kelpie_entity_embedding = Parameter(torch.rand(1, 2*self.dimension), requires_grad=True).cuda()
        # IT WOULD NOT WORK because cuda() returns a Tensor, not a Parameter.

        # Therefore kelpie_entity_embedding would not be a Parameter anymore.

        self.kelpie_entity_embedding = Parameter(torch.rand(1, self.entity_dimension).cuda(), requires_grad=True)

        self.entity_embeddings = torch.cat([frozen_entity_embeddings, self.kelpie_entity_embedding], 0)
        self.relation_embeddings = frozen_relation_embeddings
        self.core_tensor = frozen_core

        # copy the batchnorms of the original TuckER model and keep them frozen
        self.batch_norm_1 = copy.deepcopy(self.model.batch_norm_1)  # copy weights and stuff
        self.batch_norm_2 = copy.deepcopy(self.model.batch_norm_2)  # copy weights and stuff
        self.batch_norm_1.weight.requires_grad = False
        self.batch_norm_1.bias.requires_grad = False
        self.batch_norm_2.weight.requires_grad = False
        self.batch_norm_2.bias.requires_grad = False
        self.batch_norm_1.eval()
        self.batch_norm_2.eval()

    # Override
    def predict_samples(self,
                        samples: np.array,
                        original_mode: bool = False):
        """
        This method overrides the Model predict_samples method
        by adding the possibility to run predictions in original_mode
        which means,
        :param samples: the DIRECT samples. Will be inverted to perform head prediction
        :param original_mode:
        :return:
        """

        direct_samples = samples

        # assert all samples are direct
        assert (samples[:, 1] < self.dataset.num_direct_relations).all()

        # if we are in original_mode, make sure that the kelpie entity is not featured in the samples to predict
        # otherwise, make sure that the original entity is not featured in the samples to predict
        forbidden_entity_id = self.kelpie_entity_id if original_mode else self.original_entity_id
        assert np.isin(forbidden_entity_id, direct_samples[:][0, 2]) == False

        # use the TuckER implementation method to obtain scores, ranks and prediction results.
        # these WILL feature the forbidden entity, so we now need to filter them
        scores, ranks, predictions = super().predict_samples(direct_samples)

        # remove any reference to the forbidden entity id
        # (that may have been included in the predicted entities)
        for i in range(len(direct_samples)):
            head_predictions, tail_predictions = predictions[i]
            head_rank, tail_rank = ranks[i]

            # remove the forbidden entity id from the head predictions (note: it could be absent due to filtering)
            # and if it was before the head target decrease the head rank by 1
            forbidden_indices = np.where(head_predictions == forbidden_entity_id)[0]
            if len(forbidden_indices) > 0:
                index = forbidden_indices[0]
                head_predictions = np.concatenate([head_predictions[:index], head_predictions[index + 1:]], axis=0)
                if index < head_rank:
                    head_rank -= 1

            # remove the kelpie entity id from the tail predictions  (note: it could be absent due to filtering)
            # and if it was before the tail target decrease the head rank by 1
            forbidden_indices = np.where(tail_predictions == forbidden_entity_id)[0]
            if len(forbidden_indices) > 0:
                index = forbidden_indices[0]
                tail_predictions = np.concatenate([tail_predictions[:index], tail_predictions[index + 1:]], axis=0)
                if index < tail_rank:
                    tail_rank -= 1

            predictions[i] = (head_predictions, tail_predictions)
            ranks[i] = (head_rank, tail_rank)

        return scores, ranks, predictions


    # Override
    def predict_sample(self,
                       sample: np.array,
                       original_mode: bool = False):
        """
        Override the
        :param sample: the DIRECT sample. Will be inverted to perform head prediction
        :param original_mode:
        :return:
        """

        assert sample[1] < self.dataset.num_direct_relations

        scores, ranks, predictions = self.predict_samples(np.array([sample]), original_mode)
        return scores[0], ranks[0], predictions[0]

    #Override
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if not isinstance(module, torch.nn.BatchNorm1d):
                module.train(mode)
        return self

    def update_embeddings(self):
        with torch.no_grad():
            self.entity_embeddings[self.kelpie_entity_id] = self.kelpie_entity_embedding