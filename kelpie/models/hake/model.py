from typing import Tuple, Any

import numpy as np
import torch
from torch import nn

from kelpie.dataset import Dataset
from kelpie.kelpie_dataset import KelpieDataset
from kelpie.model import Model
from kelpie.models.hake.data import BatchType


class Hake(Model, nn.Module):

    def __init__(self,
                 dataset: Dataset,
                 hidden_dim: int,
                 gamma: float,
                 modulus_weight=1.0,
                 phase_weight=0.5):

        # initialize this object both as a Model and as a nn.Module
        Model.__init__(self, dataset)
        nn.Module.__init__(self)

        self.dataset = dataset
        self.num_entities = dataset.num_entities  # number of entities in dataset
        self.num_relations = dataset.num_relations  # number of relations in dataset

        # Hake-specific
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(self.num_entities, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(self.num_relations, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
        )

        nn.init.zeros_(
            tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159262358979323846


    def _func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)


    def score(self, samples: np.array) -> np.array:

        head = self.entity_embedding[samples[:, 0]]  # list of entity embeddings for the heads of the facts
        rel = self.relation_embedding[samples[:, 1]]  # list of relation embeddings for the relations of the heads
        tail = self.entity_embedding[samples[:, 2]]  # list of entity embeddings for the tails of the facts

        return self._func(head, rel, tail, BatchType.SINGLE).cpu().numpy()


    def forward(self, samples: np.array) -> np.array:

        head = self.entity_embedding[samples[:, 0]]  # list of entity embeddings for the heads of the facts
        rel = self.relation_embedding[samples[:, 1]]  # list of relation embeddings for the relations of the heads
        tail = torch.cat([self.entity_embedding[samples[:, 0]], self.entity_embedding[samples[:, 2]]])
        # ^ list of entity embeddings for both the heads and the tails of the facts

        return self._func(head, rel, tail, BatchType.TAIL_BATCH).cpu().numpy()


    def predict_samples(self, samples: np.array) -> Tuple[Any, Any, Any]:

        direct_samples = samples

        # make sure that all the passed samples are "direct" samples
        assert np.all(direct_samples[:, 1] < self.dataset.num_direct_relations)

        # output data structures
        scores, ranks, predictions = [], [], []

        # invert samples to perform head predictions
        inverse_samples = self.dataset.invert_samples(direct_samples)

        # obtain scores, ranks and predictions both for direct and inverse samples
        direct_scores, tail_ranks, tail_predictions = self.predict_tails(direct_samples)
        inverse_scores, head_ranks, head_predictions = self.predict_tails(inverse_samples)

        for i in range(direct_samples.shape[0]):
            # add to the scores list a couple containing the scores of the direct and of the inverse sample
            scores.append((direct_scores[i], inverse_scores[i]))

            # add to the ranks list a couple containing the ranks of the head and of the tail
            ranks.append((int(head_ranks[i]), int(tail_ranks[i])))

            # add to the prediction list a couple containing the lists of predictions
            predictions.append((head_predictions[i], tail_predictions[i]))

        return scores, ranks, predictions


    def predict_tails(self, samples: np.array) -> Tuple[Any, Any, Any]:
        """
            Returns filtered scores, ranks and predicted entities for each passed fact.
            :param samples: a torch.LongTensor of triples (head, relation, tail).
                          The triples can also be "inverse triples" with (tail, inverse_relation_id, head)
            :return:
        """

        ranks = torch.ones(len(samples))  # initialize with ONES

        with torch.no_grad():

            # for each fact <cur_head, cur_rel, cur_tail> to predict, get all (cur_head, cur_rel) couples
            # and compute the scores using any possible entity as a tail
            all_scores = self.score(samples)
            # ^ 2d matrix: each row corresponds to a sample and has the scores for all entities

            # from the obtained scores, extract the the scores of the actual facts <cur_head, cur_rel, cur_tail>
            targets = torch.zeros(size=(len(samples), 1)).cuda()
            for i, (_, _, tail_id) in enumerate(samples):
                targets[i, 0] = all_scores[i, tail_id].item()

            # set to -1e6 the scores obtained using tail entities that must be filtered out (filtered scenario)
            # In this way, those entities will be ignored in rankings
            for i, (head_id, rel_id, tail_id) in enumerate(samples):
                # get the list of tails to filter out; include the actual target tail entity too
                filter_out = self.dataset.to_filter[(head_id, rel_id)]

                if tail_id not in filter_out:
                    filter_out.append(tail_id)

                all_scores[i, torch.LongTensor(filter_out)] = -1e6

            # fill the ranks data structure and convert it to a Python list
            ranks += torch.sum((all_scores >= targets).float(), dim=1).cpu()  # ranks was initialized with ONES
            ranks = ranks.cpu().numpy().tolist()

            all_scores = all_scores.cpu().numpy()
            targets = targets.cpu().numpy()

            # save the list of all obtained scores
            scores = [targets[i, 0] for i in range(len(samples))]

            predictions = []
            for i, (head_id, rel_id, tail_id) in enumerate(samples):
                filter_out = self.dataset.to_filter[(head_id, rel_id)]
                if tail_id not in filter_out:
                    filter_out.append(tail_id)

                predicted_tails = np.where(all_scores[i] > -1e6)[0]

                # get all not filtered tails and corresponding scores for current fact
                # predicted_tails = np.where(all_scores[i] != -1e6)
                predicted_tails_scores = all_scores[i, predicted_tails]  # for cur_tail in predicted_tails]

                # note: the target tail score and the tail id are in the same position in their respective lists!
                # predicted_tails_scores = np.append(predicted_tails_scores, scores[i])
                # predicted_tails = np.append(predicted_tails, [tail_id])

                # sort the scores and predicted tails list in the same way
                permutation = np.argsort(-predicted_tails_scores)

                predicted_tails_scores = predicted_tails_scores[permutation]
                predicted_tails = predicted_tails[permutation]

                # include the score of the target tail in the predictions list
                # after ALL entities with greater or equal scores (MIN policy)
                j = 0
                while j < len(predicted_tails_scores) and predicted_tails_scores[j] >= scores[i]:
                    j += 1

                predicted_tails_scores = np.concatenate((predicted_tails_scores[:j],
                                                         np.array([scores[i]]),
                                                         predicted_tails_scores[j:]))
                predicted_tails = np.concatenate((predicted_tails[:j],
                                                  np.array([tail_id]),
                                                  predicted_tails[j:]))

                # add to the results data structure
                predictions.append(predicted_tails)  # as a np array!

        return scores, ranks, predictions



class KelpieHake(Hake):
    def __init__(
            self,
            dataset: KelpieDataset,
            model: Hake):

        Hake.__init__(self,
                        dataset=dataset,
                        hidden_dim=model.hidden_dim,
                        gamma=model.gamma,
                        modulus_weight=model.modulus_weight,
                        phase_weight=model.phase_weight)

        self.model = model
        self.original_entity_id = dataset.original_entity_id
        self.kelpie_entity_id = dataset.kelpie_entity_id

        # extract the values of the trained embeddings for entities and relations and freeze them.
        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        frozen_relation_embeddings = model.relation_embeddings.clone().detach()

        # Therefore entity_to_explain_embedding would not be a Parameter anymore.
        self.kelpie_entity_embedding = nn.Parameter(torch.zeros(1, self.hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.kelpie_entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.entity_embeddings = torch.cat([frozen_entity_embeddings, self.kelpie_entity_embedding], 0)
        self.relation_embeddings = frozen_relation_embeddings


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

        # use the Model implementation method to obtain scores, ranks and prediction results.
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

    def update_embeddings(self):
        with torch.no_grad():
            self.entity_embeddings[self.kelpie_entity_id] = self.kelpie_entity_embedding