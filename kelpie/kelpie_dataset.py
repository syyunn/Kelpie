import copy
import numpy
from kelpie.dataset import Dataset

class KelpieDataset(Dataset):
    """
        Since Datasets handle the correspondence between textual entities and ids,
        the KelpieDataset has the responsibility to decide the id of the kelpie entity
        and to store the train, valid and test samples specific for the original entity and for the kelpie entity

        A KelpieDataset is never *loaded* from file: it is always generated from a pre-existing, already loaded Dataset.
    """

    def __init__(self,
                 dataset: Dataset,
                 entity_id: int):

        super(KelpieDataset, self).__init__(name=dataset.name,
                                            separator=dataset.separator,
                                            load=False)

        if dataset.num_entities == -1:
            raise Exception("The Dataset passed to initialize a MultiKelpieDataset must be already loaded")

        # the KelpieDataset is now basically empty (because load=False was used in the super constructor)
        # so we must manually copy (and sometimes update) all the important attributes from the loaded Dataset
        self.num_entities = dataset.num_entities + 1    # add the Kelpie entity to the count!
        self.num_relations = dataset.num_relations
        self.num_direct_relations = dataset.num_direct_relations

        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.entity_name_2_id = copy.deepcopy(dataset.entity_name_2_id)
        self.entity_id_2_name = copy.deepcopy(dataset.entity_id_2_name)
        self.relation_name_2_id = copy.deepcopy(dataset.relation_name_2_id)
        self.relation_id_2_name = copy.deepcopy(dataset.relation_id_2_name)

        # in our nomenclature the "original entity" is the entity to explain the prediction of;
        # the "kelpie entity "is the fake entity created (and later post-trained) to explain the original entity
        self.original_entity_id = entity_id
        self.original_entity_name = self.entity_id_2_name[self.original_entity_id]
        # add the kelpie entity to the dataset
        self.kelpie_entity_id = dataset.num_entities       # the new entity is always the last one

        self.entity_name_2_id["kelpie"] = self.kelpie_entity_id
        self.entity_id_2_name[self.kelpie_entity_id] = "kelpie"

        # note that we are not copying all the triples and samples from the original dataset,
        # because the KelpieComplExDataset DOES NOT NEED THEM.
        # The train, valid, and test samples of the KelpieComplExDataset
        # are only those of the original dataset that feature the original entity!
        self.original_train_samples = self._extract_samples_with_entity(dataset.train_samples, self.original_entity_id)
        self.original_valid_samples = self._extract_samples_with_entity(dataset.valid_samples, self.original_entity_id)
        self.original_test_samples = self._extract_samples_with_entity(dataset.test_samples, self.original_entity_id)

        # build the train, valid and test samples (both direct and inverse) for the kelpie entity
        # by replacing the original entity id with the kelpie entity id
        self.kelpie_train_samples = self._replace_entity_in_samples(self.original_train_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_valid_samples = self._replace_entity_in_samples(self.original_valid_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_test_samples = self._replace_entity_in_samples(self.original_test_samples, self.original_entity_id, self.kelpie_entity_id)

        # update the to_filter sets to include the filter lists for the facts with the Kelpie entity
        for (entity_id, relation_id) in dataset.to_filter:
            if entity_id == self.original_entity_id:
                self.to_filter[(self.kelpie_entity_id, relation_id)] = copy.deepcopy(self.to_filter[(entity_id, relation_id)])

        # add the kelpie entity in the filter list for all original facts
        for (entity_id, relation_id) in self.to_filter:
            # if the couple (entity_id, relation_id) was in the original dataset,
            # ALWAYS add the kelpie entity to the filtering list
            if (entity_id, relation_id) in dataset.to_filter:
                self.to_filter[(entity_id, relation_id)].append(self.kelpie_entity_id)

            # else, it means that the entity id is the kelpie entity id.
            # in this case add the kelpie entity id to the list only if the original entity id is already in the list
            elif self.original_entity_id in self.to_filter[(entity_id, relation_id)]:
                self.to_filter[(entity_id, relation_id)].append(self.kelpie_entity_id)

    ### private utility methods
    @staticmethod
    def _extract_samples_with_entity(samples, entity_id):
        return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]

    @staticmethod
    def _replace_entity_in_samples(samples, old_entity_id, new_entity_id):
        result = numpy.copy(samples)

        for i in range(len(result)):
            if result[i, 0] == old_entity_id:
                result[i, 0] = new_entity_id
            if result[i, 2] == old_entity_id:
                result[i, 2] = new_entity_id

        return result


class MultiKelpieDataset(Dataset):
    """
        Since Datasets handle the correspondence between textual entities and ids,
        the MultiKelpieDataset has the responsibility to decide the ids of the kelpie entities
        and to store the train, valid and test samples specific for the original entities and for the kelpie entities

        A MultiKelpieDataset is never *loaded* from file.
        It is always generated from a pre-existing, already loaded Dataset.
    """

    def __init__(self,
                 dataset: Dataset,
                 entity_id: int):

        super(MultiKelpieDataset, self).__init__(name=dataset.name,
                                                separator=dataset.separator,
                                                load=False)

        if dataset.num_entities == -1:
            raise Exception("The Dataset passed to initialize a MultiKelpieDataset must be already loaded")

        # the entities to create kelpie versions of are <entity_id> and all its neighbors.
        # in our nomenclature an "original entity" is an entity that needs to be post-trained;
        # the "kelpie entity "is the fake version created and post-trained for that entity
        self.original_entity_ids = [entity_id]
        temp_set = {entity_id}  # this is only useful for efficiency
        for (head, relation, tail) in dataset.train_samples:
            if head == entity_id and tail not in temp_set:
                temp_set.add(tail)
                self.original_entity_ids.append(tail)
            elif tail == entity_id and head not in temp_set:
                temp_set.add(head)
                self.original_entity_ids.append(head)

        # todo: maybe for efficiency, work with the entities with with least degree?

        ### the KelpieDataset is now basically empty (because load=False was used in the super constructor)
        ### so we must manually copy (and sometimes update) all the important attributes from the loaded Dataset

        # define the ids for the kelpie entities
        # these ids start from the last entity id, which is self.num_initial_entities-1
        self.num_initial_entities = dataset.num_entities
        self.num_kelpie_entities = len(self.original_entity_ids)
        self.num_entities = self.num_initial_entities + self.num_kelpie_entities   # include the kelpie entities
        self.kelpie_entity_ids = range(self.num_initial_entities, self.num_initial_entities+self.num_kelpie_entities)

        # useful maps
        self.original_id_2_kelpie_id = {self.original_entity_ids[i]: self.kelpie_entity_ids[i] for i in range(self.num_kelpie_entities)}
        self.kelpie_id_2_original_id = {self.kelpie_entity_ids[i]: self.original_entity_ids[i] for i in range(self.num_kelpie_entities)}

        self.num_relations = dataset.num_relations
        self.num_direct_relations = dataset.num_direct_relations

        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.entity_name_2_id = copy.deepcopy(dataset.entity_name_2_id)
        self.entity_id_2_name = copy.deepcopy(dataset.entity_id_2_name)
        self.relation_name_2_id = copy.deepcopy(dataset.relation_name_2_id)
        self.relation_id_2_name = copy.deepcopy(dataset.relation_id_2_name)


        # definisci gli id delle entità kelpie

        # scrivi self.extract_samples_with_entities
        # ottieni original_samples e kelpie_samples

        # create new names for all kelpie entities
        for original_entity_id in self.original_entity_ids:
            kelpie_entity_id = self.original_id_2_kelpie_id[original_entity_id]
            original_entity_name = self.entity_id_2_name[original_entity_id]
            keplie_entity_name = "K_" + original_entity_name

            self.entity_name_2_id[keplie_entity_name] = kelpie_entity_id
            self.entity_id_2_name[kelpie_entity_id] = keplie_entity_name
        self.original_entity_names = [self.entity_id_2_name[x] for x in self.original_entity_ids]

        # note that we are not copying all the triples and samples from the original dataset,
        # because the KelpieDataset DOES NOT NEED THEM.
        # The train, valid, and test samples of the KelpieDataset
        # are only those of the original dataset that feature an id in self.original_entity_ids!
        self.original_train_samples = self._extract_samples_featuring_original_entity_ids(dataset.train_samples)
        self.original_valid_samples = self._extract_samples_featuring_original_entity_ids(dataset.valid_samples)
        self.original_test_samples = self._extract_samples_featuring_original_entity_ids(dataset.test_samples)

        # build the train, valid and test samples (both direct and inverse) for the kelpie entity
        # by replacing the original entity id with the kelpie entity id
        self.kelpie_train_samples = self._replace_original_entity_ids_in_samples(self.original_train_samples)
        self.kelpie_valid_samples = self._replace_original_entity_ids_in_samples(self.original_valid_samples)
        self.kelpie_test_samples = self._replace_original_entity_ids_in_samples(self.original_test_samples)

        # update the to_filter sets to include the filter lists for the facts with the Kelpie entity
        for (entity_id, relation_id) in dataset.to_filter:
            if entity_id in self.original_entity_ids:
                kelpie_entity_id = self.original_id_2_kelpie_id[entity_id]
                self.to_filter[(kelpie_entity_id, relation_id)] = copy.deepcopy(self.to_filter[(entity_id, relation_id)])

        # add kelpie entities in the filter list whenever the corresponding original entities are present too
        for (entity_id, relation_id) in self.to_filter:
            # if the couple (entity_id, relation_id) was in the original dataset,
            # ALWAYS add ALL kelpie entity ids to the filtering lists
            if (entity_id, relation_id) in dataset.to_filter:
                for any_kelpie_entity_id in self.kelpie_entity_ids:
                    self.to_filter[(entity_id, relation_id)].append(any_kelpie_entity_id)

            # else, it means that the entity id is one of the kelpie entities.
            # in this case:
            #   add all the OTHER kelpie entity ids to the list
            #   add THAT kelpie entity id to the list only if the original entity id is already in the list
            else:
                corresponding_original_entity_id = self.kelpie_id_2_original_id[entity_id]
                for kelpie_entity_id in self.kelpie_entity_ids:
                    if kelpie_entity_id != entity_id or \
                        corresponding_original_entity_id in self.to_filter[(entity_id, relation_id)]:
                        self.to_filter[(entity_id, relation_id)].append(kelpie_entity_id)

    ### private utility methods
    def _extract_samples_featuring_original_entity_ids(self, samples):
        return samples[numpy.where(numpy.logical_or(numpy.isin(samples[:, 0], self.original_entity_ids),
                                                    numpy.isin(samples[:, 2], self.original_entity_ids)))]

    def _replace_original_entity_ids_in_samples(self, samples):
        temp_set = set(self.original_entity_ids)
        output = numpy.copy(samples)

        for i in range(len(output)):
            if output[i, 0] in temp_set:
                print(output[i, 0])
                print(self.original_id_2_kelpie_id[output[i, 0]])

                output[i, 0] = self.original_id_2_kelpie_id[output[i, 0]]
            if output[i, 2] in temp_set:
                output[i, 2] = self.original_id_2_kelpie_id[output[i, 2]]

        return output