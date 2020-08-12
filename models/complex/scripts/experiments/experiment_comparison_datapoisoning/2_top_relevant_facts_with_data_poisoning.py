"""
This module does explanation for ComplEx model
"""
import argparse
import torch
import numpy

from kelpie import kelpie_perturbation, ranking_similarity
from kelpie.data_poisoning import data_poisoning_relevance
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import KelpieEvaluator
from kelpie.kelpie_dataset import KelpieDataset
from kelpie.models.complex.model import ComplEx, KelpieComplEx
from kelpie.models.complex.optimizer import KelpieComplExOptimizer

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument('--dataset',
                    choices=datasets,
                    help="Dataset in {}".format(datasets),
                    required=True)

parser.add_argument('--model_path',
                    help="Path to the model to explain the predictions of",
                    required=True)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adagrad',
                    help="Optimizer in {} to use in post-training".format(optimizers))

parser.add_argument('--batch_size',
                    default=100,
                    type=int,
                    help="Batch size to use in post-training")

parser.add_argument('--max_epochs',
                    default=200,
                    type=int,
                    help="Number of epochs to run in post-training")

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Factorization rank.")

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate")

parser.add_argument('--perspective',
                    choices=['head', 'tail'],
                    default='head',
                    help="Explanation perspective in {}".format(['head', 'tail']))

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

#   E.G.    explain  why  /m/02mjmr (Barack Obama)  is predicted as the head for
#   /m/02mjmr (Barack Obama) 	/people/person/ethnicity	/m/033tf_ (Irish American)
#   or
#   /m/02mjmr (Barack Obama)	/people/person/places_lived./people/place_lived/location	/m/02hrh0_ (Honolulu)
args = parser.parse_args()

facts_to_explain = [
    ('/m/01l_pn', '/film/film/prequel', '/m/01y9jr'),
    ('/m/06x76', '/sports/sports_team/sport', '/m/0jm_'),
    ('/m/02x2gy0', '/award/award_category/nominees./award/award_nomination/nominated_for', '/m/04jplwp'),
    ('/m/053rxgm', '/film/film/executive_produced_by', '/m/0g_rs_'),
    ('/m/033cnk', '/food/food/nutrients./food/nutrition_fact/nutrient', '/m/0f4hc'),
    ('/m/06c97', '/organization/organization_founder/organizations_founded', '/m/05f4p'),
    ('/m/02xry', '/location/location/time_zones', '/m/02hcv8'),
    ('/m/026lgs', '/film/film/language', '/m/02h40lc'),
    ('/m/0hcs3', '/sports/pro_athlete/teams./sports/sports_team_roster/team', '/m/04mjl'),
    ('/m/0mgkg',
     '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language',
     '/m/02h40lc'),
    ('/m/01v90t', '/people/person/gender', '/m/05zppz'),
    ('/m/023kzp', '/people/person/nationality', '/m/09c7w0'),
    ('/m/01znc_', '/organization/organization_member/member_of./organization/organization_membership/organization',
     '/m/07t65'),
    ('/m/0226cw',
     '/government/politician/government_positions_held./government/government_position_held/legislative_sessions',
     '/m/07p__7'),
    ('/m/02dlfh', '/people/person/nationality', '/m/09c7w0'),
    ('/m/01f8hf', '/film/film/genre', '/m/02kdv5l'),
    ('/m/02bfxb', '/people/person/nationality', '/m/0ctw_b'),
    ('/m/09889g', '/people/person/profession', '/m/016z4k'),
    ('/m/07j87', '/food/food/nutrients./food/nutrition_fact/nutrient', '/m/02kcv4x'),
    ('/m/030znt', '/people/person/profession', '/m/02hrh1q'),
    ('/m/04p5cr', '/tv/tv_program/genre', '/m/07s9rl0'),
    ('/m/015pxr', '/people/person/gender', '/m/05zppz'),
    ('/m/07t90', '/education/educational_institution/students_graduates./education/education/major_field_of_study',
     '/m/01mkq'),
    ('/m/0bpm4yw', '/film/film/costume_design_by', '/m/03y1mlp'),
    ('/m/01r9fv', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/02f6ym'),
    ('/m/0g5pvv', '/film/film/language', '/m/02h40lc'),
    ('/m/0265vt', '/award/award_category/disciplines_or_subjects', '/m/02xlf'),
    ('/m/0315rp', '/film/film/genre', '/m/02kdv5l'),
    ('/m/06lht1', '/people/person/profession', '/m/02hrh1q'),
    ('/m/03nfnx', '/film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium',
     '/m/0735l'),
    ('/m/01lyv', '/music/genre/artists', '/m/01n8gr'),
    ('/m/0fthdk', '/people/person/nationality', '/m/09c7w0'),
    ('/m/06151l', '/people/person/profession', '/m/02hrh1q'),
    ('/m/07j87', '/food/food/nutrients./food/nutrition_fact/nutrient', '/m/07hnp'),
    ('/m/025m8l', '/award/award_category/category_of', '/m/0c4ys'),
    ('/m/02r6c_', '/people/person/profession', '/m/01d_h8'),
    ('/m/04jpl', '/location/location/time_zones', '/m/03bdv'),
    ('/m/0r02m', '/location/location/time_zones', '/m/02lcqs'),
    ('/m/0ftps', '/music/group_member/membership./music/group_membership/group', '/m/0b_xm'),
    ('/m/02pv_d', '/people/person/profession', '/m/02jknp')
]

for (head, relation, tail) in facts_to_explain:
    #deterministic!
    seed = 42
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.deterministic = True

    #############  LOAD DATASET

    # get the fact to explain and its perspective entity
    entity_to_explain = head if args.perspective.lower() == "head" else tail

    # load the dataset and its training samples
    original_dataset = Dataset(name=args.dataset, separator="\t", load=True)

    # get the ids of the elements of the fact to explain and the perspective entity
    head_id, relation_id, tail_id = original_dataset.get_id_for_entity_name(head), \
                                    original_dataset.get_id_for_relation_name(relation), \
                                    original_dataset.get_id_for_entity_name(tail)
    original_entity_id = head_id if args.perspective == "head" else tail_id

    # create the fact to explain as a numpy array of its ids
    original_triple = (head_id, relation_id, tail_id)
    original_sample = numpy.array(original_triple)

    # check that the fact to explain is actually a test fact
    assert(original_sample in original_dataset.test_samples)


    #############   INITIALIZE MODELS AND THEIR STRUCTURES
    # instantiate and load the original model from filesystem
    original_model = ComplEx(dataset=original_dataset,
                             dimension=args.dimension,
                             init_random=True,
                             init_size=args.init) # type: ComplEx
    original_model.load_state_dict(torch.load(args.model_path))
    original_model.to('cuda')

    outlines = []

    all_train_samples = original_dataset.train_samples
    train_samples_containing_head = [(h, r, t) for (h, r, t) in all_train_samples if h == head or t == head]

    training_sample_relevance_couples = data_poisoning_relevance.compute_fact_relevance(model=original_model,
                                                                                        dataset=original_dataset,
                                                                                        sample_to_explain=(head_id, relation_id, tail_id),
                                                                                        perspective="head")

    most_influential_skipped_facts_str = []
    for (training_sample, relevance) in training_sample_relevance_couples[:10]:
        most_influential_skipped_facts_str.append(original_dataset.entity_id_2_name[training_sample[0]] + ";" + \
                                                   original_dataset.relation_id_2_name[training_sample[1]] + ";" + \
                                                   original_dataset.entity_id_2_name[training_sample[2]])


    outlines.append(";".join([head, relation, tail]) + "[" + ",".join(most_influential_skipped_facts_str) + "]\n")

    with open("output_2.txt", "a") as outfile:
        outfile.writelines(outlines)
