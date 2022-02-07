import optuna
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn
from graph_bert_training import optuna_entry_point
from helpers.ParamRepeatPruner import ParamRepeatPruner

import datetime
import logging
import os
import sys

from graph_bert.DatasetLoader import DatasetLoader
from helpers.create_dataset import DatasetCreator
from helpers.logging_helpers import Writer, ExitOnExceptionHandler
from helpers.parse_args import load_args
from helpers.preprocess import Preprocessor
from helpers.pretrain import GraphBertNodeAttributeReconstruction


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def compute_score(results, vector_size):
    (keys, values) = zip(*results.items())
    kv_g = KeyedVectors(vector_size=vector_size)
    kv_g.add_vectors(keys, np.array(values))

    apk_rel = 0

    for word in tqdm(results):
        gbert_words = [i[0] for i in kv_g.most_similar(word)][:1]
        parents = [i.name() for i in wn.synset(word).hypernyms()]
        relatives = [i.name() for parent in parents for i in wn.synset(parent).hypernyms()] + \
                    [i.name() for parent in parents for i in wn.synset(parent).hyponyms()] + parents
        apk_rel += apk(relatives, gbert_words)
    return apk_rel / len(results)


def objective(trial):
    options = load_args()
    hidden_size = intermediate_size = options.hidden_size

    params = {}
    params['k'] = trial.suggest_categorical('k', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    params['nfeature'] = options.feature_size
    params['x_size'] = options.feature_size
    params['graph_size'] = options.graph_size
    params['lr'] = options.lr
    params['max_epoch'] = options.max_epoch
    params['residual_type'] = options.residual_type
    params['num_attention_heads'] = options.num_attention_heads
    params['num_hidden_layers'] = options.num_hidden_layers
    params['hidden_size'] = hidden_size
    params['intermediate_size'] = intermediate_size

    # params['nfeature'] = trial.suggest_
    # params['x_size'] = trial.suggest_
    # params['graph_size'] = trial.suggest_
    # params['lr'] = trial.suggest_
    # params['max_epoch'] = trial.suggest_
    # params['residual_type'] = trial.suggest_
    # params['num_attention_heads'] = trial.suggest_
    # params['num_hidden_layers'] = trial.suggest_
    # params['hidden_size'] = trial.suggest_
    # params['intermediate_size'] = trial.suggest_

    # Check if parameter setting is repeated
    repeated = prune_params.check_params()
    if repeated > -1:
        raise optuna.exceptions.TrialPruned()

    if options.dataset_name is not None:
        dataset_name = options.dataset_name
    elif options.dir_path is not None:
        dataset_name = f"{os.path.split(options.dir_path)[-1]}_{options.type}"
    else:
        logging.critical("dataset name should be provided in your configuration")
        dataset_name = None

    idx_features_labels = np.genfromtxt("{}/node".format('../data/' + dataset_name + '/'), dtype=np.dtype(str))
    id2label = {int(i): j for i, j in zip(idx_features_labels[:, 0], idx_features_labels[:, -1])}

    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = f"{options.dir_path}/{dataset_name}/"
    data_obj.dataset_name = dataset_name
    data_obj.k = params['k']
    data_obj.compute_s = True

    if options.use_cuda:
        data_obj.device = "cuda"

    embs1, embs2 = optuna_entry_point(params, data_obj, id2label, dataset_name, options.pretrained_path)

    score1 = compute_score(embs1, params['x_size'])
    score2 = compute_score(embs2, params['x_size'])

    score = (score1 + score1) / 2

    trial.report(score)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return score


if __name__ == '__main__':
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42), direction="maximize")
    prune_params = ParamRepeatPruner(study)
    study.optimize(objective, n_trials=100)

    print(study.best_params)
    print(compute_score(optuna_entry_point(**study.best_params)))
