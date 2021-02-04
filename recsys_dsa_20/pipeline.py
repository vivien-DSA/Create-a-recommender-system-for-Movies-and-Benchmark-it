###############################################LOAD THE libraries ########################################################

import time

import numpy as np
from pathlib import Path

from collections import defaultdict
from surprise.dataset import DatasetAutoFolds
from surprise import Dataset
from surprise import Reader
from surprise.trainset import Trainset
from  surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.prediction_algorithms.knns import KNNBasic
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic


###############################################LOAD THE DATA ########################################################

def load_ratings_from_surprise() -> DatasetAutoFolds:
    ratings = Dataset.load_builtin('ml-100k')
    return ratings


def load_ratings_from_file(ratings_filepath : Path) -> DatasetAutoFolds:
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    ratings = Dataset.load_from_file(ratings_filepath, reader)
    return ratings


def get_data(from_surprise : bool = True) -> DatasetAutoFolds:
    data = load_ratings_from_surprise() if from_surprise else load_ratings_from_file()
    return data
#########################################TRAIN EVALUATE AND PREDICT ########################################################

# def get_trained_model(model_class: AlgoBase, train_set: Trainset, **model_kwargs) -> AlgoBase:
#     model = model_class(sim_options = model_kwargs['sim_options']) if model_kwargs else model_class()
#     model.fit(train_set)
#     return model
fit_time = 0
def get_trained_model(model_class: AlgoBase, train_set: Trainset, *model_kwargs) -> AlgoBase:
    global fit_time 
    model = model_class(sim_options = model_kwargs[0]['sim_options']) if model_kwargs else model_class()
#     print(f'get_trained the model {str(model_class)} with model_kwargs: {model_kwargs[0]}.') if model_kwargs else print(f'get_trained the model {str(model_class)} without model_kwargs.' )
    starts = time.time()   
    model.fit(train_set)
#     print("It has been {0} seconds since the fit started".format(time.time() - starts))
    fit_time = time.time() - starts
    return model


def evaluate_model(model: AlgoBase, test_set: [(int, int, float)]) -> dict:
    global fit_time
    starts = time.time()   
    predictions = model.test(test_set)
#     print("It has been {0} seconds since the evaluation started".format(time.time() - starts))

    metrics_dict = {}
    metrics_dict['RMSE'] = accuracy.rmse(predictions, verbose=False)
    metrics_dict['MAE'] = accuracy.rmse(predictions, verbose=False)
    metrics_dict['test_time'] = time.time() - starts
    metrics_dict['fit_time'] = fit_time
    return metrics_dict


def train_and_evalute_model_pipeline(model_class: AlgoBase, model_kwargs: dict = {},
                                     from_surprise: bool = True,
                                     test_size: float = 0.2) -> (AlgoBase, dict):
    data = get_data(from_surprise)
    train_set, test_set = train_test_split(data, test_size, random_state=42)
    model = get_trained_model(model_class, train_set) if not model_kwargs else get_trained_model(model_class, train_set, model_kwargs)
    metrics_dict = evaluate_model(model, test_set)
    return model, metrics_dict


def make_predictions(model: AlgoBase, test_set: [(int, int, float)]) -> dict:
    predictions = model.test(test_set)
    return predictions


