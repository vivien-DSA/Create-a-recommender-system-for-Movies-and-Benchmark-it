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

################################################## RETRIEVE TOP MOVIES AND RECOMMENDATIONS################################################

def get_top_n(predictions, n=10) -> defaultdict:
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def create_top_n_dataframe(top_n, data):
    # Save the top_n in a data frame    
    dataf = pd.DataFrame()
    # To retrieve the true rating we will check on the test result
    test_a = np.array(data)
    for uid, user_ratings in top_n.items():
        for m, r in user_ratings:
            tr = test_a[(test_a[:,0] == uid) & (test_a[:,1] == m)].item(2)
            dataf = pd.concat([dataf, pd.DataFrame([[int(uid), int(m), round(r, 2), tr]],
                                               columns = ['UserId', 'movieId', 'PredictedRating', 'TrueRating'])],
                          ignore_index=True)
    return dataf

def create_recommendation_dataframe(top_n_df, userId):
    movies_recom_all_users = top_n_df.merge(movies, how='left', on=['movieId'])
    output = movies_recom_all_users[movies_recom_all_users['UserId'] == userId].drop(columns = ['UserId', 'movieId'])
    cols = ['title', 'genres', 'PredictedRating', 'TrueRating']
    output = output[cols]
    return output

#####################################################################################################################


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


