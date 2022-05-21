import numpy as np 
import pandas as pd 
import math

import os
import sys

from surprise import SVD, NMF, KNNBaseline, SVDpp, KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV

### to run script, must pass argument to indicate which algorithm to use
### possibilities are: svd, nmf, knn


def print_gscv_values(gsCV):
  print('*** Best parameters: ***')
  print(gsCV.best_params)
  print('***')
  print('*** Best score: ***')
  print(gsCV.best_score)
  print('***')

def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

def do_preds(model, sub_data):
  res_ls = []
  for u,m in sub_data:
    pred = model.predict(uid=u,iid=m)
    res_ls.append(pred)
  return res_ls

def run_svd(data, sub_data, sub_users, sub_movies):
  #init svd
  print('running svd')
  algo = SVD
  param_dict = {
    'n_factors':[50, 100, 200],
    'biased': [True, False],
    'lr_all':[0.005, 0.05],
    'reg_all':[0.02, 0.1],
    'random_state':[42]
    }

  gsCV = GridSearchCV(algo_class=algo, param_grid=param_dict, refit=True, n_jobs=10, joblib_verbose=1)

  gsCV.fit(data)

  print(' #### Printing best values of SVD run #### ')
  print_gscv_values(gsCV=gsCV)

  sub_preds = do_preds(gsCV.best_estimator, sub_data)
  svd_df = pd.DataFrame()
  svd_df['users'] = sub_users
  svd_df['movies'] = sub_movies
  svd_df['preds'] = sub_preds
  svd_df.to_csv('./results/SVDpreds.csv', index=False)
  print('Saved predictions for SVD')


def run_nmf(data, sub_data, sub_users, sub_movies):
  #init nmf
  print('running nmf')
  algo = NMF
  param_dict = {
    'n_factors':[15, 20],
    'biased': [True, False],
    'reg_pu': [0.06, 0.1],
    'reg_qi': [0.06, 0.1],
    'random_state':[42]
    }

  gsCV = GridSearchCV(algo_class=algo, param_grid=param_dict, refit=True, n_jobs=10, joblib_verbose=1)

  gsCV.fit(data)

  print(' #### Printing best values of NMF run #### ')
  print_gscv_values(gsCV=gsCV)

  sub_preds = do_preds(gsCV.best_estimator, sub_data)
  nmf_df = pd.DataFrame()
  nmf_df['users'] = sub_users
  nmf_df['movies'] = sub_movies
  nmf_df['preds'] = sub_preds
  nmf_df.to_csv('./results/NMFpreds.csv', index=False)
  print('Saved predictions for NMF')


def run_knn(data, sub_data, sub_users, sub_movies):
  #init knn
  print('running knn')
  algo = KNNBaseline
  param_dict = {
    'k':[10, 25, 40],
    'sim_options': {'name': ['pearson_baseline'], 'shrinkage': [0,1]},
    'bsl_options': {'name': ['als', 'sgd']}
    }

  gsCV = GridSearchCV(algo_class=algo, param_grid=param_dict, refit=True, n_jobs=10, joblib_verbose=1)

  gsCV.fit(data)

  print(' #### Printing best values of KNNBaseline run #### ')
  print_gscv_values(gsCV=gsCV)

  sub_preds = do_preds(gsCV.best_estimator['rmse'], sub_data)
  knn_df = pd.DataFrame()
  knn_df['users'] = sub_users
  knn_df['movies'] = sub_movies
  knn_df['preds'] = sub_preds
  knn_df.to_csv('./results/KNNBaselinepreds.csv', index=False)
  print('Saved predictions for KNNBaseline')

def run_svd_single(data, sub_data, sub_users, sub_movies):
  print('running svd single model')
  trainset = data.build_full_trainset()
  model = SVD(n_factors=200, biased=True, lr_all=0.005, reg_all=0.1, random_state=42)
  model.fit(trainset)

  sub_preds = do_preds(model, sub_data)
  svd_df = pd.DataFrame()
  svd_df['users'] = sub_users
  svd_df['movies'] = sub_movies
  svd_df['preds'] = sub_preds
  svd_df.to_csv('./results/SVDpreds.csv', index=False)
  print('Saved predictions for SVD')

def run_nmf_single(data, sub_data, sub_users, sub_movies):
  print('running nmf single model')
  trainset = data.build_full_trainset()
  model = NMF(n_factors=20, biased=False, reg_pu=0.1, reg_qi=0.1, random_state=42)
  model.fit(trainset)

  sub_preds = do_preds(model, sub_data)
  nmf_df = pd.DataFrame()
  nmf_df['users'] = sub_users
  nmf_df['movies'] = sub_movies
  nmf_df['preds'] = sub_preds
  nmf_df.to_csv('./results/NMFpreds.csv', index=False)
  print('Saved predictions for NMF')


if __name__ == "__main__":

  ### get data
  #number_of_users, number_of_movies = (10000, 1000)
  data_pd = pd.read_csv('./data/data_train.csv')
  sub_pd = pd.read_csv('./data/sampleSubmission.csv')

  train_users, train_movies, train_predictions = extract_users_items_predictions(data_pd) #use whole data bc doing gridsearchcv

  train_df = pd.DataFrame()
  train_df['users'] = train_users
  train_df['movies'] = train_movies
  train_df['ratings'] = train_predictions

  data = Dataset.load_from_df(train_df, Reader(rating_scale=(1,5)))

  sub_users, sub_movies, sub_preds_wrong = extract_users_items_predictions(sub_pd)

  sub_data = zip(sub_users, sub_movies)

  ### run selected model
  cmdline = sys.argv[1] #arg 0 is name of file

  if (cmdline == 'svd'):
    run_svd_single(data, sub_data, sub_users, sub_movies)
  elif (cmdline == 'nmf'):
    run_nmf_single(data, sub_data, sub_users, sub_movies)
  else:
    run_knn(data, sub_data, sub_users, sub_movies)