import pandas as pd
import numpy as np

from data import extract_users_items_predictions


def submit(data_dir, predictions):
    submission_pd = pd.read_csv(data_dir+'sampleSubmission.csv')
    submission_users, submission_movies, _ = extract_users_items_predictions(submission_pd)

    ratings = predictions[submission_movies, submission_users]
    submission_pd.iloc[:, 1] = np.clip(ratings, a_min=1, a_max=5)

    file_name = 'submission.csv'
    submission_pd.to_csv(file_name, encoding='utf-8', index=False)