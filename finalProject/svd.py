from typing import Callable, Any

import pandas as pd
import numpy as np
from surprise import KNNWithMeans, SVD, SVDpp, KNNBaseline, KNNBasic, KNNWithZScore, BaselineOnly, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from collections import defaultdict

def trainSVDPP(User_Watched:pd.DataFrame) -> Callable[[Any], pd.DataFrame]:
    """
        Trains a svd++ model, given the user_watched dataframe.
        Returns a prediction function that takes in a user function and yields prediction outputs.
    """
    assert isinstance(User_Watched, pd.DataFrame), "Inputted data is not a pd.DataFrame"
    COLS = ['UserID', 'MovieID', 'Number_Watched_log']
    assert all((col in set(User_Watched.columns) for col in COLS)), f"Necesary columns not found. Necessary columns{COLS}"
    
    reader = Reader(rating_scale=(0, 10))
    surprise_data = Dataset.load_from_df(User_Watched[['UserID', 'MovieID', 'Number_Watched_log']], reader)
    trainset, testset = train_test_split(surprise_data, test_size=.30, random_state=7)
    svd_param_grid = {'n_epochs': [10, 15, 20, 25, 30], 'lr_all': [0.003, 0.005, 0.007],
                      'reg_all': [0.0025, 0.005, 0.01]}

    svdpp_gs = GridSearchCV(SVDpp, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
    svdpp_gs.fit(surprise_data)

    svd = SVD(n_epochs=20, lr_all=0.005, reg_all=0.005)
    reader = Reader()
    data = User_Watched
    data = data.rename(columns={"Number_Watched_log": "rating"})

    data = Dataset.load_from_df(data[['UserID', 'MovieID', 'rating']], reader)
    # Run 5-fold cross-validation and print the results
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # sample full trainset
    trainset = data.build_full_trainset()

    # Train the algorithm on the trainset
    svd.fit(trainset)

    # Get all the possible movies that were used
    movies = User_Watched['MovieID'].unique()

    def get_svd_predictions(uid):
        # Convert the movies array to a list
        movieList = movies.tolist()

        predictions = [svd.predict(uid=uid, iid=movie, r_ui=None)[3] for movie in movies]

        # Create a new DataFrame with the movie IDs and predictions
        df = pd.DataFrame({'MovieID':  list(movies), 'Prediction': predictions})
        # Sort the DataFrame by the Prediction column in descending order
        df = df.sort_values(by='Prediction', ascending=False)

        return df

    return get_svd_predictions
