import pandas as pd
import numpy as np
from datetime import datetime
import os

def clean_data(watchings_path: str, movies_path: str, movie_watchings_path: str) -> pd.DataFrame:
    """
        Cleans and processes raw data from the client.
        watchings: Watching events in terms of the provider and user
        movies: Metadata regarding movies
        movie_watchins: Watching events in terms of the user and movie

        Returns: A analysis ready datarame with some added convinience columns
    """
    assert isinstance(watchings_path, str), "watchings_path must be a string"
    assert isinstance(movies_path, str), "movies_path must be a string"
    assert isinstance(movie_watchings_path, str), "movie_watchings_path must be a string"
    assert os.path.isfile(watchings_path), f"{watchings_path} is not a valid file path"
    assert os.path.isfile(movies_path), f"{movies_path} is not a valid file path"
    assert os.path.isfile(movie_watchings_path), f"{movie_watchings_path} is not a valid file path"


    watchings = pd.read_csv(watchings_path).filter(['WatchingID', 'UserID', 'WatchDate', 'ProviderID'], axis=1)
    movies = pd.read_csv(movies_path).filter(['MovieID', 'MovieType', 'MovieRank'], axis=1)
    movie_watchings = pd.read_csv(movie_watchings_path).filter(['WatchingID', 'ProviderID', 'MovieID', 'WatchDate'], axis=1)

    #Merge: movie_watchings and watching both have information regarding a single watching event
    df1 = pd.merge(watchings, movie_watchings, on='WatchingID').rename(columns={"ProviderID_x": "ProviderID"}).rename(columns={"WatchDate_x": "WatchDate"}).filter(['WatchingID', 'UserID', 'ProviderID', 'MovieID', 'WatchDate'], axis=1)
    #Add in data about the movie being watched to each watch event
    df2 = pd.merge(df1, movies, on='MovieID')

    # Convert the 'time' column to a datetime object
    df2['WatchDate'] = pd.to_datetime(df2['WatchDate'])

    # Calculate the number of days between the 'time' column and the current date
    df2['Days_Since_Watched'] = (datetime.now() - df2['WatchDate']).apply(lambda x: x.days)

    # Extract the month from the 'time' column
    df2['Month'] = df2['WatchDate'].apply(lambda x: x.month)

    df = df2

    for col in set(df.columns) - {'WatchDate'}:
        df[col] = df[col].apply(__convert_int__)

    df.drop_duplicates(subset=df.columns.difference(['WatchDate']), inplace=True)

    return df


def get_user_watched(df: pd.DataFrame) -> pd.DataFrame:
    user_watched = df.filter(['WatchingID', 'UserID', 'MovieID'], axis=1)

    # drop the ID column, then count how many times a user has watched each movie
    user_watched = user_watched.drop(columns=['WatchingID']).groupby(['UserID', 'MovieID']).size().reset_index(name='Number_Watched')

    for col in user_watched:
        user_watched[col] = user_watched[col].apply(__convert_int__)

    #Add 1, so if one time watch is then one
    user_watched['Number_Watched_log'] = user_watched['Number_Watched'].apply(np.log) + 1

    users_counts = user_watched['UserID'].value_counts().rename('users_counts')
    users_data = user_watched.merge(users_counts.to_frame(),
                                    left_on='UserID',
                                    right_index=True)

    subset_df = users_data[
        (users_data.users_counts >= 3) & (users_data.Number_Watched >= 5)]

    df2 = subset_df['MovieID'].value_counts().rename('df2')
    df2_data = subset_df.merge(df2.to_frame(),
                               left_on='MovieID',
                               right_index=True)

    df2_data = df2_data[df2_data.df2 >= 5]

    user_watched = df2_data.drop(['df2', 'users_counts'], axis=1)

    return user_watched
