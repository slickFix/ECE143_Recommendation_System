import numpy as np
import pandas as pd
import datetime

from finalProject.hotel_trending import output_score
from finalProject.monthly_reccomendation import monthlyTrendyWrapper
from finalProject.svdpp import trainSVDPP


def make_prediction(uid: int, weights: tuple[int,int,int], movies: pd.DataFrame, user_watched:pd.DataFrame, df:pd.DataFrame, k = 10):
    assert isinstance(weights, tuple), "Weights are not a tuple"
    assert all((isinstance(weight, int) for weight in weights)), "One or more of the weights is not a int"
    assert all((weight >= 0 for weight in weights)), "One or more of the weights is negative"
    assert sum(weights) == 100, "Sum of weights is not 100"
    assert isinstance(uid, int)
    assert isinstance(movies, pd.DataFrame)
    assert isinstance(user_watched, pd.DataFrame)
    assert isinstance(df, pd.DataFrame)
    assert "MovieID" in movies.columns

    collaborativeFilteringWeight, monthlyTrendingWeight, hotelTrendingWeight = weights

    #Get Collaborative Filtering Prediction
    svdpp_prediction = trainSVDPP(user_watched)(uid).set_index("MovieID").rename(columns = {"Prediction":"SVDpp_Prediction"})
    svdpp_prediction["SVDpp_Prediction"] = (svdpp_prediction["SVDpp_Prediction"]/np.sum(svdpp_prediction["SVDpp_Prediction"]))*collaborativeFilteringWeight

    #Get Monthly Trending Prediction
    monthPrediction = monthlyTrendyWrapper(df, datetime.datetime.now().month, bookings=250, sd_diff=1.5, weightEqual=False).rename(columns = {"ranking":"Monthly_Trending_Prediction"}).set_index("MovieID")
    monthPrediction["Monthly_Trending_Prediction"] = (monthPrediction["Monthly_Trending_Prediction"] / np.sum(
        monthPrediction["Monthly_Trending_Prediction"])) * monthlyTrendingWeight

    #Get Hotel Trending Prediction
    hotelTrending = output_score(df).rename(columns = {"score":"Hotel_Trending_Prediction"}).set_index("MovieID")
    hotelTrending["Hotel_Trending_Prediction"] = (hotelTrending["Hotel_Trending_Prediction"] / np.sum(hotelTrending["Hotel_Trending_Prediction"])) * hotelTrendingWeight

    movies = movies[["MovieID"]].set_index("MovieID")
    moviesAndAllPredictions = movies.join(svdpp_prediction, on="MovieID").join(monthPrediction, on="MovieID").join(hotelTrending, on="MovieID").fillna(0)

    moviesAndAllPredictions["TotalScore"] = moviesAndAllPredictions["Hotel_Trending_Prediction"] + moviesAndAllPredictions["Monthly_Trending_Prediction"] + moviesAndAllPredictions["SVDpp_Prediction"]
    moviesAndAllPredictions.sort_values(by = ["TotalScore"], inplace=True, ascending = False)
    return moviesAndAllPredictions[["TotalScore"]].iloc[:k]


if __name__ == "__main__":
    df = pd.read_csv("intermediates/df.csv")
    movies = pd.read_csv("../data/movies.csv")
    user_watched = pd.read_csv("intermediates/user_watched.csv")
    print(make_prediction(2976,(80,10,10),movies, user_watched, df))