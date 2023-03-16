import numpy as np
import pandas as pd
import datetime

from finalProject.hotel_trending import output_score
from finalProject.monthly_reccomendation import monthlyTrendyWrapper
from finalProject.svdpp import trainSVDPP

def normalizeColumn(series: pd.Series):
    return series/np.max(series)

class FinalModel():
    def __init__(self, movies: pd.DataFrame, user_watched:pd.DataFrame, df:pd.DataFrame):
        assert isinstance(movies, pd.DataFrame)
        assert isinstance(user_watched, pd.DataFrame)
        assert isinstance(df, pd.DataFrame)
        assert "MovieID" in movies.columns
        self.moviesDF = movies
        self.user_watchedDF = user_watched
        self.dfDF = df

    def train(self):
        # Get Collaborative Filtering Prediction
        self.svdpPredictor = trainSVDPP(self.user_watchedDF)

        # Get Monthly Trending Prediction
        monthPrediction = monthlyTrendyWrapper(self.dfDF, datetime.datetime.now().month, bookings=250, sd_diff=1.5,
                                               weightEqual=False).rename(
            columns={"ranking": "Monthly_Trending_Prediction"}).set_index("MovieID")

        # Get Hotel Trending Prediction
        hotelTrending = output_score(self.dfDF).rename(columns={"score": "Hotel_Trending_Prediction"}).set_index("MovieID")

        movies = self.user_watchedDF[["MovieID"]].drop_duplicates().set_index("MovieID")
        self.moviesAndAllPredictions = movies.join(monthPrediction, on="MovieID").join(hotelTrending, on="MovieID").fillna(0)
    def predict(self, uid: int, weights: tuple[int,int,int], k:int = 10):
        assert hasattr(self, "moviesAndAllPredictions"), "Did not train before trying to predict"
        assert isinstance(weights, tuple), "Weights are not a tuple"
        assert all((isinstance(weight, int) for weight in weights)), "One or more of the weights is not a int"
        assert all((weight >= 0 for weight in weights)), "One or more of the weights is negative"
        assert sum(weights) == 100, "Sum of weights is not 100"
        assert isinstance(uid, int)

        #Get Collaborative Filtering Prediction
        svdpp_prediction = self.svdpPredictor(uid).set_index("MovieID").rename(columns={"Prediction": "SVDpp_Prediction"})

        total = self.moviesAndAllPredictions.join(svdpp_prediction, on = "MovieID").fillna(0)

        total["Hotel_Trending_Prediction"] = normalizeColumn(total["Hotel_Trending_Prediction"])
        total["Monthly_Trending_Prediction"] = normalizeColumn(total["Monthly_Trending_Prediction"])
        total["SVDpp_Prediction"] = normalizeColumn(total["SVDpp_Prediction"])

        collaborativeFilteringWeight, monthlyTrendingWeight, hotelTrendingWeight = weights

        total["TotalScore"] = total["Hotel_Trending_Prediction"]*hotelTrendingWeight + \
                                                total["Monthly_Trending_Prediction"]*monthlyTrendingWeight + \
                                                total["SVDpp_Prediction"]*collaborativeFilteringWeight
        total.sort_values(by=["TotalScore"], inplace=True, ascending=False)
        return total[["TotalScore"]].iloc[:k]

if __name__ == "__main__":
    df = pd.read_csv("intermediates/df.csv")
    movies = pd.read_csv("../data/movies.csv")
    user_watched = pd.read_csv("intermediates/user_watched.csv")

    final = FinalModel(movies, user_watched, df)
    final.train()

    print(final.predict(2976,(80,10,10)))