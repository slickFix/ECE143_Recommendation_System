from data_cleaning import clean_data, get_user_watched

df = clean_data('../data/watchings.csv', '../data/movies.csv', '../data/movie_watchings.csvwatchings.csv')
user_watched = get_user_watched(df)

df.to_csv("intermediates/df.csv", encoding = "utf-8-sig")
user_watched.to_csv("intermediates/user_watched.csv", encoding = "utf-8-sig")

