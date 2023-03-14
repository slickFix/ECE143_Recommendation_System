import tkinter as tk
import pandas as pd

from finalProject.data_cleaning import clean_data, get_user_watched
from finalProject.weighted_prediction import FinalModel


print("Importing data")
watchings = pd.read_csv('../data/watchings.csv')
movies = pd.read_csv('../data/movies.csv')
movie_watchings = pd.read_csv('../data/movie_watchings.csv')

print("Cleaning Data, and generating intermediate cleaned data")
df = clean_data(watchings,movies , movie_watchings)
user_watched = get_user_watched(df)

USER_IDS = watchings["UserID"].unique().tolist()
DF_DATA = None

final = FinalModel(movies, user_watched, df)
final.train()


def generate_dataframe(uid, weights):
    print(f"Getting prediction for user: {uid}, with weights: {weights} ")
    DF_DATA = final.predict(uid, weights).reset_index()
    return DF_DATA

class RecommenderUI:
    def __init__(self, master):
        self.master = master
        master.title("Movie Recommender")

        # Initialize the slider values
        self.slider_1_value = 33
        self.slider_2_value = 33
        self.slider_3_value = 34

        # Create the sliders for divvying up the total
        self.slider_frame = tk.Frame(master)
        self.slider_frame.pack(fill=tk.X)

        self.slider_label_1 = tk.Label(self.slider_frame, text="Collaborative Filtering %")
        self.slider_label_1.pack(side=tk.LEFT)
        self.slider_1 = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 variable=tk.DoubleVar(value=self.slider_1_value),
                                 command=self.update_sliders)
        self.slider_1.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.slider_label_2 = tk.Label(self.slider_frame, text="Monthly Trending %")
        self.slider_label_2.pack(side=tk.LEFT)
        self.slider_2 = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 variable=tk.DoubleVar(value=self.slider_2_value),
                                 command=self.update_sliders)
        self.slider_2.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.slider_label_3 = tk.Label(self.slider_frame, text="Hotel Trending %")
        self.slider_label_3.pack(side=tk.LEFT)
        self.slider_3 = tk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                 variable=tk.DoubleVar(value=self.slider_3_value),
                                 command=self.update_sliders)
        self.slider_3.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create the dropdown for selecting a user ID
        self.dropdown_frame = tk.Frame(master)
        self.dropdown_frame.pack(fill=tk.X)

        self.dropdown_label = tk.Label(self.dropdown_frame, text="Select a User ID:")
        self.dropdown_label.pack(side=tk.LEFT)
        self.selected_user_id = tk.StringVar()
        self.selected_user_id.set(USER_IDS[0])

        self.dropdown = tk.OptionMenu(self.dropdown_frame, self.selected_user_id, *USER_IDS)
        self.dropdown.pack(side=tk.LEFT)

        # Create the button for generating the dataframe
        self.button = tk.Button(master, text="Go", command=self.generate_dataframe)
        self.button.pack()

        # Create the output frame for displaying the dataframe
        self.output_frame = tk.Frame(master)
        self.output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_label = tk.Label(self.output_frame, text="Output")
        self.output_label.pack()

        self.output_table = pd.DataFrame(DF_DATA)
        self.output_table_widget = tk.Label(self.output_frame, text=self.output_table.to_string(index=False))
        self.output_table_widget.pack(fill=tk.BOTH, expand=True)

    def update_sliders(self, *_):
        # Get the current slider values
        slider_1_value = self.slider_1.get()
        slider_2_value = self.slider_2.get()
        slider_3_value = self.slider_3.get()

        # Calculate the total and adjust the sliders if necessary
        total = slider_1_value + slider_2_value + slider_3_value
        if total != 100:
            excess = total - 100
            if slider_1_value > excess:
                self.slider_1.set(slider_1_value - excess)
            elif slider_2_value > excess:
                self.slider_2.set(slider_2_value - excess)
            else:
                self.slider_3.set(slider_3_value - excess)

                # Update the slider values
            self.slider_1_value = self.slider_1.get()
            self.slider_2_value = self.slider_2.get()
            self.slider_3_value = self.slider_3.get()

    def generate_dataframe(self):
        user_id = int(self.selected_user_id.get())
        df = generate_dataframe(user_id, (self.slider_1_value, self.slider_2_value, self.slider_3_value))

        self.output_table = df
        self.output_table_widget.config(text=self.output_table.to_string(index=False))

# Start the application

root = tk.Tk()
app = RecommenderUI(root)
root.mainloop()
