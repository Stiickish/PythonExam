import numpy as np
import pandas as pd
from sklearn import preprocessing

# Read the data from folder and then print out the first few lines
data = pd.read_csv("../dataset/dataset.csv")
print(data.head())

# Create new DataFrame called sort_data. Drop unwanted columns
sort_data = data.drop(['time_signature', 'Unnamed: 0', 'key', 'explicit', 'album_name', 'mode'], axis=1)
sort_data.drop_duplicates(subset=['track_id'], inplace=True)
print(sort_data.head())

# Handle missing values. Drop rows that have missing values
sort_data.dropna(inplace=True)
print("After dropping rows with missing values:")
print(sort_data.head())

# Check if dataset has at least one sample
if sort_data.shape[0] > 0:
    # Normalize the data to a scale of 0-1, with 1 being the highest and 0 being the lowest
    scaler = preprocessing.MinMaxScaler()
    names = sort_data.select_dtypes(include=np.number).columns
    d = scaler.fit_transform(sort_data.select_dtypes(include=np.number))
    data_normalized = pd.DataFrame(d, columns=names)
    data_normalized['artists'] = sort_data['artists']  # Add the 'artists' column back
    data_normalized.set_index(sort_data.loc[:, 'track_id'], inplace=True)

    # Save the cleaned data to a new file
    data_normalized.to_csv("../dataset/cleaned_dataset.csv", index=True)
    print("Cleaned data saved to 'cleaned_dataset.csv'")

    print(data_normalized.head())
else:
    print("No samples remaining after cleaning the dataset")
