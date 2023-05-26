# k-nearest neighbors on Music Metadata Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import csv
from datetime import datetime


# # Load a CSV file
# def load_csv(filename):
#     dataset = list()
#     with open(filename, 'r', encoding='utf-8') as file:
#         csv_reader = reader(file)
#         header = next(csv_reader)  # Read the header
#         for row in csv_reader:
#             if not row:
#                 continue
#             dataset.append(row)
#     return dataset

# Load a CSV file
def load_csv(filename, num_rows=None):
    dataset = list()
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = reader(file)
        header = next(csv_reader)  # Read the header
        for i, row in enumerate(csv_reader):
            if not row:
                continue
            dataset.append(row)
            if num_rows is not None and i >= num_rows - 1:
                break
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        value = row[column]
        if value is None:
            continue  # Skip None values
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                row[column] = float(value)


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        if isinstance(value, int):
            lookup[value] = value  # Assign the integer value as is
        else:
            lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate the algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold_index, fold in enumerate(folds):
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        print(f"Completed fold {fold_index + 1}/{n_folds}")
    return scores


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(1, len(row1) - 1):
        try:
            value1 = float(row1[i])
            value2 = float(row2[i])
            distance += (value1 - value2) ** 2
        except ValueError:
            print("Error at row1:", row1)
            print("Error at row2:", row2)
            raise
    print("Distance: ", distance)
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions


# Test the kNN on the Music Metadata dataset
seed(1)
filename = '../dataset/cleaned_dataset.csv'
dataset = load_csv(filename, num_rows=100)
for i in range(1, len(dataset[0])):
    str_column_to_float(dataset, i)

# convert class column to integers
str_column_to_int(dataset, -1)

# evaluate algorithm
n_folds = 5
num_neighbors = 5

start_time = datetime.now()
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
mean_accuracy = sum(scores) / float(len(scores))
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

# Save the trained model and evaluation results as CSV
model_header = ['algorithm', 'num_neighbors']
model_data = [k_nearest_neighbors.__name__, num_neighbors]
result_header = ['fold', 'accuracy']
result_data = [[i + 1, score] for i, score in enumerate(scores)]

# Save model to CSV
with open('model.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(model_header)
    writer.writerow(model_data)

# Save result to CSV
with open('result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(result_header)
    writer.writerows(result_data)
