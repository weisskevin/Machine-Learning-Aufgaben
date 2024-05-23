import pandas as pd
import numpy as np


def load_data(header_file, data_file):
    with open(header_file, 'r') as file:
        header = file.readline().strip().split(',')
    data_set = pd.read_csv(data_file, header=None, names=header)
    return data_set


def find_outliers(outlier_dataset):
    species_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    outlier_dataset['species'] = outlier_dataset['species'].map(species_mapping)

    z_scores = np.abs((outlier_dataset - outlier_dataset.mean()) /outlier_dataset.std())
    threshhold = 3
    outliers = np.where(z_scores > threshhold)

    species_mapping = {0:"Iris-setosa", 1:"Iris-versicolor", 2:"Iris-virginica"}
    outlier_dataset['species'] = outlier_dataset['species'].map(species_mapping)

    for i in range(len(outliers)):
        print(outlier_dataset.iloc[outliers[i]].values)


def main():
    iris_outlier_dataset = load_data("datensaetze/irisoutliers.header", "datensaetze/irisoutliers.data")
    print("Iris Outliers:")
    find_outliers(iris_outlier_dataset)

main()
