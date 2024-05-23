import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN


def load_data_and_process(header_file, data_file):
    with open(header_file, 'r') as file:
        header = file.readline().strip().split(',')
    data_set = pd.read_csv(data_file, header=None)
    columns_to_drop = [col for col in data_set.columns if data_set.columns.get_loc(col) == 0 or data_set.columns.get_loc(col) % 2 == 0]
    data_set = data_set.drop(columns=columns_to_drop)
    data_set.columns = header
    return data_set


def main():
    clust_data = load_data_and_process("datensaetze/clustering_data.header","datensaetze/clustering_data.data")

    plt.scatter(clust_data.iloc[:, 0], clust_data.iloc[:, 1], s=50)
    plt.title('Scatter Plot before clustering')
    plt.show()

    kmeans = KMeans(n_clusters=4, random_state=0)
    labels = kmeans.fit_predict(clust_data)

    plt.scatter(clust_data.iloc[:, 0], clust_data.iloc[:, 1], c=labels, cmap='viridis', s=50, marker='o')
    plt.title('Scatter Plot after clustering')
    plt.show()

    new_row = pd.Series({'V1': 502810613, 'V2': 502810613})
    clust_outlier = pd.DataFrame([new_row], columns=clust_data.columns)
    clust_data_with_outlier = pd.concat([clust_data, clust_outlier], ignore_index=True)

    kmeans_with_five = KMeans(n_clusters=5, random_state=0)
    labels2 = kmeans.fit_predict(clust_data_with_outlier)
    labels3 = kmeans_with_five.fit_predict(clust_data_with_outlier)

    plt.scatter(clust_data_with_outlier.iloc[:, 0], clust_data_with_outlier.iloc[:, 1])
    plt.title('Scatter Plot with Outlier (no clustering)')
    plt.show()

    plt.scatter(clust_data_with_outlier.iloc[:, 0], clust_data_with_outlier.iloc[:, 1], c=labels2, cmap='viridis', s=50, marker='o')
    plt.title('Scatter Plot with Outlier (4 Clusters)')
    plt.show()

    plt.scatter(clust_data_with_outlier.iloc[:, 0], clust_data_with_outlier.iloc[:, 1], c=labels3, cmap='viridis', s=50, marker='o')
    plt.title('Scatter Plot with Outlier (5 Clusters)')
    plt.show()

    clust_ext_data = load_data_and_process("datensaetze/clustering_data_ext.header", "datensaetze/clustering_data_ext.data")
    labels4 = kmeans.fit_predict(clust_ext_data)
    plt.title('Scatter Plot of clust_ext (4 Clusters)')
    print(clust_ext_data)
    for i in range((int)(len(clust_ext_data)/2)-1):
        plt.scatter(clust_ext_data.iloc[:, i], clust_ext_data.iloc[:, i+1], c=labels4, cmap='viridis', s=50, marker='o')
        i = i + 2
    plt.show()

    return 0

main()