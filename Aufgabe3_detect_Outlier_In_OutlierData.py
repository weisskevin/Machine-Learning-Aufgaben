import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def load_data(header_file, data_file):

    data_set = pd.read_csv(data_file, names=["x1", "x2", "x3"])

    return data_set

def main():
    outlier_dataset = load_data("datensaetze/3doutlier.header", "datensaetze/3doutlier.data")

    dbscan = DBSCAN(eps=0.2, min_samples=5)
    clusters = dbscan.fit_predict(outlier_dataset)

    outliers = outlier_dataset[dbscan.labels_ == -1]
    print("Ausreißer:")
    print(outliers)

    plt.figure(figsize=(8, 6))
    plt.scatter(outlier_dataset.iloc[:, 0], outlier_dataset.iloc[:, 1], c=clusters, cmap='viridis', s=50, marker='o')
    plt.title('DBSCAN Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label')
    plt.show()
main()