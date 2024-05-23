import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def load_data(header_file, data_file):
    with open(header_file, 'r') as file:
        header = file.readline().strip().split(',')
        attribute_types = file.readline().strip().split(',')
        class_id = attribute_types.index('class')
        class_name = header[class_id]

    data_set = pd.read_csv(data_file, header=None, names=header)
    x_values = data_set.drop(columns=[class_name])
    y_values = data_set[class_name]

    return x_values, y_values


def main():

    iris_x, iris_y = load_data("datensaetze/iris.header","datensaetze/iris.data")
    iris_x = iris_x[iris_y != "Iris-setosa"]
    iris_y = iris_y[iris_y != "Iris-setosa"]
    iris_y = label_binarize(iris_y, classes=["Iris-versicolor","Iris-virginica"]).ravel()

    plt.figure()
    colors = ['blue', 'green', 'red', 'cyan']
    names = ["sepal_length","sepal_width","petal_length","petal_width"]
    for i in range(len(iris_x.columns)):
        print(i)
        fpr, tpr, _ = roc_curve(iris_y, iris_x.iloc[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Attribut {names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Kurve für jedes Attribut des Iris-Datensatzes (versicolor vs. virginica)')
    plt.legend(loc="lower right")
    plt.show()

    return 0

main()

#Aufgabe b)
# Blaue Kurve: Erreicht 1.0 TPR bei 0.2 FPR, danach steigt FPR kontinuierlich auf 1.0
# Rote Kurve: Erreicht 1.0 TPR bei 0.8 FPR, danach steigt FPR kontinuierlich auf 1.0

# Blauer Kurve => Besserer Klassifikator da Area Under Curve größer als bei Rot

#Score Blau: zunächst ein niedriger Wert und dann hohe Werte z.B. 0.2 dann 0.9 /1.0
#Score Rot: Wie blau nur umgekehrt