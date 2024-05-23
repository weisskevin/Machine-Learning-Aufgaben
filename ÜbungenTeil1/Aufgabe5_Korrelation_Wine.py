import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Aufgabe 5 a)


def load_data(header_file, data_file):
    with open(header_file, 'r') as file:
        header = file.readline().strip().split(',')

    data_set = pd.read_csv(data_file, header=None, names=header)

    return data_set


def visualize_data(correlation_matrix):
    # Visualisierung der Korrelationen
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korrelationsmatrix der Weinattribute')
    plt.show()


def main():
    #Retrieve x,y from datasets
    wine_dataset = load_data('../datensaetze/wine.header', 'datensaetze/wine.data')

    # Korrelationsmatrix berechnen
    corr = wine_dataset.corr()



    visualize_data(corr)


main()

#Aufgabe 5 b)
# Es lässt sich aus der Heatmap ableiten, dass die Attribute Total Phenols, Flavanoids, Hue,OD.. of diluted Wines und Proline
# sich besonders gut für die Vorhersagen der Abstammung der Weine eignet, da diese eine hohe Korrelationsrate zu der Klasse
# Cultivar aufweisen.