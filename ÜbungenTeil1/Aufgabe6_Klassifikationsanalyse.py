import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(header_file, data_file):
    with open(header_file, 'r') as file:
        header = file.readline().strip().split(',')

    data_set = pd.read_csv(data_file, delim_whitespace=True, header=None, names=header)
    data_set = data_set.drop(columns=['index'])

    return data_set


def visualize_data_boxplot(data_set, param):
    plt.figure(figsize=(8, 6))
    boxplot = data_set.boxplot(column=param, by='classattr', grid=False)
    plt.xlabel('Klassentyp')
    plt.ylabel('Wert von '+param)
    plt.title('Boxplot von '+param+' nach Klassentyp')
    plt.suptitle('')
    plt.show()

def visualize_data_corr(correlation_matrix):
    # Visualisierung der Korrelationen
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korrelationsmatrix der Weinattribute')
    plt.show()


def main():
    cl_df = load_data('../datensaetze/cl_problem.header', 'datensaetze/cl_problem.data')
    print(cl_df.dtypes)
    print(cl_df.describe())
    visualize_data_boxplot(cl_df, 'attr1')
    visualize_data_boxplot(cl_df, 'attr2')
    visualize_data_boxplot(cl_df, 'attr3')
    visualize_data_boxplot(cl_df, 'attr4')

    class_mapping = {'a': 1, 'b': 2, 'c': 3}
    cl_df['classattr'] = cl_df['classattr'].map(class_mapping)
    corr = cl_df.corr()
    visualize_data_corr(corr)

main()

#Aufgabe 1
# Durch einen Blick auf die Boxplots der einzelnen Attribute in Verbindung mit den Klassentypen
# Lässt sich schließen, dass alle Klassen durchaus schwierig sich beschreiben lassen, durch die bereitgestellten Daten.

#Aufgabe 2
# Laut Korrelationsmatrix sind die Attribute 1 und 4 dafür geeignet das Klassenattribut vorherzusagen,
# da diese einen hohen Korrelationswert zum Klassenattribut aufweisen.