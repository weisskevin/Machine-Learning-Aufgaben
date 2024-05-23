import sklearn as sk
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

def visualize_confusion_matrices(list_of_cm, classes,traces, best_k):
    #Prepate subplots for 21
    f, axes = plt.subplots(3, 7, figsize=(25,10), sharey='row')
    cmIndex = 0
    bestString = ""
    for rowNumber in range(3):
        for columnNumber in range(7):
            disp = ConfusionMatrixDisplay(list_of_cm[cmIndex], display_labels=classes)
            cmIndex += 1
            if traces[cmIndex] == best_k:
                bestString = "BEST"
            else:
                bestString = ""
            disp.plot(ax=axes[rowNumber, columnNumber], xticks_rotation=45,cmap='Blues')
            disp.ax_.set_title('K:'+str(cmIndex)+' T:'+str(traces[cmIndex])+" "+bestString)
            disp.im_.colorbar.remove()
            disp.ax_.set_xlabel('')
            if columnNumber != 0:
                disp.ax_.set_ylabel('')

    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    f.colorbar(disp.im_, ax=axes)
    plt.show()


def use_model(x, y):
#   split data in training and testing data
    x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_training_scaled = scaler.fit_transform(x_training)
    x_testing_scaled = scaler.fit_transform(x_testing)

    best_k = 0
    best_accuracy_score = 0

    list_of_cm = []
    traces = {}
    for i in range(1, 22):
        #Create model
        k_nearest = KNeighborsClassifier(n_neighbors=i)
        k_nearest.fit(x_training_scaled, y_training)

        #Use model to predict
        y_prediction = k_nearest.predict(x_testing_scaled)

        #Create confusion matrix
        cm = confusion_matrix(y_testing, y_prediction)
        list_of_cm.append(cm)

        #Calculate
        current_k = np.trace(cm)
        traces[i] = current_k
        current_accuracy_score = accuracy_score(y_testing, y_prediction)

        if current_k > best_k:
            best_k = current_k
            best_accuracy_score = current_accuracy_score
    print("Best T for current dataset: "+str(best_k)+"\nWith an accuracy of "+str(int(best_accuracy_score*100))+"% \n\n")
    visualize_confusion_matrices(list_of_cm, np.unique(y_testing), traces, best_k)



def main():
    #Retrieve x,y from datasets
    iris_x, iris_y = load_data('../datensaetze/iris.header', 'datensaetze/iris.data')
    abalone_x, abalone_y = load_data('../datensaetze/abalone.header', 'datensaetze/abalone.data')

    #use data in model
    use_model(iris_x, iris_y)
    use_model(abalone_x, abalone_y)

main()