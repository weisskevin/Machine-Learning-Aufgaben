import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
import seaborn as sns

def GenerateDTreeFrom_nD_Data(data, labels, metric):
    clf = DecisionTreeClassifier(criterion=metric)
    clf.fit(data, labels)
    return clf

def main():
    iris = load_iris()
    data = iris.data
    labels = iris.target
    treeIrisGini = GenerateDTreeFrom_nD_Data(data=data, labels=labels, metric="gini")
    plot_tree(treeIrisGini, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.show()

    vektor1 = [[5.0, 4.1, 1.8, 0.5]]
    vektor2 = [[5.9, 3.0, 5.1, 1.8]]
    class_vektor1 = treeIrisGini.predict(vektor1)
    class_vektor2 = treeIrisGini.predict(vektor2)
    print("Vektor1 prediciton: ", iris.target_names[class_vektor1[0]])
    print("Vektor2 prediction: ", iris.target_names[class_vektor2[0]])

    pca_iris = PCA(n_components=3)
    pca_components_iris = pca_iris.fit_transform(data)

    plt.figure(figsize=(10,8))
    sns.scatterplot(x=pca_components_iris[:,0],y=pca_components_iris[:,1],hue=[iris.target_names[label] for label in labels])
    sns.scatterplot(x=pca_iris.transform(vektor1)[:, 0], y=pca_iris.transform(vektor1)[:, 1], marker='X', color='red', s=100, label=f'Vektor 1 ({iris.target_names[class_vektor1[0]]})')
    sns.scatterplot(x=pca_iris.transform(vektor2)[:, 0], y=pca_iris.transform(vektor2)[:, 1], marker='X', color='blue', s=100, label=f'Vektor 2 ({iris.target_names[class_vektor2[0]]})')
    plt.title('PCA Iris')
    plt.show()

main()