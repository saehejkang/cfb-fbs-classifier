import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def knn_scatter(x_train, x_test, y_train, y_test, feature_selection, knn_clf) -> None:
    """
    This function makes a scatter plot showing the data we are performing kNN on but averaged
    down to just 2 features (using PCA).
    :param x_train: x training data (after PCA)
    :param x_test: x testing data (after PCA)
    :param y_train: y training data (after PCA)
    :param y_test: y testing data (after PCA)
    :param feature_selection: The type of classifier used
    :param knn_clf: Our kNN classifier
    :return: None
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(['r', 'y', 'b'])

    # Plotting training and testing data
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap ,label='Training Data', edgecolor='k', s=50)
    ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cmap, label='Testing Data', edgecolor='k', s=50, marker='^')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title(f'{feature_selection}')

    # Create a legend showing training vs testing data
    handles1 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='Training Data'),
               plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=10, label='Testing Data')]
    legend1 = ax.legend(loc='upper right', handles=handles1)
    ax.add_artist(legend1)

    # Create a legend showing label classifications
    handles2 = [plt.Rectangle((0, 0), 1, 1, color='r', label='Not Bowl-Eligible'),
                plt.Rectangle((0, 0), 1, 1, color='y', label='Bowl-Eligible'),
                plt.Rectangle((0, 0), 1, 1, color='g', label='Playoff Team')]
    legend2 = ax.legend(loc='upper left', handles=handles2)
    ax.add_artist(legend2)

    # # create decision boundries
    # h = 0.02  # step size in the mesh
    # x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    # y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Z = knn_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)

    # Create a second legend for specific elements (upper right corner)
    plt.tight_layout()
    plt.savefig(f"../data/plots/scatter_plot_{feature_selection}.png")
    plt.close()

def misclassification_count(m: list) -> None:
    """
    This function makes a bar graph showing the number of times each sample
    was misclassified amongst all runs of our kNN algorithms.
    :param m: list of tuples where each tuple contains:
    tuple[0]: The team (or sample)
    tuple[1]: The number of times it was misclassified
    :return: None
    """
    labels = [item[0] for item in m]
    values = [item[1] for item in m]
    clfs = [item[2] for item in m]

    bars = plt.bar(labels, values)
    for bar, clf in zip(bars, clfs):
        if clf == 0:
            bar.set_color('red')
        elif clf == 1:
            bar.set_color('yellow')
        elif clf == 2:
            bar.set_color('green')

    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color) for color in ['green', 'yellow', 'red']]
    plt.legend(legend_elements, ['Playoff Team', 'Bowl-eligible', 'Not Bowl-eligible'])

    plt.xlabel('Teams')
    plt.ylabel('Number of Misclassifications')
    plt.title('Number of Misclassifications by Team (From Test Set)')
    plt.xticks(rotation=90)
    plt.gca().yaxis.set_major_locator(MultipleLocator(3))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"../data/plots/misclassifications_count.png")
    plt.close()

def accuracy_comparison(labels, values) -> None:
    """
    This graphs a comparison graph of different kNN runs by comparing accuracy scores
    :param labels: List of labels where each label is a classifier type
    :param values: List of values where each value is an accuracy score for the
    corresponding classifier
    :return: None
    """
    plt.bar(labels, values)
    plt.ylabel('Accuracy Scores')
    plt.title('Accuracy Score Comparison')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"../data/plots/accuracy_comparison.png")
    plt.close()

def confusion(test, pred, feature_selection, k) -> None:
    """
    This function prints a confusion matrix using the actual(test) labels vs the predicted labels.
    The confusion matrix is a heatmap where the darker the blue, the more correct predictions were made.
    Whereas, the lighter the color, the less correct predictions were made.
    :param k: value of k used in the kNN algo
    :param feature_selection: type of feature selection we used
    :param test: test labels (actual)
    :param pred: predicted labels
    :return: None
    """
    # Generate confusion matrix
    matrix = confusion_matrix(test, pred)

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"../data/plots/confusion_matrix_{feature_selection}_{k}.png")
    plt.close()
