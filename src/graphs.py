from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def knn_scatter(x_train, x_test, y_train, y_test, feature_selection):
    cmap = ListedColormap(['r', 'y', 'b'])

    # Plotting training and testing data
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap ,label='Training Data', edgecolor='k', s=50)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cmap, label='Testing Data', edgecolor='k', s=50, marker='^')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'{feature_selection}')
    #plt.legend()
    # Create a legend with black points
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='Training Data'),
               plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=10, label='Testing Data')]
    plt.legend(handles=handles)
    plt.savefig(f"../data/plots/scatter_plot_{feature_selection}.png")
    plt.close()

def misclassification_count(m: list):
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

    plt.legend(bars[::2], ['Playoff Team', 'Bowl-eligible', 'Not Bowl-eligible'])
    plt.xlabel('Teams')
    plt.ylabel('Number of Misclassifications')
    plt.title('Number of Misclassifications by Team (From Test Set)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"../data/plots/misclassifications_count.png")
    plt.close()

def accuracy_comparison(labels, values):
    """
    This graphs a comparison graph of different kNN runs by comparing accuracy scores
    :param labels:
    :param values:
    :return: None
    """
    plt.bar(labels, values)
    plt.ylabel('Accuracy Scores')
    plt.title('Accuracy Score Comparison')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"../data/plots/accuracy_comparison.png")
    plt.close()

def confusion(test, pred, feature_selection, k):
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
