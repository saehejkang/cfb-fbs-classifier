from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy_comparison(labels, values):
    """
    This graphs a comparison graph of different kNN runs by comparing accuracy scores
    :param labels:
    :param values:
    :return: None
    """
    plt.bar(labels, values)
    plt.xlabel('Labels')
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
    Whereas, the lighter the color, the less correst predictions were made.
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
    #plt.show()
    plt.savefig(f"../data/plots/confusion_matrix_{feature_selection}_{k}.png")
    plt.close()
