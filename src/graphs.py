from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def confusion(test, pred):
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
    plt.show()
