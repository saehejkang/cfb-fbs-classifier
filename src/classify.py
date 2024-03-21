from sklearn.feature_selection import RFE
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def knn(x, y) -> None:
    # use a Support Vector Machine classifier as an estimator for RFE
    clf = SVC(kernel='linear')

    rfe = RFE(estimator=clf, n_features_to_select=5)
    rfe.fit_transform(x, y)

    print(x.columns[rfe.support_])

    # Create a new DataFrame with only the selected features
    new_df = x[x.columns[rfe.support_]].copy()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(new_df, y, test_size=0.2, random_state=42)

    # Initialize and train the kNN classifier
    k = 3  # Choose the number of neighbors (k)
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)

    # Evaluate the model
    accuracy = knn_classifier.score(x_test, y_test)
    print(f'Accuracy: {accuracy}')