from typing import List

from pandas import DataFrame
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import graphs

scores = []

def get_rfe_features(x, y, estimator_type="support_vector_machine", num_features=5) -> DataFrame:
    """
    This function performs recursive feature elimination using the estimator passed in
    :param num_features: number of features that we want to isolate
    :param x: features
    :param y: labels
    :param estimator_type: estimator to use with RFE
    :return: DataFrame representing the data from only the columns found by the RFE
    """
    if estimator_type == "support_vector_machine":
        # Initialize Support Vector Machine estimator
        clf = SVC(kernel='linear')

        # apply recursive feature elimination
        rfe = RFE(estimator=clf, n_features_to_select=num_features)
        rfe.fit_transform(x, y)
    elif estimator_type == "logistic_regression":
        # scale the data so that logistic regression can converge
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Initialize Logistic Regression estimator
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')

        # apply recursive feature elimination
        rfe = RFE(estimator=clf, n_features_to_select=num_features)
        rfe.fit_transform(x_scaled, y)

    # print(x.columns[rfe.support_])
    # Create a new DataFrame with only the selected features
    return x[x.columns[rfe.support_]].copy()


def add_scores(lst: List):
    for entry in scores:
        lst.append(entry)

def knn(x, y, k) -> None:
    scores.clear()
    feature_selection_method_list = []
    svc_features = get_rfe_features(x, y, estimator_type="support_vector_machine", num_features=5)
    feature_selection_method_list.append("RFE-Support_Vector_Machine")
    logistic_regression_features = get_rfe_features(x, y, estimator_type="logistic_regression", num_features=5)
    feature_selection_method_list.append("RFE-Logistic_Regression")

    for features, feature_selection in zip([svc_features, logistic_regression_features], feature_selection_method_list):
        with open(f"../data/logs/{feature_selection}_{k}.log", "w") as f:
            f.write(f"Top 5 features: {features.columns.to_numpy()}\n\n")

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        # Initialize and train the kNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)

        # Evaluate the model
        accuracy = knn_classifier.score(x_test, y_test)

        #write accuracy to log
        with open(f"../data/logs/{feature_selection}_{k}.log", "a") as f:
            f.write(f"Accuracy: {round(accuracy, 4)}\n\n")

        #add accuracy to dict
        scores.append((f"{feature_selection}-{k}", round(accuracy, 4)))

        # Assuming knn_classifier is your trained kNN classifier
        y_pred = knn_classifier.predict(x_test)

        graphs.confusion(y_test, y_pred, feature_selection, k)
