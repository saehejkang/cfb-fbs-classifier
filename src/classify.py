import numpy as np
from pandas import DataFrame
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import graphs

def get_rfe_features(x, y, estimator_type = "support_vector_machine", num_features = 5) -> DataFrame:
    """
    This function performs recursive feature elimination using the estimator passed in
    :param estimator_type: estimator to use with RFE
    :return: DataFrame representing the data from only the columns found by the RFE
    """
    if estimator_type == "support_vector_machine":
        #Initialize Support Vector Machine estimator
        clf = SVC(kernel='linear')

        #apply recursive feature elimination
        rfe = RFE(estimator=clf, n_features_to_select=num_features)
        rfe.fit_transform(x, y)
    elif estimator_type == "logistic_regression":
        #scale the data so that logistic regression can converge
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Initialize Logistic Regression estimator
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')

        #apply recursive feature elimination
        rfe = RFE(estimator=clf, n_features_to_select=num_features)
        rfe.fit_transform(x_scaled, y)

    #print(x.columns[rfe.support_])
    # Create a new DataFrame with only the selected features
    return x[x.columns[rfe.support_]].copy()

def knn(x, y, k) -> None:

    svc_features = get_rfe_features(x, y, estimator_type="support_vector_machine", num_features=5)
    logistic_regression_features = get_rfe_features(x, y, estimator_type="logistic_regression", num_features=5)

    for feature_selection in [svc_features, logistic_regression_features]:
        print(f"Performing kNN classification using {feature_selection.columns.to_numpy()}")
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(feature_selection, y, test_size=0.2, random_state=42)

        # Initialize and train the kNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)

        # Evaluate the model
        accuracy = knn_classifier.score(x_test, y_test)
        print(f'Accuracy: {accuracy}')

        # Assuming knn_classifier is your trained kNN classifier
        y_pred = knn_classifier.predict(x_test)

        graphs.confusion(y_test, y_pred)
