from typing import List

from pandas import DataFrame
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import graphs

scores = []

def user_chosen_features(x, choice) -> DataFrame:
    if choice == "saehej":
        pass
    elif choice == "tyler":
        features = ['Total Offense', 'Scoring Offense', 'Total Defense', 'Scoring Defense', 'Turnover Margin']
    elif choice == "offense":
        features = ['Total Offense', 'Rushing Offense', 'Passing Offense', 'Team Passing Efficiency', 'Scoring Offense']
    elif choice == "defense":
        features = ['Total Defense', 'Rushing Defense', 'Passing Yards Allowed', 'Team Passing Efficiency Defense', 'Scoring Defense']

    return x[features].copy()


def get_rfe_features(x, y, estimator_type="support_vector_machine", num_features=5) -> DataFrame:
    """
    This function performs recursive feature elimination using the estimator passed in
    :param num_features: number of features that we want to isolate
    :param x: features
    :param y: labels
    :param estimator_type: estimator to use with RFE
    :return: DataFrame representing the data from only the columns found by the RFE
    """
    rfe = None
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
    elif estimator_type == "decision_tree_classifier":
        # initialize the Decision Tree Classifier
        clf = DecisionTreeClassifier()

        # apply the recursive feature elimination
        rfe = RFE(estimator=clf, n_features_to_select=num_features)
        rfe.fit(x, y)

    # print(x.columns[rfe.support_])
    # Create a new DataFrame with only the selected features
    if rfe is not None:
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
    decision_tree_features = get_rfe_features(x, y, estimator_type="decision_tree_classifier", num_features=5)
    feature_selection_method_list.append("RFE-Decision_Tree_Classifier")

    # user defined features
    tyler_features = user_chosen_features(x, "tyler")
    feature_selection_method_list.append("User_Defined_Tyler")
    offense_features = user_chosen_features(x, "offense")
    feature_selection_method_list.append('User_Defined_Offense')
    defense_features = user_chosen_features(x, "defense")
    feature_selection_method_list.append('User_Defined_Defense')

    for features, feature_selection in zip([svc_features, logistic_regression_features, decision_tree_features, tyler_features, offense_features, defense_features], feature_selection_method_list):
        with open(f"../data/logs/{feature_selection}_{k}.log", "w") as f:
            f.write(f"Top 5 features: {features.columns.to_numpy()}\n\n")

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        # Initialize and train the kNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(x_train, y_train)

        # Evaluate the model
        accuracy = knn_classifier.score(x_test, y_test)

        # write accuracy to log
        with open(f"../data/logs/{feature_selection}_{k}.log", "a") as f:
            f.write(f"Accuracy: {round(accuracy, 4)}\n\n")

        # add accuracy to dict
        scores.append((f"{feature_selection}-{k}", round(accuracy, 4)))

        # Assuming knn_classifier is your trained kNN classifier
        y_pred = knn_classifier.predict(x_test)

        graphs.confusion(y_test, y_pred, feature_selection, k)
