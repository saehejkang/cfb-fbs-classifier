import data_cleaner, classify, graphs

accuracy_scores = []

if __name__ == '__main__':
    # run the datascraper to collect data
    # college_data = datascraper.run_datascraper()

    # clean the data
    data = data_cleaner.clean_data()

    # isolate the features into their own Dataframe
    features = data.iloc[:, 3:]
    labels = data['Final_Standing']

    # classify
    classify.knn(x=features, y=labels, k=3)
    classify.add_scores(accuracy_scores)
    classify.knn(x=features, y=labels, k=5)
    classify.add_scores(accuracy_scores)
    classify.knn(x=features, y=labels, k=7)
    classify.add_scores(accuracy_scores)

    labels, values = zip(*accuracy_scores)

    graphs.accuracy_comparison(labels, values)
