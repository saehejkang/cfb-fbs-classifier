import data_cleaner, classify, graphs

ac = {}
misclassifications = {}

if __name__ == '__main__':
    # run the datascraper to collect data
    # college_data = datascraper.run_datascraper()

    misclassified_count_by_team = []

    # clean the data
    data = data_cleaner.clean_data()

    # isolate the features/labels into their own Dataframes
    features = data.iloc[:, 3:]
    labels = data['Final_Standing']

    d = ['College', 'Final_Standing']
    # isolate just the teams
    teams = data[d].copy()

    for k in [3, 5, 7]:
        for results in classify.knn(x=features, y=labels, k=k):
            # get the predictions
            predictions = results[0]

            # get the accuracy info
            accuracy_info = results[1]

            # loop through predictions and track misclassifications
            for index, status in predictions.items():
                college = teams['College'][index]
                if not status:
                    if college in misclassifications.keys():
                        misclassifications[college] += 1
                    else:
                        misclassifications[college] = 1

            # handle accuracy scores
            if k == 3:
                ac[accuracy_info[0]] = {}
            ac[accuracy_info[0]][accuracy_info[1]] = accuracy_info[2]

    # sort the misclassification's dict, so it is in descending order
    sorted_misclassifications = sorted(misclassifications.items(), key=lambda x: x[1])
    sorted_misclassifications.reverse()

    # go through misclassifications data and add true classification
    misclf_final = []
    for entry in sorted_misclassifications:
        school = entry[0]
        count = entry[1]
        for i in range(teams.shape[0]):
            if teams['College'][i] == school:
                # print(f"College {teams['College'][i]} was {teams['Final_Standing'][i]}")
                orig_clf = teams['Final_Standing'][i]
                misclf_final.append((school, count, orig_clf))

    # plot misclassifications count
    graphs.misclassification_count(misclf_final)

    # plot aggregate accuracy scores for all classifiers
    labels = tuple([f"{key}-{inner_key}" for key in ac.keys() for inner_key in ac[key].keys()])
    values = tuple([ac[key][inner_key] for key in ac.keys() for inner_key in ac[key].keys()])
    graphs.accuracy_comparison(labels, values)
