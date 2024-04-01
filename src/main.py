import data_cleaner, classify, graphs

accuracy_scores = []
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
    print(type(teams))
    print(teams.to_numpy())

    # classify k = 3
    for val in classify.knn(x=features, y=labels, k=3):
        for index, status in val.items():
            college = teams['College'][index]
            if not status:
                if college in misclassifications.keys():
                    misclassifications[college] += 1
                else:
                    misclassifications[college] = 1
            # print(f"{teams['College'][index]}: {status}")
    classify.add_scores(accuracy_scores)

    # classify k = 5
    for val in classify.knn(x=features, y=labels, k=5):
        for index, status in val.items():
            college = teams['College'][index]
            if not status:
                if college in misclassifications.keys():
                    misclassifications[college] += 1
                else:
                    misclassifications[college] = 1
            # print(f"{teams['College'][index]}: {status}")
    classify.add_scores(accuracy_scores)

    # classify k = 7
    for val in classify.knn(x=features, y=labels, k=7):
        for index, status in val.items():
            college = teams['College'][index]
            if not status:
                if college in misclassifications.keys():
                    misclassifications[college] += 1
                else:
                    misclassifications[college] = 1
            # print(f"{teams['College'][index]}: {status}")
    classify.add_scores(accuracy_scores)

    # sort the misclassification's dict, so it is in descending order
    sorted_misclassifications = sorted(misclassifications.items(), key=lambda x:x[1])
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

    # get information of aggregate accuracy scores
    labels, values = zip(*accuracy_scores)

    # make plot comparing accuracy scores
    graphs.accuracy_comparison(labels, values)
