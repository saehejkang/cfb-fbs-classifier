import datascraper
import data_cleaner, classify

if __name__ == '__main__':
    # run the datascraper to collect data
    #college_data = datascraper.run_datascraper()

    #clean the data
    data = data_cleaner.clean_data()

    # isolate the features into their own Dataframe
    x = data.iloc[:, 3:]
    y = data['Final_Standing']

    #classify
    classify.knn(x, y)

