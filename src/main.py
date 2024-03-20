import datascraper
import data_cleaner

if __name__ == '__main__':
    # run the datascraper to collect data
    #college_data = datascraper.run_datascraper()

    #clean the data
    data = data_cleaner.clean_data()

    print(data)
