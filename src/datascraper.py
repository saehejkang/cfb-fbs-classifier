import requests
from bs4 import BeautifulSoup
import csv


def read_from_url(url, college, college_id):
    # set the init values of the dictionary
    finalData = {}
    finalData['College'] = college
    finalData['Id'] = college_id

    # http request for site
    session = requests.Session()
    response = session.get(url, headers={'User-Agent': 'Mozilla/5.0'})

    soup = BeautifulSoup(response.text, "html.parser")
    # find the content area tage
    soup.find(id="contentarea")
    # find the td tag
    soup.find_all("td")
    # find the tag with the rankings target
    rankings = soup.find_all(target="Rankings")
    # iterate through the stats
    for stat in rankings:
        # find the rank needed in the corresponding tag
        rank = stat.findNext("td", align="right")
        # find the value needed in the corresponding tag
        value = rank.findNext("td", align="right")
        # output the stat name
        stat = stat.text
        # output the stat value
        val = value.text.strip()
        finalData[stat] = val
    return finalData


def run_datascraper():
    fields = ["College", "Id", "Total Offense", "Rushing Offense", "Passing Offense",
              "Team Passing Efficiency", "Scoring Offense", "Total Defense", "Rushing Defense",
              "Passing Yards Allowed", "Team Passing Efficiency Defense", "Scoring Defense",
              "Turnover Margin", "3rd Down Conversion Pct", "4th Down Conversion Pct", "3rd Down Conversion Pct Defense",
              "4th Down Conversion Pct Defense", "Red Zone Offense", "Red Zone Defense", "Net Punting", "Punt Returns",
              "Kickoff Returns", "First Downs Offense", "First Downs Defense", "Fewest Penalties Per Game",
              "Fewest Penalty Yards Per Game", "Time of Possession"]
    dict_to_write = []
    fileName = '../data/cfb_teams_to_id.csv'

    with open(fileName, newline='', mode='r') as csvfile:
        college_data = csv.reader(csvfile, delimiter=',', quotechar='|')
        for college in college_data:
            # skip the first row of the csv file
            if college[1] != "Id":
                url = "https://stats.ncaa.org/teams/{id}".format(id=college[1])
                data = read_from_url(url, college[0], college[1])
                dict_to_write.append(data)
        with open(fileName, 'w') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            # write the fields to the file
            writer.writeheader()
            # writing data rows
            writer.writerows(dict_to_write)
