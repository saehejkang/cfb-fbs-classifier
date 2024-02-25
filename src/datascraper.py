import requests
from bs4 import BeautifulSoup


def read_from_url(url):
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
        print(stat.text)
        # output the stat value
        print(value.text.strip())


def write_to_csv():
    # TODO
    return
