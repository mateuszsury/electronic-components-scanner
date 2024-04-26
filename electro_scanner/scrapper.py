import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import csv
from tqdm import tqdm
import string


def get_manufacturers():
    url = 'https://www.datasheetcatalog.com/other_manufacturer.html'
    page = requests.get(url)
    soup_page = BeautifulSoup(page.text, features='html.parser')
    tables = soup_page.findAll('table')
    del soup_page
    table = tables[4]
    del tables
    links = table.findAll('a')
    hrefs = []
    for link in links:
        if link.get('href'):
            hrefs.append(link)
    return hrefs


def process_link(link, progress_bar):
    i = 1
    while True:
        try:
            with open('elements.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                end_link = link.get('href')[:-2] + str(i) + '/'
                new_url = 'https://www.datasheetcatalog.com' + end_link
                i += 1
                new_page = requests.get(new_url)
                new_soup_page = BeautifulSoup(new_page.text, features='html.parser')
                element_paragrafs = new_soup_page.find('table', class_="TabelMare").findAll('a')
                for el in element_paragrafs:
                    writer.writerow([el.text])
        except:
            progress_bar.update(1)
            break


def alldatasheet(asci, progress_bar):
    for i in range(1, 201):
        try:
            with open('elements.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                url = f'https://components.alldatasheet.net/all.jsp?index={asci}&page={i}'
                page = requests.get(url)
                soup_page = BeautifulSoup(page.text, features='html.parser')
                table = soup_page.findAll('a')
                for element in table:
                    if 'Searchword' in str(element) and 'style' in str(element):
                        writer.writerow([element.text])
            progress_bar.update(1)
        except:
            break


def main():
    links = get_manufacturers()

    with open('elements.csv', 'w', newline=''):
        pass
    progress_bar2 = tqdm(total=len(string.ascii_uppercase) * 200, desc="Scraper 1")
    with ThreadPoolExecutor() as executor:
        for asci in string.ascii_uppercase:
            executor.submit(alldatasheet, asci, progress_bar2)
        executor.shutdown(wait=True)
    progress_bar2.close()

    progress_bar1 = tqdm(total=len(links), desc="Scraper 2")
    with ThreadPoolExecutor() as executor:
        for link in links:
            executor.submit(process_link, link, progress_bar1)
        executor.shutdown(wait=True)
    progress_bar1.close()






if __name__ == "__main__":
    main()
