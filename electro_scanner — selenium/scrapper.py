import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import csv
from tqdm import tqdm


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
                # Inkrementacja paska postępu
        except:
            progress_bar.update(1)
            break


def main():
    links = get_manufacturers()
    progress_bar = tqdm(total=len(links), desc="Progress")

    with open('elements.csv', 'w', newline=''):
        pass

    with ThreadPoolExecutor() as executor:
        for link in links:
            executor.submit(process_link, link, progress_bar)


    # Zakonczenie pracy paska postępu
    progress_bar.close()


if __name__ == "__main__":
    main()
