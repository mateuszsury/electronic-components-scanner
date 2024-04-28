import difflib
import requests
from bs4 import BeautifulSoup

def smd_resistor_value(code):
    try:
        if code.startswith('R'):
            value = float(code[1:]) / 10
            return str(value) + 'Ω'
        elif 'R' in code:
            digits, multiplier = code.split('R')
            value = float(digits + '.' + multiplier)
            return str(value) + 'Ω'
        elif len(code) == 3:
            digits = code[:2]
            multiplier = int(code[2])
        elif len(code) == 4:
            digits = code[:3]
            multiplier = int(code[3])
        else:
            return False

        value = int(digits) * (10 ** multiplier)
        if value >= 1000000:
            return str(value / 1000000) + 'MΩ'
        elif value >= 1000:
            return str(value / 1000) + 'kΩ'
        else:
            return str(value) + 'Ω'
    except:
        return False

def compare(string1, string2):
    matcher = difflib.SequenceMatcher(None, string1, string2)
    similarity = matcher.ratio()
    if similarity >= 0.5:
        return True
    else:
        return False



def find_element_online(ocr):

    try:

        if len(ocr) <= 2:
            print('Short text')
            url = "https://www.alldatasheet.net/view_marking.jsp?Searchword={}".format(ocr)
            page = requests.get(url)
            soup_page = BeautifulSoup(page.text, features='html.parser')
            tables = soup_page.findAll('table')
            table = tables[11]
            links = table.findAll('a')
            hrefs = []
            for i in range(0, len(links), 4):
                try:
                    if i == 3*4:
                        break
                    hrefs.append(links[i].get('href')[2:])
                except:
                    print(f"{i/4} element not found")
            _data = []
            _datasheets = []
            for href in hrefs:
                try:
                    complete_href = 'https://' + href
                    new_page = requests.get(complete_href)
                    new_soup_page = BeautifulSoup(new_page.text, features='html.parser')
                    data = new_soup_page.find('meta', {'name': 'description'})
                    if data:
                        content = data['content']
                        marking = content.split('Marking: ')[1]
                        marking = marking.split('. Part ')[0]
                        if marking != ocr:
                            raise Exception('Data not found')
                        description = content.split('. Description: ')[1]
                        description = description.split('. Manufacturer')[0]
                        _data.append(description)
                        _datasheets.append(complete_href.replace('https://www.alldatasheet.net/datasheet-pdf/marking', 'https://pdf1.alldatasheet.net/datasheet-pdf/view-marking'))
                except:
                    print('Error while getting data')
            return _data, _datasheets
        else:
            try:
                print('Long text')
                url = "https://www.alldatasheet.net/view.jsp?Searchword={}".format(ocr)
                print(url)
                page = requests.get(url)
                soup_page = BeautifulSoup(page.text, features='html.parser')
                tables = soup_page.findAll('table')
                table = tables[13]
                links = table.findAll('a')
                hrefs = []
                for i in range(0, len(links), 4):
                    try:
                        if i == 3 * 4:
                            break
                        hrefs.append(links[i].get('href')[2:])
                    except:
                        print(f"{i / 4} element not found")
                _data = []
                _datasheets = []
                for href in hrefs:
                    try:
                        complete_href = 'https://' + href
                        new_page = requests.get(complete_href)
                        new_soup_page = BeautifulSoup(new_page.text, features='html.parser')
                        data = new_soup_page.find('meta', {'name': 'description'})
                        if data:
                            content = data['content']
                            marking = content.split('Part #: ')[1]
                            marking = marking.split('. Download')[0]
                            if marking != ocr:
                                raise Exception('Data not found')
                            description = content.split('. Description: ')[1]
                            description = description.split('. Manufacturer')[0]
                            _data.append(description)
                            _datasheets.append(
                                complete_href.replace('https://www.alldatasheet.net/datasheet-pdf/marking',
                                                      'https://pdf1.alldatasheet.net/datasheet-pdf/view-marking'))
                    except:
                        print('Error while getting data')
                if not _data:
                    raise Exception
            except:
                print('Data not found, checking other sites..')
                _data = []
                _datasheets = []
                try:
                    url = "https://search.datasheetcatalog.net/key/{}".format(ocr)
                    print(url)
                    page = requests.get(url)
                    soup_page = BeautifulSoup(page.text, features='html.parser')
                    tables = soup_page.findAll('table')
                    table = tables[3]
                    short_data = table.findAll('td')
                    _data = []
                    for i in range(2, len(short_data), 4):
                        while len(_data) < 3:
                            _data.append(str(short_data[i]).strip('<td>').strip('</'))
                        break

                    links = table.findAll('a')
                    hrefs = []
                    i = 0
                    j = 0
                    err = 0
                    while True:
                        try:
                            if i == 3:
                                break
                            if links[j].get('title').strip(' datasheet') == ocr:
                                hrefs.append(links[j].get('href'))
                                i += 1
                                j += 1
                            else:
                                j += 1
                                continue
                        except:
                            err += 1
                            if err > 50:
                                break
                    _datasheets = []
                    for href in hrefs:
                        try:
                            new_page = requests.get(href)
                            new_soup_page = BeautifulSoup(new_page.text, features='html.parser')
                            datasheet = new_soup_page.findAll('a')
                            _datasheet = None
                            for d in datasheet:
                                if len(str(d.get('href'))) != 33 and len(str(d.get('href'))) != 36 and 'https://' in str(d.get('href')) and 'javascript' in str(d):
                                    _datasheet = str(d.get('href')).split("'")[1]
                            _datasheets.append(_datasheet)

                        except:
                            print('Error while getting data')
                except:
                    print('Element not found')
                if not _data or not _datasheets:
                    print('Short text')
                    url = "https://www.alldatasheet.net/view_marking.jsp?Searchword={}".format(ocr)
                    page = requests.get(url)
                    soup_page = BeautifulSoup(page.text, features='html.parser')
                    tables = soup_page.findAll('table')
                    table = tables[11]
                    links = table.findAll('a')
                    hrefs = []
                    for i in range(0, len(links), 4):
                        try:
                            if i == 3 * 4:
                                break
                            hrefs.append(links[i].get('href')[2:])
                        except:
                            print(f"{i / 4} element not found")
                    _data = []
                    _datasheets = []
                    for href in hrefs:
                        try:
                            complete_href = 'https://' + href
                            new_page = requests.get(complete_href)
                            new_soup_page = BeautifulSoup(new_page.text, features='html.parser')
                            data = new_soup_page.find('meta', {'name': 'description'})
                            if data:
                                content = data['content']
                                marking = content.split('Marking: ')[1]
                                marking = marking.split('. Part ')[0]
                                if marking != ocr:
                                    raise Exception('Data not found')
                                description = content.split('. Description: ')[1]
                                description = description.split('. Manufacturer')[0]
                                _data.append(description)
                                _datasheets.append(
                                    complete_href.replace('https://www.alldatasheet.net/datasheet-pdf/marking',
                                                          'https://pdf1.alldatasheet.net/datasheet-pdf/view-marking'))
                        except:
                            print('Error while getting data')
                    if not _data:
                        raise Exception
                    return _data, _datasheets

            else:
                if len(_data) > len(_datasheets):
                    del _data[len(_datasheets):]
                elif len(_datasheets) > len(_data):
                    del _datasheets[len(_data):]
                return _data, _datasheets
    except:
        return None
        print('Element not found')

def find_element(ocr):
    if len(ocr) == 4 or len(ocr) == 3:
        resistor = smd_resistor_value(ocr)
        if not resistor:
            return find_element_online(ocr)
        else:
            print('Resistor text')
            data = ('Resistor', resistor)
            return data
    else:
        return find_element_online(ocr)

def modify_link(original_link):
    parts = original_link.split('/')
    parts[2] = 'pdf1.alldatasheet.net'  # Zamiana "www" na "pdf1"
    parts[-4] = 'download'
    return '/'.join(parts)