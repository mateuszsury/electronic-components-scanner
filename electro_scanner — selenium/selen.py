import difflib
import re

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

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


def find_element_online(ocr):

    options = Options()
    options.add_argument("--headless")  # Uruchom Chrome w trybie bez głowy

    driver = webdriver.Chrome(options=options)

    try:
        if len(ocr) == 1 or len(ocr) == 2:
            print('Short text')
            driver.get("https://www.alldatasheet.net/view_marking.jsp?Searchword={}".format(ocr))

            selector = driver.find_element(By.CSS_SELECTOR, "body > table:nth-child(16) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > b:nth-child(1) > font:nth-child(2)")
            elements = [None, None, None]

            for i, element in enumerate(elements):
                elements[i] = driver.find_element(By.CSS_SELECTOR,
                                                  "tbody tr[id='cell{}'] td:nth-child(3) a:nth-child(1)".format(
                                                      str(int(selector.text) - i)))
            datasheet_urls = []
            short_data = []
            for element in elements:
                try:
                    driver1 = webdriver.Chrome(options=options)
                    driver1.get(element.get_attribute('href'))
                    datasheet_link = driver1.find_element(By.CSS_SELECTOR, "body > div:nth-child(10) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(4) > td:nth-child(2) > a:nth-child(1)")
                    datasheet_urls.append(datasheet_link.get_attribute('href'))
                    data_table = driver1.find_element(By.XPATH, "//td[@class='newdatasheetyo'][count(a) > 0]")
                    extracted_a = data_table.find_elements(By.XPATH, '//a')
                    short_data_ = None
                    element_name = None
                    for a in extracted_a:
                        if a.text and a.text != 'ALLDATASHEET.NET' and a.text != 'Preview' and a.text != 'PDF' and a.text != 'Download' and a.text != 'HTML' and a.text != 'Chat AI' and a.text != 'Download Datasheet' and a.text != 'Privacy Policy' and a.text != 'More results':
                            if not short_data_:
                                element_name = a.text
                                short_data_ = a.text + ' '
                            else:
                                if not compare(element_name, a.text):
                                    if not a.text.isnumeric():
                                        short_data_ += a.text + " "
                                else:
                                    break
                    print(short_data_)
                    short_data.append(short_data_)
                except:
                    print('Failed to get element info')
            return short_data, datasheet_urls
            driver1.quit()
            driver.quit()
        else:
            try:
                print('Long text')
                driver.get("https://search.datasheetcatalog.net/key/{}".format(ocr))
                element = driver.find_element(By.CSS_SELECTOR, "a[title='{} datasheet']".format(ocr))

                datasheet_urls = None
                short_data = None

                print(element.get_attribute('href'))
                try:
                    driver1 = webdriver.Chrome(options=options)
                    driver1.get(element.get_attribute('href'))
                    datasheet_link = driver1.find_element(By.CSS_SELECTOR, "tbody tr:nth-child(1) td:nth-child(3) a:nth-child(2)")

                    regex = r"'(.*?)'"
                    match = re.search(regex, datasheet_link.get_attribute('href'))

                    if match:
                        datasheet_urls = match.group(1)
                        print(datasheet_urls)
                    else:
                        print("Error.")

                    short_data_ = driver1.find_element(By.XPATH, "/html[1]/body[1]/div[1]/main[1]/table[2]/tbody[1]/tr[1]/td[2]/a[1]")
                    short_data = short_data_.get_attribute('title')

                except:
                    print('Failed to get element info')
                driver1.quit()
                driver.quit()
                return short_data, datasheet_urls
            except:
                try:
                    driver.get("https://www.alldatasheet.net/view_marking.jsp?Searchword={}".format(ocr))

                    selector = driver.find_element(By.CSS_SELECTOR,
                                                   "body > table:nth-child(16) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > b:nth-child(1) > font:nth-child(2)")
                    elements = [None, None, None]
                    for i, element in enumerate(elements):
                        elements[i] = driver.find_element(By.CSS_SELECTOR,
                                                      "tbody tr[id='cell{}'] td:nth-child(3) a:nth-child(1)".format(
                                                          str(int(selector.text) - i)))

                    datasheet_urls = []
                    short_data = []
                    for element in elements:
                        try:
                            driver1 = webdriver.Chrome(options=options)
                            driver1.get(element.get_attribute('href'))
                            datasheet_link = driver1.find_element(By.CSS_SELECTOR,
                                                                  "body > div:nth-child(10) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(4) > td:nth-child(2) > a:nth-child(1)")
                            datasheet_urls.append(datasheet_link.get_attribute('href'))
                            data_table = driver1.find_element(By.XPATH, "//td[@class='newdatasheetyo'][count(a) > 0]")
                            extracted_a = data_table.find_elements(By.XPATH, '//a')
                            short_data_ = None
                            element_name = None
                            for a in extracted_a:
                                if a.text and a.text != 'ALLDATASHEET.NET' and a.text != 'Preview' and a.text != 'PDF' and a.text != 'Download' and a.text != 'HTML' and a.text != 'Chat AI' and a.text != 'Download Datasheet' and a.text != 'Privacy Policy' and a.text != 'More results':
                                    if not short_data_:
                                        element_name = a.text
                                        short_data_ = a.text + ' '
                                    else:
                                        if not compare(element_name, a.text):
                                            if not a.text.isnumeric():
                                                short_data_ += a.text + " "
                                        else:
                                            break
                            print(short_data_)
                            short_data.append(short_data_)
                        except:
                            print('Failed to get element info')
                    return short_data, datasheet_urls
                    driver1.quit()
                    driver.quit()
                except:
                    driver.get("https://www.alldatasheet.net/view.jsp?Searchword={}".format(ocr))
                    elements = [None, None, None]
                    for i, element in enumerate(elements):
                        elements[i] = driver.find_element(By.CSS_SELECTOR,
                                                          "a[aria-label='Go to preview'][onclick='myColorClick({})']".format(
                                                              str(10 - i)))

                    datasheet_urls = []
                    short_data = []

                    for element in elements:
                        try:
                            driver1 = webdriver.Chrome(options=options)
                            driver1.get(element.get_attribute('href'))
                            datasheet_link = driver1.find_element(By.CSS_SELECTOR,
                                                                  "td[id='modowny'] font[color='gray']")
                            datasheet_urls.append(modify_link(element.get_attribute('href')))
                            data_table = driver1.find_element(By.XPATH, "//td[@class='newdatasheetyo'][count(a) > 0]")
                            extracted_a = data_table.find_elements(By.XPATH, '//a')
                            short_data_ = None
                            element_name = None
                            for a in extracted_a:
                                if a.text and a.text != 'ALLDATASHEET.NET' and a.text != 'Preview' and a.text != 'PDF' and a.text != 'Download' and a.text != 'HTML' and a.text != 'Chat AI' and a.text != 'Download Datasheet' and a.text != 'Privacy Policy' and a.text != 'More results':
                                    if not short_data_:
                                        element_name = a.text
                                        short_data_ = a.text + ' '
                                    else:
                                        if not compare(element_name, a.text):
                                            if not a.text.isnumeric():
                                                short_data_ += a.text + " "
                                        else:
                                            break
                            print(short_data_)
                            short_data.append(short_data_)
                        except:
                            print('Failed to get element info')
                    return short_data, datasheet_urls
                    driver1.quit()
                    driver.quit()

    except:
        return 'Error'



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