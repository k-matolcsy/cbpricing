from selenium.webdriver.common.by import By
from selenium import webdriver
import numpy as np
import time


class Scraper(object):
    path = r'C:\Users\Kristóf\Desktop\chromedriver_win32\chromedriver.exe'
    url = \
        'https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html'

    def data(self):
        # Start the WebDriver and load the page
        driver = webdriver.Chrome(self.path)
        driver.get(self.url)
        time.sleep(2)

        # Get the table
        driver.execute_script('javascript:charts[0].switchDimension(1,1);')
        table = driver.find_element_by_class_name('ecb-contentTable')

        # Get the data and store it in a list/array
        result = []
        rows = table.find_elements(By.TAG_NAME, "tr")
        for i, row in enumerate(rows):
            if i != 0:
                data = row.find_elements(By.TAG_NAME, "td")[1]
                result.append(data.text)
        return np.array(result)
