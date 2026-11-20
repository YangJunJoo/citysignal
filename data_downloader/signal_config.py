"""Signal configuration inputs and NOEMI reports, independent of the extractor.

Browser dependencies are loaded only when a report download is requested.
Importing or running the feature extractor never requires a browser package.
"""
import os
from pathlib import Path
import time

import pandas as pd

REPORT_URL = 'https://noemi.cflsmartroads.com/ssv/report.html?id={}'
TABLE_IDS = ('intersection', 'controller', 'approach', 'lanes', 'signal')


def load_config_summary(path=None, **read_csv_kwargs):
    if path is None:
        path = os.environ.get('ATSPM_CONFIG_SUMMARY_PATH')
    if path is None:
        path = Path(os.environ.get('ATSPM_CONFIG_ROOT', '/data2/ATSPM/config_data')) / 'config_summary.csv'
    return pd.read_csv(path, **read_csv_kwargs)


def init_driver():
    """Use the same Chrome driver setup as the original report downloader."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    return webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                            options=webdriver.ChromeOptions())


def scrape_config_report(siia_id, signal_id, output_dir, *, driver_factory=None, wait_time=3):
    """Save supported report tables directly under the caller's output directory."""
    from bs4 import BeautifulSoup

    driver = (driver_factory or init_driver)()
    try:
        driver.get(REPORT_URL.format(siia_id))
        if wait_time:
            time.sleep(wait_time)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        saved = []
        for table in soup.find_all('table'):
            table_id = table.get('id')
            if table_id not in TABLE_IDS:
                continue
            head, body = table.find('thead'), table.find('tbody')
            if head is None or body is None:
                raise ValueError(f'Incomplete configuration table: {table_id}')
            columns = [cell.get_text(strip=True) for cell in head.find_all('th')]
            rows = [[cell.get_text(strip=True) for cell in row.find_all('th')] +
                    [cell.get_text(strip=True) for cell in row.find_all('td')]
                    for row in body.find_all('tr')]
            frame = pd.DataFrame(rows, columns=columns)
            destination = Path(output_dir) / table_id / f'{signal_id}.csv'
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.csv.tmp')
            try:
                frame.to_csv(temporary, index=False)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            saved.append(destination)
        return saved
    finally:
        driver.quit()
