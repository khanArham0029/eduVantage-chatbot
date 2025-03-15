from selenium.webdriver import ChromeOptions
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

SBR_WEBDRIVER = os.getenv("SBR_WEBDRIVER")


def scrape_website(website):
    print("Connecting to Scraping Browser...")
    options = ChromeOptions()
    options.add_argument("--headless")  # Run headless Chrome
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Specify the path to your local chromedriver.exe
    service = ChromeService(executable_path="./chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(website)
        print("Waiting captcha to solve...")
        # Add your captcha solving logic here if needed
        print("Navigated! Scraping page content...")
        html = driver.page_source
    finally:
        driver.quit()

    return html


def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get text or further process the content
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content


def split_dom_content(dom_content, max_length=6000):
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]
