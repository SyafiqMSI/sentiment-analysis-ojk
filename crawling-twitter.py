from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHROME_PATH = "/driver/chromedriver.exe"
GECKO_PATH = "/driver/geckodriver.exe"

def create_driver(use_chrome):
    if use_chrome:
        options = ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-infobars")
        options.add_argument("webdriver.chrome.driver=" + CHROME_PATH)
        options.headless = False
        # service = ChromeService(executable_path=CHROME_PATH)
        return webdriver.Chrome(options=options)
    else:
        options = FirefoxOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-infobars")
        options.headless = False
        service = FirefoxService(executable_path=GECKO_PATH)
        return webdriver.Firefox(service=service, options=options)

def login(driver):
    try:
        username_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//input[@autocomplete='username']"))
        )
        username_input.send_keys(os.getenv('TWITTER_USERNAME'))
        
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']/ancestor::button"))
        )
        next_button.click()

        password_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//input[@name='password']"))
        )
        password_input.send_keys(os.getenv('TWITTER_PASSWORD'))
        
        login_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']/ancestor::button"))
        )
        login_button.click()

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//div[@data-testid='primaryColumn']"))
        )
        logging.info("Login berhasil")
    except Exception as e:
        logging.error(f"Gagal login: {e}")
        raise
    
def get_tweet(element):
    try:
        user = element.find_element(By.XPATH, ".//*[contains(text(), '@')]").text
        text = element.find_element(By.XPATH, ".//div[@lang]").text
        date = element.find_element(By.XPATH, ".//time").get_attribute("datetime")

        try:
            reply_to_user = element.find_element(By.XPATH, ".//div[@dir='ltr']//a[contains(@href, '/')]").text
        except:
            reply_to_user = None

        try:
            tweet_link = element.find_element(By.XPATH, ".//a[@href and contains(@href, '/status/')]").get_attribute("href")
        except:
            tweet_link = None

        try:
            media_links = [media.get_attribute("src") for media in element.find_elements(By.XPATH, ".//img[contains(@src, 'media')]")]
        except:
            media_links = []

        tweet_data = [user, text, date, tweet_link, media_links, reply_to_user]
        return tweet_data
    except Exception as e:
        logging.error(f"Error extracting tweet data: {e}")
        return None
    
def click_retry_button(driver):
    try:
        retry_button = driver.find_element(By.XPATH, "//div[text()='Retry']/parent::div")
        retry_button.click()
        logging.info("Clicked the retry button")
        WebDriverWait(driver, 15).until(EC.invisibility_of_element_located((By.XPATH, "//div[text()='Retry']")))
    except:
        pass

def handle_technical_error(driver):
    try:
        error_message = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'technical error')]"))
        )
        if error_message:
            logging.warning("Technical error detected. Refreshing the page.")
            driver.refresh()
            time.sleep(10)
    except:
        pass

def scrape_tweets(driver):
    user_data, text_data, date_data, tweet_link_data, media_link_data, reply_to_data = [], [], [], [], [], []
    tweet_ids = set()
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            logging.info("Searching for tweets...")
            tweets = WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.XPATH, "//article")))
            logging.info(f"Number of tweets found: {len(tweets)}")
            
            if len(tweets) == 0:
                logging.warning("No tweets found. Retrying...")
                retry_count += 1
                time.sleep(10)
                driver.refresh()
                continue

            for tweet in tweets:
                tweet_list = get_tweet(tweet)
                if tweet_list:
                    tweet_id = ''.join(tweet_list[:2])
                    if tweet_id not in tweet_ids:
                        tweet_ids.add(tweet_id)
                        user_data.append(tweet_list[0])
                        text_data.append(" ".join(tweet_list[1].split()))
                        date_data.append(tweet_list[2])
                        tweet_link_data.append(tweet_list[3])
                        media_link_data.append(tweet_list[4])
                        reply_to_data.append(tweet_list[5])

            if len(user_data) >= 10000:
                break

            last_height = driver.execute_script("return document.body.scrollHeight")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            new_height = driver.execute_script("return document.body.scrollHeight")

            click_retry_button(driver)
            handle_technical_error(driver)

            if new_height == last_height:
                break

        except Exception as e:
            logging.error(f"Error during scrolling or tweet retrieval: {e}")
            retry_count += 1
            time.sleep(10)
            driver.refresh()

    return user_data, text_data, date_data, tweet_link_data, media_link_data, reply_to_data

def save_daily_results(df, date):
    os.makedirs('data', exist_ok=True)
    filename = f'data/hasil-crawling-{date.strftime("%Y-%m-%d")}.csv'
    df.to_csv(filename, index=False, sep=";")
    
    logging.info(f"Saved results to {filename}")

def main():
    os.makedirs('log', exist_ok=True)
    start_date = datetime(2021, 5, 1)
    end_date = datetime(2024, 5, 31)
    current_date = end_date
    use_chrome = True  # Mulai dengan Chrome

    while current_date >= start_date:
        try:
            driver = create_driver(use_chrome)
            logging.info(f"Using {'Chrome' if use_chrome else 'Firefox'} for date: {current_date.strftime('%Y-%m-%d')}")
            
            search_url = f'https://x.com/search?q=OJK+lang%3Aid+until%3A{current_date.strftime("%Y-%m-%d")}+since%3A{start_date.strftime("%Y-%m-%d")}&src=typed_query&f=live'
            driver.get(search_url)
            
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@autocomplete='username']"))
                )
                logging.info("Halaman login terdeteksi, mencoba login...")
                login(driver)
            except:
                logging.info("Tidak perlu login, melanjutkan ke pencarian...")

            time.sleep(15)
            driver.save_screenshot(f'log/screenshot_before_search_{current_date.strftime("%Y-%m-%d")}.png')

            user_data, text_data, date_data, tweet_link_data, media_link_data, reply_to_data = scrape_tweets(driver)

            driver.save_screenshot(f'log/screenshot_after_search_{current_date.strftime("%Y-%m-%d")}.png')
            driver.quit()

            df = pd.DataFrame({
                'User': user_data,
                'Text': text_data,
                'Date': date_data,
                'Tweet Link': tweet_link_data,
                'Media Links': media_link_data,
                'Reply To': reply_to_data
            })

            logging.info(f"DataFrame for {current_date.strftime('%Y-%m-%d')}:")
            logging.info(df)

            save_daily_results(df, current_date)

            current_date -= timedelta(days=1)
            use_chrome = not use_chrome 

        except Exception as e:
            logging.error(f"Error occurred: {e}")
            logging.info("Possibly reached daily limit or encountered an error. Closing browser and trying again.")
            try:
                driver.quit()
            except:
                pass
            time.sleep(60) 
            continue

if __name__ == "__main__":
    main()