import csv
import re
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Function to 
# 
# ape data from each post
def scrape_post(post, category):
    try:
        title_element = post.query_selector('h2.entry-title a')
        title = title_element.inner_text().strip()
        link = title_element.get_attribute('href').strip()
        
        date_element = post.query_selector('span.post-date-metadata')
        date = date_element.inner_text().strip() if date_element else 'N/A'
        
        meta_info = post.query_selector('p.fusion-single-line-meta').inner_text().split('/')
        read_time = meta_info[1].strip() if len(meta_info) > 1 else 'N/A'
        
        views_element = post.query_selector('p.fusion-single-line-meta i.fa-eye')
        if views_element:
            views_text = views_element.evaluate('el => el.nextSibling.textContent').strip()
            views = re.search(r'\d+', views_text)
            views = views.group() if views else 'N/A'
        else:
            views = 'N/A'
        
        return {
            'Title': title,
            'Link': link,
            'Date': date,
            'Reading Time': read_time,
            'Views': views,
            'Category': category.capitalize()
        }
    except Exception as e:
        print(f"Error scraping post: {e}")
        return None

# Function to scrape each page and return data
def scrape_page(page, url, category):
    try:
        page.goto(url, timeout=60000)  # Increased timeout to 60 seconds
        page.wait_for_selector('div.fusion-post-content', timeout=60000)
    except PlaywrightTimeoutError:
        print(f"Timeout while loading {url}. Skipping this page.")
        return []

    posts = page.query_selector_all('div.fusion-post-content')

    return [scrape_post(post, category) for post in posts if scrape_post(post, category) is not None]

# Function to save data to CSV
def save_to_csv(data, filename):
    if not data:
        print("No data to save.")
        return
    
    fieldnames = ['Title', 'Link', 'Date', 'Reading Time', 'Views', 'Category']
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Data successfully saved to '{filename}'.")

# Function to scrape the text content from the <div class="post-content"> of each link
def scrape_post_content(page, link):
    try:
        page.goto(link, timeout=60000)
        page.wait_for_selector('div.post-content', timeout=60000)
        title = page.title().strip()
        content_element = page.query_selector('div.post-content')
        content = content_element.inner_text().strip() if content_element else 'N/A'
        return {
            'Title': title,
            'Link': link,
            'Content': content
        }
    except PlaywrightTimeoutError:
        print(f"Timeout while loading {link}. Skipping this page.")
        return None

# Function to save the post content to a new CSV
def save_content_to_csv(data, filename='scraped_content.csv'):
    if not data:
        print("No data to save.")
        return
    
    fieldnames = ['Title', 'Link', 'Content']
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Content successfully saved to '{filename}'.")

def scrape_all_categories():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        all_data = []
        base_urls = [
            'https://interviewfocus.com/category/internship/page/',
            'https://interviewfocus.com/category/college/page/',
            'https://interviewfocus.com/category/career/page/'
        ]
        
        for base_url in base_urls:
            category_match = re.search(r'/category/([^/]+)/', base_url)
            category = category_match.group(1) if category_match else 'Unknown'
            
            page_number = 1
            while True:
                url = f'{base_url}{page_number}/'
                print(f"Scraping {url}...")
                data = scrape_page(page, url, category)
                
                if not data:
                    print(f"No data found on {url}. Moving to next category.")
                    break
                
                all_data.extend(data)
                page_number += 1
        
        # Save the first set of data to CSV
        first_csv_filename = 'scraped_data.csv'
        save_to_csv(all_data, first_csv_filename)
        
        # Read links from the first CSV and scrape content
        second_csv_data = []
        with open(first_csv_filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                link = row['Link']
                print(f"Scraping content from {link}...")
                content_data = scrape_post_content(page, link)
                if content_data:
                    second_csv_data.append(content_data)
        
        # Save the scraped content to the second CSV
        save_content_to_csv(second_csv_data)
        
        browser.close()

if __name__ == "__main__":
    scrape_all_categories()
