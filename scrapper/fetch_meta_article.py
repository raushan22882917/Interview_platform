import csv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

def extract_article_data_to_csv(url, output_file, retries=3):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        attempt = 0
        while attempt < retries:
            try:
                page.goto(url, timeout=60000)

                # Select all article cards
                article_cards = page.query_selector_all('a.article-card')
                if article_cards:
                    # Prepare the CSV file
                    with open(output_file, 'w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Tag', 'Date', 'Title', 'Excerpt', 'Read More Link', 'Image URL'])

                        # Extract details from each article card
                        for article_card in article_cards:
                            # Use default values for missing elements
                            tag_element = article_card.query_selector('span.article-card-tag')
                            tag = tag_element.inner_text() if tag_element else 'N/A'

                            date_element = article_card.query_selector('span.article-card-date')
                            date = date_element.inner_text() if date_element else 'N/A'

                            title_element = article_card.query_selector('div.article-card-title')
                            title = title_element.inner_text() if title_element else 'N/A'

                            excerpt_element = article_card.query_selector('div.article-card-excerpt')
                            excerpt = excerpt_element.inner_text() if excerpt_element else 'N/A'

                            read_more_link = article_card.get_attribute('href') or 'N/A'

                            image_element = article_card.query_selector('img')
                            image_url = image_element.get_attribute('src') if image_element else 'N/A'

                            # Write the data to the CSV
                            writer.writerow([tag, date, title, excerpt, read_more_link, image_url])

                    print(f"Data extracted and saved to {output_file}")
                    break  # Exit the loop if successful

            except PlaywrightTimeoutError:
                attempt += 1
                print(f"Attempt {attempt} failed to load the page: {url}. Retrying...")

        browser.close()

# URL and output CSV file
url = "https://igotanoffer.com/blogs/tech"
output_file = "data/article_data.csv"

# Extract data and save to CSV
extract_article_data_to_csv(url, output_file)

print(f"Data extraction process completed.")
