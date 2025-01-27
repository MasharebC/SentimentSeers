import requests
from bs4 import BeautifulSoup

def scrape_amazon_reviews(product_url, num_pages=1):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    reviews = []
    for page in range(1, num_pages + 1):
        page_url = f"{product_url}&pageNumber={page}"
        print(f"Scraping page {page}...")
        response = requests.get(page_url, headers=headers)

        # Debugging information
        print(f"Status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: Unable to retrieve page. Response text: {response.text[:500]}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        review_blocks = soup.find_all("div", {"data-hook": "review"})

        for block in review_blocks:
            try:
                title = block.find("a", {"data-hook": "review-title"}).text.strip()
                body = block.find("span", {"data-hook": "review-body"}).text.strip()
                stars = block.find("i", {"data-hook": "review-star-rating"}).text.strip()
                reviews.append({
                    "title": title,
                    "body": body,
                    "stars": stars,
                })
            except AttributeError:
                continue

    return reviews

if __name__ == "__main__":
    product_reviews_url = input("Please enter the URL of the Amazon product reviews page: ").strip()
    num_pages = int(input("How many pages of reviews would you like to scrape? (e.g., 1-10): ").strip())
    
    print("\nScraping reviews... This may take a few seconds.")
    reviews = scrape_amazon_reviews(product_reviews_url, num_pages)

    if reviews:
        print(f"\nScraped {len(reviews)} reviews successfully!")
        for review in reviews[:5]:  # Display the first 5 reviews
            print(review)
    else:
        print("No reviews were scraped. Please check the URL or try again.")
