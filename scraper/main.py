from asyncio.log import logger
import json
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup
import requests


def run_scraper(soup: BeautifulSoup, url: str) -> List[Dict]:
    pass


def main(urls: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        save_path = output_dir / f"{Path(url).stem}.json"
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            # Ensure we handle HTTP errors
            response.raise_for_status()
        except requests.RequestException as e:
            # Log an error message if the request fails
            logger.error(f'Failed to fetch URL: "{url}" - {e}')
            continue

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.content, "lxml")

        try:
            # Run the scraper function to extract law articles
            law_articles = run_scraper(soup=soup, url=url)
        except Exception as e:
            # Log an error message if scraping fails
            logger.error(f'Failed to scrape data from URL: "{url}" - {e}')
            continue

        try:
            # Save the scraped data as a JSON file
            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(law_articles, file, indent=4, ensure_ascii=False)
            # Log a success message
            logger.info(f'Successfully saved data to "{save_path}"')
        except Exception as e:
            # Log an error message if saving fails
            logger.error(f'Failed to save data to "{save_path}" - {e}')


if __name__ == "__main__":
    main(["https://en.wikipedia.org/wiki/Paul_Allen"], Path("scraper/data"))
