import os
import urllib.request
import urllib.parse
import time
from typing import List, Dict, Any
import logging
import argparse
import xml.etree.ElementTree as ET
import re

from utils import ensure_directory, save_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_filename(title: str) -> str:
    """Convert a paper title to a safe filename."""
    # Replace special characters with underscores
    return re.sub(r'[^a-zA-Z0-9]', '_', title)[:100]


def fetch_arxiv_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Fetch paper metadata from arXiv API based on the query."""
    base_url = 'http://export.arxiv.org/api/query?'

    # URL encode the query to handle spaces and special characters
    encoded_query = urllib.parse.quote(query)
    search_query = f'search_query=all:{encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'

    logger.info(f"Fetching papers with query: {query}, max results: {max_results}")
    with urllib.request.urlopen(base_url + search_query) as response:
        response_text = response.read().decode('utf-8')

    # Parse the response
    root = ET.fromstring(response_text)

    # Define namespace
    namespace = {'atom': 'http://www.w3.org/2005/Atom',
                 'arxiv': 'http://arxiv.org/schemas/atom'}

    papers = []
    for entry in root.findall('.//atom:entry', namespace):
        paper = {}

        # Extract basic metadata
        paper['title'] = entry.find('./atom:title', namespace).text.strip()
        paper['summary'] = entry.find('./atom:summary', namespace).text.strip()
        paper['published'] = entry.find('./atom:published', namespace).text

        # Extract authors
        authors = entry.findall('./atom:author/atom:name', namespace)
        paper['authors'] = [author.text for author in authors]

        # Extract arxiv specific metadata
        paper['id'] = entry.find('./atom:id', namespace).text
        paper['arxiv_url'] = paper['id']  # Same as ID in this case

        # Extract PDF link
        links = entry.findall('./atom:link', namespace)
        for link in links:
            if link.attrib.get('title') == 'pdf':
                paper['pdf_url'] = link.attrib.get('href')
                break

        # Extract categories/tags
        categories = entry.findall('./arxiv:primary_category', namespace)
        paper['categories'] = [cat.attrib.get('term') for cat in categories]

        papers.append(paper)

    logger.info(f"Found {len(papers)} papers matching the query")
    return papers


def download_papers(papers: List[Dict[str, Any]], output_dir: str) -> None:
    """Download PDFs for a list of papers."""
    ensure_directory(output_dir)

    for i, paper in enumerate(papers):
        if 'pdf_url' not in paper:
            # Try to construct the PDF URL from the arxiv ID
            arxiv_id = paper['id'].split('/')[-1]
            paper['pdf_url'] = f'http://arxiv.org/pdf/{arxiv_id}.pdf'
            logger.info(f"Constructed PDF URL: {paper['pdf_url']}")
            logger.warning(f"No PDF URL found for paper: {paper['title']}")
            continue

        pdf_url = paper['pdf_url']
        if not pdf_url.endswith('.pdf'):
            pdf_url = pdf_url.replace('abs', 'pdf') + '.pdf'

        safe_filename = f"{clean_filename(paper['title'])}.pdf"
        output_path = os.path.join(output_dir, safe_filename)

        # Skip if already downloaded
        if os.path.exists(output_path):
            logger.info(f"Paper already exists: {output_path}")
            continue

        logger.info(f"Downloading [{i + 1}/{len(papers)}]: {paper['title']}")
        try:
            urllib.request.urlretrieve(pdf_url, output_path)
            logger.info(f"Successfully downloaded to {output_path}")

            # Save metadata
            metadata_path = os.path.join(output_dir, f"{clean_filename(paper['title'])}_metadata.json")
            save_json(paper, metadata_path)

            # Be nice to arXiv API - don't hammer it
            time.sleep(3)
        except Exception as e:
            logger.error(f"Failed to download {pdf_url}: {str(e)}")


def print_download_instructions(papers: List[Dict[str, Any]]) -> None:
    """Print manual download instructions for papers."""
    print("\nManual Download Instructions:")
    print("-------------------------------")
    print("If automatic download fails, use these links to download manually:")

    for i, paper in enumerate(papers):
        title = paper['title']
        if 'pdf_url' in paper:
            url = paper['pdf_url']
        else:
            arxiv_id = paper['id'].split('/')[-1]
            url = "http://arxiv.org/pdf/{arxiv_id}.pdf"

        print(f"{i + 1}) {title}")
        print(f"    URL: {url}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Download papers from arXiv')
    parser.add_argument('--query', type=str, default='retrieval augmented generation',
                        help='Search query for arXiv API')
    parser.add_argument('--max_results', type=int, default=10,
                        help='Maximum number of papers to download')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Directory to save downloaded papers')

    args = parser.parse_args()

    papers = fetch_arxiv_papers(args.query, args.max_results)

    # Try to download papers
    download_papers(papers, args.output_dir)

    # Also print manual instructions in case automatic download fails
    print_download_instructions(papers)

    # Save the collection metadata
    collection_metadata = {
        'query': args.query,
        'date_collected': time.strftime('%Y-%m-%d'),
        'papers': papers
    }
    save_json(collection_metadata, os.path.join(args.output_dir, 'collection_metadata.json'))


if __name__ == "__main__":
    main()
