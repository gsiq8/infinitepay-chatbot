import requests
from bs4 import BeautifulSoup
import time
import json
import os
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass
from typing import List, Dict, Set
import re
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    url: str
    title: str
    content: str
    content_type: str
    metadata: Dict
    
class SimpleInfinitePayScraper:
    def __init__(self, base_url: str = "https://www.infinitepay.io", delay: float = 1.5):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.visited_urls: Set[str] = set()
        self.scraped_content: List[ScrapedContent] = []
        
        # Security filters for sensitive content
        self.sensitive_patterns = [
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card numbers
            r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # CPF-like patterns
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b(?:senha|password|token|key|secret)\s*[:=]\s*\S+',  # Passwords/tokens
        ]
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the target domain"""
        parsed = urlparse(url)
        return parsed.netloc in ['www.infinitepay.io', 'infinitepay.io']
    
    def sanitize_content(self, content: str) -> str:
        """Remove sensitive information from content"""
        sanitized = content
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(pattern, '[DADOS_PROTEGIDOS]', sanitized, flags=re.IGNORECASE)
        return sanitized
    
    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def scrape_page(self, url: str) -> ScrapedContent:
        """Scrape a single page"""
        try:
            logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Use html.parser (built-in) instead of lxml
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Extract main content
            content = self.extract_text_content(soup)
            
            # Sanitize content for security
            content = self.sanitize_content(content)
            
            # Extract metadata
            metadata = {
                'description': '',
                'keywords': '',
                'author': '',
                'lang': 'pt',  # Default to Portuguese
            }
            
            # Extract meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', '').lower()
                property_attr = meta.get('property', '').lower()
                content_attr = meta.get('content', '')
                
                if name in ['description', 'keywords', 'author']:
                    metadata[name] = content_attr
                elif property_attr == 'og:description':
                    metadata['description'] = content_attr
                elif name == 'language' or meta.get('http-equiv', '').lower() == 'content-language':
                    metadata['lang'] = content_attr
            
            return ScrapedContent(
                url=url,
                title=title_text,
                content=content,
                content_type='webpage',
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def discover_urls(self, start_url: str, max_depth: int = 2) -> List[str]:
        """Discover URLs to scrape"""
        urls_to_visit = [(start_url, 0)]
        discovered_urls = set()
        
        while urls_to_visit:
            current_url, depth = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls or depth > max_depth:
                continue
                
            if not self.is_valid_url(current_url):
                continue
                
            try:
                response = self.session.get(current_url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    
                    if self.is_valid_url(full_url) and full_url not in discovered_urls:
                        discovered_urls.add(full_url)
                        if depth < max_depth:
                            urls_to_visit.append((full_url, depth + 1))
                
                self.visited_urls.add(current_url)
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error discovering URLs from {current_url}: {str(e)}")
        
        return list(discovered_urls)
    
    def scrape_website(self, start_url: str = None, max_pages: int = 20) -> List[ScrapedContent]:
        """Scrape the entire website"""
        if start_url is None:
            start_url = self.base_url
            
        logger.info(f"Starting website scrape from: {start_url}")
        
        # Discover URLs
        urls = self.discover_urls(start_url)[:max_pages]
        logger.info(f"Found {len(urls)} URLs to scrape")
        
        # Scrape each URL
        for url in urls:
            content = self.scrape_page(url)
            if content and len(content.content.strip()) > 100:  # Only keep substantial content
                self.scraped_content.append(content)
            
            time.sleep(self.delay)  # Be respectful
        
        logger.info(f"Scraped {len(self.scraped_content)} pages successfully")
        return self.scraped_content
    
    def save_content(self, output_dir: str = "scraped_data"):
        """Save scraped content to various formats"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save as JSON
        json_data = []
        for content in self.scraped_content:
            json_data.append({
                'url': content.url,
                'title': content.title,
                'content': content.content,
                'content_type': content.content_type,
                'metadata': content.metadata
            })
        
        with open(f"{output_dir}/scraped_content.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Save as individual text files
        text_dir = Path(output_dir) / "text_files"
        text_dir.mkdir(exist_ok=True)
        
        for i, content in enumerate(self.scraped_content):
            filename = f"page_{i:03d}_{content.title[:50].replace('/', '_')}.txt"
            with open(text_dir / filename, 'w', encoding='utf-8') as f:
                f.write(f"URL: {content.url}\n")
                f.write(f"Title: {content.title}\n")
                f.write(f"Language: {content.metadata.get('lang', 'pt')}\n")
                f.write("=" * 50 + "\n")
                f.write(content.content)
        
        # Save summary
        summary = {
            'total_pages': len(self.scraped_content),
            'total_words': sum(len(c.content.split()) for c in self.scraped_content),
            'languages': list(set(c.metadata.get('lang', 'pt') for c in self.scraped_content)),
            'content_types': list(set(c.content_type for c in self.scraped_content)),
            'urls_scraped': [c.url for c in self.scraped_content]
        }
        
        with open(f"{output_dir}/scraping_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Content saved to {output_dir}/")
        logger.info(f"Summary: {summary['total_pages']} pages, {summary['total_words']} words")

def main():
    """Main scraping function"""
    scraper = SimpleInfinitePayScraper(delay=2.0)  # Be extra respectful
    
    # Start scraping
    content = scraper.scrape_website(
        start_url="https://www.infinitepay.io",
        max_pages=20  # Reduced for demo
    )
    
    # Save content
    scraper.save_content("infinitepay_data")
    
    print(f"\n✅ Scraping completed!")
    print(f"📄 {len(content)} pages scraped")
    print(f"📁 Data saved to 'infinitepay_data/' directory")
    print(f"🔍 Check scraping_summary.json for overview")

if __name__ == "__main__":
    main()