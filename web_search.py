"""
Web Search Module for Research Agent

This module provides enhanced web search capabilities using multiple search engines
and web scraping functionality.
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time
from urllib.parse import urljoin, urlparse
import logging


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


class WebSearchEngine:
    """Base class for web search engines."""
    
    def __init__(self, api_key: str = None, config: Dict[str, Any] = None):
        self.api_key = api_key
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform search and return results."""
        raise NotImplementedError("Subclasses must implement search method")


class GoogleSearchEngine(WebSearchEngine):
    """Google Custom Search API implementation."""
    
    def __init__(self, api_key: str, search_engine_id: str, config: Dict[str, Any] = None):
        super().__init__(api_key, config)
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        if not self.api_key or not self.search_engine_id:
            self.logger.warning("Google API key or search engine ID not provided")
            return []
            
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)  # Google API max is 10 per request
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_google_results(data)
                    else:
                        self.logger.error(f"Google search failed with status {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Google search error: {e}")
            return []
            
    def _parse_google_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Google search API response."""
        results = []
        items = data.get('items', [])
        
        for item in items:
            result = SearchResult(
                title=item.get('title', ''),
                url=item.get('link', ''),
                snippet=item.get('snippet', ''),
                source='google',
                timestamp=datetime.now(),
                metadata={
                    'display_link': item.get('displayLink', ''),
                    'formatted_url': item.get('formattedUrl', '')
                }
            )
            results.append(result)
            
        return results


class BingSearchEngine(WebSearchEngine):
    """Bing Search API implementation."""
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        super().__init__(api_key, config)
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using Bing Search API."""
        if not self.api_key:
            self.logger.warning("Bing API key not provided")
            return []
            
        headers = {'Ocp-Apim-Subscription-Key': self.api_key}
        params = {
            'q': query,
            'count': min(num_results, 50),  # Bing allows up to 50
            'textDecorations': False,
            'textFormat': 'Raw'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_bing_results(data)
                    else:
                        self.logger.error(f"Bing search failed with status {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Bing search error: {e}")
            return []
            
    def _parse_bing_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Bing search API response."""
        results = []
        web_pages = data.get('webPages', {}).get('value', [])
        
        for item in web_pages:
            result = SearchResult(
                title=item.get('name', ''),
                url=item.get('url', ''),
                snippet=item.get('snippet', ''),
                source='bing',
                timestamp=datetime.now(),
                metadata={
                    'display_url': item.get('displayUrl', ''),
                    'date_last_crawled': item.get('dateLastCrawled', '')
                }
            )
            results.append(result)
            
        return results


class DuckDuckGoSearchEngine(WebSearchEngine):
    """DuckDuckGo search implementation (scraping-based)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(None, config)
        self.base_url = "https://duckduckgo.com/html/"
        
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using DuckDuckGo (web scraping)."""
        params = {'q': query}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_duckduckgo_results(html, num_results)
                    else:
                        self.logger.error(f"DuckDuckGo search failed with status {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            return []
            
    def _parse_duckduckgo_results(self, html: str, num_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo HTML response."""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find result containers
        result_containers = soup.find_all('div', class_='result')[:num_results]
        
        for container in result_containers:
            try:
                title_elem = container.find('a', class_='result__a')
                snippet_elem = container.find('a', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    result = SearchResult(
                        title=title_elem.get_text(strip=True),
                        url=title_elem.get('href', ''),
                        snippet=snippet_elem.get_text(strip=True),
                        source='duckduckgo',
                        timestamp=datetime.now()
                    )
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Error parsing DuckDuckGo result: {e}")
                continue
                
        return results


class WebContentScraper:
    """Scrapes content from web pages for detailed analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.WebContentScraper")
        self.session_timeout = self.config.get('timeout', 30)
        
    async def scrape_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a single URL."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.session_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._extract_content(html, url)
                    else:
                        self.logger.warning(f"Failed to scrape {url}: HTTP {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None
            
    def _extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract structured content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract title
        title = ""
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text(strip=True)
            
        # Extract main content
        content = ""
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '.main-content', '.article-content',
            '#content', '#main-content', '#article-content'
        ]
        
        for selector in main_selectors:
            main_elem = soup.select_one(selector)
            if main_elem:
                content = main_elem.get_text(strip=True, separator=' ')
                break
                
        # Fallback to body content
        if not content:
            body = soup.find('body')
            if body:
                content = body.get_text(strip=True, separator=' ')
                
        # Extract metadata
        meta_description = ""
        meta_elem = soup.find('meta', attrs={'name': 'description'})
        if meta_elem:
            meta_description = meta_elem.get('content', '')
            
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') or href.startswith('/'):
                full_url = urljoin(url, href)
                links.append({
                    'text': link.get_text(strip=True),
                    'url': full_url
                })
                
        return {
            'title': title,
            'content': content[:5000],  # Limit content length
            'meta_description': meta_description,
            'links': links[:20],  # Limit number of links
            'word_count': len(content.split()),
            'scraped_at': datetime.now().isoformat()
        }
    
    async def scrape_multiple(self, urls: List[str], max_concurrent: int = 5) -> Dict[str, Dict[str, Any]]:
        """Scrape content from multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> tuple:
            async with semaphore:
                content = await self.scrape_content(url)
                return url, content
                
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scraped_content = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Scraping task failed: {result}")
                continue
                
            url, content = result
            if content:
                scraped_content[url] = content
                
        return scraped_content


class EnhancedWebSearch:
    """Enhanced web search with multiple engines and content scraping."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.EnhancedWebSearch")
        self.engines = self._initialize_engines()
        self.scraper = WebContentScraper(config.get('scraper', {}))
        
    def _initialize_engines(self) -> Dict[str, WebSearchEngine]:
        """Initialize available search engines."""
        engines = {}
        
        # Google Custom Search
        if self.config.get('google_api_key') and self.config.get('google_search_engine_id'):
            engines['google'] = GoogleSearchEngine(
                self.config['google_api_key'],
                self.config['google_search_engine_id']
            )
            
        # Bing Search
        if self.config.get('bing_api_key'):
            engines['bing'] = BingSearchEngine(self.config['bing_api_key'])
            
        # DuckDuckGo (always available)
        engines['duckduckgo'] = DuckDuckGoSearchEngine()
        
        return engines
    
    async def search(self, query: str, num_results: int = 10, 
                    engines: List[str] = None, scrape_content: bool = False) -> List[SearchResult]:
        """
        Perform enhanced search across multiple engines.
        
        Args:
            query: Search query
            num_results: Number of results per engine
            engines: List of engines to use (None for all available)
            scrape_content: Whether to scrape full content from result URLs
            
        Returns:
            List of SearchResult objects
        """
        if engines is None:
            engines = list(self.engines.keys())
            
        # Filter to available engines
        available_engines = [e for e in engines if e in self.engines]
        
        if not available_engines:
            self.logger.warning("No available search engines")
            return []
            
        # Perform searches concurrently
        search_tasks = [
            self.engines[engine].search(query, num_results)
            for engine in available_engines
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine results
        all_results = []
        for i, results in enumerate(search_results):
            if isinstance(results, Exception):
                self.logger.error(f"Search failed for {available_engines[i]}: {results}")
                continue
            all_results.extend(results)
            
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
                
        # Optionally scrape content
        if scrape_content and unique_results:
            urls = [result.url for result in unique_results[:10]]  # Limit scraping
            scraped_content = await self.scraper.scrape_multiple(urls)
            
            # Add scraped content to results
            for result in unique_results:
                if result.url in scraped_content:
                    content_data = scraped_content[result.url]
                    result.metadata = result.metadata or {}
                    result.metadata.update({
                        'scraped_content': content_data['content'],
                        'scraped_title': content_data['title'],
                        'word_count': content_data['word_count']
                    })
                    
        return unique_results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available search engines."""
        return list(self.engines.keys())


# Example usage
async def main():
    """Example usage of the web search module."""
    config = {
        # Add your API keys here
        # 'google_api_key': 'your_google_api_key',
        # 'google_search_engine_id': 'your_search_engine_id',
        # 'bing_api_key': 'your_bing_api_key'
    }
    
    search = EnhancedWebSearch(config)
    
    # Perform search
    query = "artificial intelligence medical diagnosis 2024"
    results = await search.search(query, num_results=5, scrape_content=True)
    
    print(f"Found {len(results)} results for: {query}")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Source: {result.source}")
        print(f"   Snippet: {result.snippet[:100]}...")
        if result.metadata and 'scraped_content' in result.metadata:
            print(f"   Content preview: {result.metadata['scraped_content'][:100]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())