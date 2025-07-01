"""
API Integration Module for Research Agent

This module provides a flexible framework for integrating various APIs
and external services into the research workflow.
"""

import asyncio
import aiohttp
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
from urllib.parse import urljoin, urlparse
import hashlib
import pickle
import os


@dataclass
class APIResponse:
    """Standardized API response format."""
    success: bool
    data: Any
    status_code: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""


@dataclass
class APIEndpoint:
    """Configuration for an API endpoint."""
    name: str
    base_url: str
    auth_type: str  # 'none', 'api_key', 'bearer', 'basic', 'oauth'
    auth_config: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    rate_limit: Optional[Dict[str, Any]] = None
    timeout: int = 30
    retry_config: Dict[str, Any] = field(default_factory=dict)


class APIClient(ABC):
    """Abstract base class for API clients."""
    
    def __init__(self, endpoint: APIEndpoint, cache_dir: str = None):
        self.endpoint = endpoint
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.session = None
        self.rate_limiter = RateLimiter(endpoint.rate_limit) if endpoint.rate_limit else None
        self.cache = APICache(cache_dir) if cache_dir else None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.endpoint.timeout),
            headers=self.endpoint.headers
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    @abstractmethod
    async def make_request(self, method: str, path: str, **kwargs) -> APIResponse:
        """Make a request to the API."""
        pass
        
    def _prepare_auth(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Prepare authentication headers."""
        auth_headers = headers.copy()
        
        if self.endpoint.auth_type == 'api_key':
            key_name = self.endpoint.auth_config.get('key_name', 'X-API-Key')
            api_key = self.endpoint.auth_config.get('api_key')
            if api_key:
                auth_headers[key_name] = api_key
                
        elif self.endpoint.auth_type == 'bearer':
            token = self.endpoint.auth_config.get('token')
            if token:
                auth_headers['Authorization'] = f'Bearer {token}'
                
        elif self.endpoint.auth_type == 'basic':
            username = self.endpoint.auth_config.get('username')
            password = self.endpoint.auth_config.get('password')
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                auth_headers['Authorization'] = f'Basic {credentials}'
                
        return auth_headers


class HTTPAPIClient(APIClient):
    """Generic HTTP API client."""
    
    async def make_request(self, method: str, path: str, 
                          params: Dict[str, Any] = None,
                          data: Dict[str, Any] = None,
                          json_data: Dict[str, Any] = None,
                          use_cache: bool = True,
                          cache_ttl: int = 3600) -> APIResponse:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            params: Query parameters
            data: Form data
            json_data: JSON data for request body
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds
            
        Returns:
            APIResponse object
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
            
        # Prepare URL
        url = urljoin(self.endpoint.base_url, path)
        
        # Check cache first
        cache_key = None
        if use_cache and self.cache and method.upper() == 'GET':
            cache_key = self._generate_cache_key(url, params)
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                self.logger.debug(f"Cache hit for {url}")
                return cached_response
                
        # Rate limiting
        if self.rate_limiter:
            await self.rate_limiter.wait()
            
        # Prepare headers with authentication
        headers = self._prepare_auth({})
        
        try:
            # Make request with retry logic
            response_data = await self._make_request_with_retry(
                method, url, params, data, json_data, headers
            )
            
            api_response = APIResponse(
                success=True,
                data=response_data,
                status_code=200,
                source=self.endpoint.name,
                metadata={'url': url, 'method': method}
            )
            
            # Cache successful GET requests
            if use_cache and self.cache and method.upper() == 'GET' and cache_key:
                await self.cache.set(cache_key, api_response, cache_ttl)
                
            return api_response
            
        except Exception as e:
            self.logger.error(f"API request failed for {url}: {e}")
            return APIResponse(
                success=False,
                data=None,
                status_code=getattr(e, 'status', 500),
                error_message=str(e),
                source=self.endpoint.name,
                metadata={'url': url, 'method': method}
            )
            
    async def _make_request_with_retry(self, method: str, url: str,
                                     params: Dict[str, Any] = None,
                                     data: Dict[str, Any] = None,
                                     json_data: Dict[str, Any] = None,
                                     headers: Dict[str, str] = None) -> Any:
        """Make request with retry logic."""
        retry_config = self.endpoint.retry_config
        max_retries = retry_config.get('max_retries', 3)
        backoff_factor = retry_config.get('backoff_factor', 1.0)
        retry_statuses = retry_config.get('retry_statuses', [429, 500, 502, 503, 504])
        
        for attempt in range(max_retries + 1):
            try:
                async with self.session.request(
                    method, url, params=params, data=data,
                    json=json_data, headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status in retry_statuses and attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        self.logger.warning(f"Request failed with status {response.status}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                        
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    self.logger.warning(f"Request timeout, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise
                    
        raise RuntimeError(f"Max retries exceeded for {url}")
        
    def _generate_cache_key(self, url: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key for request."""
        key_data = f"{url}:{json.dumps(params, sort_keys=True) if params else ''}"
        return hashlib.md5(key_data.encode()).hexdigest()


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, config: Dict[str, Any]):
        self.requests_per_second = config.get('requests_per_second', 1)
        self.requests_per_minute = config.get('requests_per_minute')
        self.requests_per_hour = config.get('requests_per_hour')
        
        self.last_request_time = 0
        self.request_times = []
        
    async def wait(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        # Per-second rate limiting
        if self.requests_per_second:
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.requests_per_second
            
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                await asyncio.sleep(wait_time)
                current_time = time.time()
                
        # Per-minute/hour rate limiting
        if self.requests_per_minute or self.requests_per_hour:
            self.request_times.append(current_time)
            
            # Clean old timestamps
            if self.requests_per_minute:
                cutoff = current_time - 60
                self.request_times = [t for t in self.request_times if t > cutoff]
                
                if len(self.request_times) >= self.requests_per_minute:
                    wait_time = 60 - (current_time - self.request_times[0])
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        
            if self.requests_per_hour:
                cutoff = current_time - 3600
                self.request_times = [t for t in self.request_times if t > cutoff]
                
                if len(self.request_times) >= self.requests_per_hour:
                    wait_time = 3600 - (current_time - self.request_times[0])
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        
        self.last_request_time = current_time


class APICache:
    """Simple file-based cache for API responses."""
    
    def __init__(self, cache_dir: str = ".api_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
    async def get(self, key: str) -> Optional[APIResponse]:
        """Get cached response."""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Check if cache is expired
            if datetime.now() > cached_data['expires_at']:
                os.remove(cache_file)
                return None
                
            return cached_data['response']
            
        except Exception:
            return None
            
    async def set(self, key: str, response: APIResponse, ttl: int = 3600):
        """Cache response."""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        cached_data = {
            'response': response,
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            logging.warning(f"Failed to cache response: {e}")


class APIIntegrationManager:
    """Manages multiple API integrations."""
    
    def __init__(self, config_file: str = None, cache_dir: str = ".api_cache"):
        self.clients: Dict[str, APIClient] = {}
        self.config = {}
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(f"{__name__}.APIIntegrationManager")
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
            
    def load_config(self, config_file: str):
        """Load API configurations from file."""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
                
            # Initialize clients from config
            for api_name, api_config in self.config.get('apis', {}).items():
                self.add_api_client(api_name, api_config)
                
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_file}: {e}")
            
    def add_api_client(self, name: str, config: Dict[str, Any]):
        """Add an API client."""
        try:
            endpoint = APIEndpoint(
                name=name,
                base_url=config['base_url'],
                auth_type=config.get('auth_type', 'none'),
                auth_config=config.get('auth_config', {}),
                headers=config.get('headers', {}),
                rate_limit=config.get('rate_limit'),
                timeout=config.get('timeout', 30),
                retry_config=config.get('retry_config', {})
            )
            
            # Create appropriate client type
            client_type = config.get('client_type', 'http')
            if client_type == 'http':
                self.clients[name] = HTTPAPIClient(endpoint, self.cache_dir)
            else:
                raise ValueError(f"Unsupported client type: {client_type}")
                
            self.logger.info(f"Added API client: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add API client {name}: {e}")
            
    async def call_api(self, api_name: str, method: str, path: str, **kwargs) -> APIResponse:
        """Make an API call."""
        if api_name not in self.clients:
            return APIResponse(
                success=False,
                data=None,
                status_code=404,
                error_message=f"API client '{api_name}' not found",
                source=api_name
            )
            
        client = self.clients[api_name]
        async with client:
            return await client.make_request(method, path, **kwargs)
            
    async def call_multiple_apis(self, calls: List[Dict[str, Any]]) -> List[APIResponse]:
        """Make multiple API calls concurrently."""
        tasks = []
        
        for call in calls:
            api_name = call['api_name']
            method = call['method']
            path = call['path']
            kwargs = call.get('kwargs', {})
            
            task = self.call_api(api_name, method, path, **kwargs)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(APIResponse(
                    success=False,
                    data=None,
                    status_code=500,
                    error_message=str(result),
                    source="unknown"
                ))
            else:
                processed_results.append(result)
                
        return processed_results
        
    def get_available_apis(self) -> List[str]:
        """Get list of available API clients."""
        return list(self.clients.keys())
        
    def get_api_info(self, api_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an API client."""
        if api_name not in self.clients:
            return None
            
        client = self.clients[api_name]
        return {
            'name': client.endpoint.name,
            'base_url': client.endpoint.base_url,
            'auth_type': client.endpoint.auth_type,
            'timeout': client.endpoint.timeout,
            'has_rate_limit': client.rate_limiter is not None,
            'has_cache': client.cache is not None
        }


# Pre-built API clients for common services
class NewsAPIClient(HTTPAPIClient):
    """Client for News API service."""
    
    def __init__(self, api_key: str, cache_dir: str = None):
        endpoint = APIEndpoint(
            name="newsapi",
            base_url="https://newsapi.org/v2/",
            auth_type="api_key",
            auth_config={'key_name': 'X-API-Key', 'api_key': api_key},
            rate_limit={'requests_per_hour': 1000}
        )
        super().__init__(endpoint, cache_dir)
        
    async def search_news(self, query: str, language: str = 'en', 
                         sort_by: str = 'relevancy', page_size: int = 20) -> APIResponse:
        """Search for news articles."""
        return await self.make_request(
            'GET', 'everything',
            params={
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': page_size
            }
        )
        
    async def get_top_headlines(self, country: str = 'us', category: str = None) -> APIResponse:
        """Get top headlines."""
        params = {'country': country}
        if category:
            params['category'] = category
            
        return await self.make_request('GET', 'top-headlines', params=params)


class WikipediaAPIClient(HTTPAPIClient):
    """Client for Wikipedia API."""
    
    def __init__(self, cache_dir: str = None):
        endpoint = APIEndpoint(
            name="wikipedia",
            base_url="https://en.wikipedia.org/api/rest_v1/",
            auth_type="none",
            headers={'User-Agent': 'ResearchAgent/1.0'},
            rate_limit={'requests_per_second': 1}
        )
        super().__init__(endpoint, cache_dir)
        
    async def search_pages(self, query: str, limit: int = 10) -> APIResponse:
        """Search Wikipedia pages."""
        return await self.make_request(
            'GET', 'page/search',
            params={'q': query, 'limit': limit}
        )
        
    async def get_page_summary(self, title: str) -> APIResponse:
        """Get page summary."""
        return await self.make_request('GET', f'page/summary/{title}')


class ArxivAPIClient(HTTPAPIClient):
    """Client for arXiv API."""
    
    def __init__(self, cache_dir: str = None):
        endpoint = APIEndpoint(
            name="arxiv",
            base_url="http://export.arxiv.org/api/",
            auth_type="none",
            rate_limit={'requests_per_second': 0.5}  # Be respectful
        )
        super().__init__(endpoint, cache_dir)
        
    async def search_papers(self, query: str, max_results: int = 10) -> APIResponse:
        """Search arXiv papers."""
        return await self.make_request(
            'GET', 'query',
            params={
                'search_query': f'all:{query}',
                'max_results': max_results,
                'sortBy': 'relevance'
            }
        )


# Example usage and factory functions
def create_research_api_manager(api_keys: Dict[str, str] = None) -> APIIntegrationManager:
    """Create a pre-configured API manager for research."""
    manager = APIIntegrationManager()
    
    # Add common research APIs if keys are provided
    if api_keys:
        if 'news_api_key' in api_keys:
            manager.clients['newsapi'] = NewsAPIClient(api_keys['news_api_key'])
            
        # Wikipedia doesn't need API key
        manager.clients['wikipedia'] = WikipediaAPIClient()
        
        # arXiv doesn't need API key
        manager.clients['arxiv'] = ArxivAPIClient()
        
    return manager


async def main():
    """Example usage of API integration system."""
    # Create API manager
    api_keys = {
        # 'news_api_key': 'your_news_api_key_here'
    }
    
    manager = create_research_api_manager(api_keys)
    
    # Example: Search Wikipedia
    if 'wikipedia' in manager.clients:
        response = await manager.call_api('wikipedia', 'GET', 'page/search', 
                                        params={'q': 'artificial intelligence', 'limit': 5})
        
        if response.success:
            print("Wikipedia Search Results:")
            for page in response.data.get('pages', []):
                print(f"- {page.get('title')}: {page.get('description', 'No description')}")
        else:
            print(f"Wikipedia search failed: {response.error_message}")
            
    # Example: Multiple API calls
    calls = [
        {
            'api_name': 'wikipedia',
            'method': 'GET',
            'path': 'page/search',
            'kwargs': {'params': {'q': 'machine learning', 'limit': 3}}
        },
        {
            'api_name': 'arxiv',
            'method': 'GET',
            'path': 'query',
            'kwargs': {'params': {'search_query': 'all:machine learning', 'max_results': 3}}
        }
    ]
    
    results = await manager.call_multiple_apis(calls)
    
    print(f"\nMultiple API calls completed:")
    for i, result in enumerate(results):
        print(f"Call {i+1}: {'Success' if result.success else 'Failed'}")
        if not result.success:
            print(f"  Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())