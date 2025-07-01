"""
Citation Manager for Research Agent

This module provides comprehensive citation management, source tracking,
and bibliography generation capabilities.
"""

import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from urllib.parse import urlparse
import logging


class CitationStyle(Enum):
    """Supported citation styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    VANCOUVER = "vancouver"


class SourceType(Enum):
    """Types of sources."""
    WEBSITE = "website"
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    NEWS_ARTICLE = "news_article"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    REPORT = "report"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    PODCAST = "podcast"
    VIDEO = "video"
    DATASET = "dataset"
    SOFTWARE = "software"
    UNKNOWN = "unknown"


@dataclass
class Author:
    """Represents an author with various name formats."""
    first_name: str = ""
    last_name: str = ""
    middle_name: str = ""
    full_name: str = ""
    
    def __post_init__(self):
        if not self.full_name and (self.first_name or self.last_name):
            name_parts = [self.first_name, self.middle_name, self.last_name]
            self.full_name = " ".join(part for part in name_parts if part)
        elif self.full_name and not (self.first_name or self.last_name):
            self._parse_full_name()
            
    def _parse_full_name(self):
        """Parse full name into components."""
        parts = self.full_name.strip().split()
        if len(parts) >= 2:
            self.first_name = parts[0]
            self.last_name = parts[-1]
            if len(parts) > 2:
                self.middle_name = " ".join(parts[1:-1])
                
    def get_formatted_name(self, format_type: str = "full") -> str:
        """Get formatted name in various styles."""
        if format_type == "last_first":
            return f"{self.last_name}, {self.first_name}"
        elif format_type == "initials":
            first_initial = self.first_name[0] + "." if self.first_name else ""
            middle_initial = self.middle_name[0] + "." if self.middle_name else ""
            return f"{first_initial} {middle_initial} {self.last_name}".strip()
        else:
            return self.full_name


@dataclass
class Citation:
    """Represents a complete citation with all metadata."""
    id: str
    source_type: SourceType
    title: str
    authors: List[Author] = field(default_factory=list)
    url: str = ""
    publication_date: Optional[datetime] = None
    access_date: datetime = field(default_factory=datetime.now)
    publisher: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    isbn: str = ""
    publication_place: str = ""
    edition: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    credibility_score: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
            
    def _generate_id(self) -> str:
        """Generate unique ID for the citation."""
        content = f"{self.title}{self.url}{len(self.authors)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class SourceDetector:
    """Detects and classifies source types from URLs and metadata."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SourceDetector")
        self._load_patterns()
        
    def _load_patterns(self):
        """Load patterns for source type detection."""
        self.domain_patterns = {
            SourceType.JOURNAL_ARTICLE: [
                'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com', 'jstor.org',
                'nature.com', 'science.org', 'cell.com', 'nejm.org',
                'bmj.com', 'thelancet.com', 'springer.com', 'wiley.com',
                'elsevier.com', 'ieee.org', 'acm.org'
            ],
            SourceType.NEWS_ARTICLE: [
                'cnn.com', 'bbc.com', 'reuters.com', 'nytimes.com',
                'washingtonpost.com', 'wsj.com', 'guardian.com',
                'npr.org', 'abc.com', 'cbs.com', 'fox.com'
            ],
            SourceType.BLOG_POST: [
                'medium.com', 'wordpress.com', 'blogger.com', 'tumblr.com',
                'substack.com'
            ],
            SourceType.SOCIAL_MEDIA: [
                'twitter.com', 'facebook.com', 'linkedin.com', 'instagram.com',
                'youtube.com', 'tiktok.com', 'reddit.com'
            ],
            SourceType.SOFTWARE: [
                'github.com', 'gitlab.com', 'sourceforge.net', 'pypi.org',
                'npmjs.com', 'cran.r-project.org'
            ]
        }
        
        self.url_patterns = {
            SourceType.JOURNAL_ARTICLE: [
                r'/doi/', r'/pubmed/', r'/article/', r'/paper/', r'/abstract/'
            ],
            SourceType.NEWS_ARTICLE: [
                r'/news/', r'/article/', r'/story/', r'/\d{4}/\d{2}/\d{2}/'
            ],
            SourceType.CONFERENCE_PAPER: [
                r'/conference/', r'/proceedings/', r'/symposium/'
            ],
            SourceType.THESIS: [
                r'/thesis/', r'/dissertation/', r'/phd/', r'/masters/'
            ],
            SourceType.REPORT: [
                r'/report/', r'/whitepaper/', r'/technical-report/'
            ]
        }
        
    def detect_source_type(self, url: str, title: str = "", 
                          metadata: Dict[str, Any] = None) -> SourceType:
        """
        Detect source type from URL and metadata.
        
        Args:
            url: Source URL
            title: Source title
            metadata: Additional metadata
            
        Returns:
            Detected SourceType
        """
        metadata = metadata or {}
        
        if not url:
            return SourceType.UNKNOWN
            
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            path = parsed_url.path.lower()
            
            # Check domain patterns
            for source_type, domains in self.domain_patterns.items():
                if any(domain_pattern in domain for domain_pattern in domains):
                    return source_type
                    
            # Check URL path patterns
            for source_type, patterns in self.url_patterns.items():
                if any(re.search(pattern, path) for pattern in patterns):
                    return source_type
                    
            # Check file extensions
            if url.lower().endswith('.pdf'):
                if any(keyword in title.lower() for keyword in ['paper', 'article', 'journal']):
                    return SourceType.JOURNAL_ARTICLE
                else:
                    return SourceType.REPORT
                    
            # Check metadata
            if metadata.get('type') == 'journal-article':
                return SourceType.JOURNAL_ARTICLE
            elif metadata.get('type') == 'news':
                return SourceType.NEWS_ARTICLE
                
            # Default classification based on domain type
            if domain.endswith('.edu'):
                return SourceType.JOURNAL_ARTICLE
            elif domain.endswith('.gov'):
                return SourceType.REPORT
            elif domain.endswith('.org'):
                return SourceType.WEBSITE
            else:
                return SourceType.WEBSITE
                
        except Exception as e:
            self.logger.warning(f"Error detecting source type for {url}: {e}")
            return SourceType.UNKNOWN


class MetadataExtractor:
    """Extracts metadata from web content and URLs."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MetadataExtractor")
        
    def extract_from_html(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML content."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            metadata = {}
            
            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text(strip=True)
                
            # Meta tags
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name', '').lower()
                property_name = tag.get('property', '').lower()
                content = tag.get('content', '')
                
                if not content:
                    continue
                    
                # Standard meta tags
                if name in ['author', 'description', 'keywords', 'publisher']:
                    metadata[name] = content
                elif name == 'citation_author':
                    if 'authors' not in metadata:
                        metadata['authors'] = []
                    metadata['authors'].append(content)
                elif name.startswith('citation_'):
                    key = name.replace('citation_', '')
                    metadata[key] = content
                    
                # Open Graph tags
                elif property_name.startswith('og:'):
                    key = property_name.replace('og:', 'og_')
                    metadata[key] = content
                    
                # Twitter Card tags
                elif name.startswith('twitter:'):
                    key = name.replace('twitter:', 'twitter_')
                    metadata[key] = content
                    
            # JSON-LD structured data
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    json_data = json.loads(script.string)
                    metadata['json_ld'] = json_data
                    
                    # Extract common fields
                    if '@type' in json_data:
                        metadata['structured_type'] = json_data['@type']
                    if 'author' in json_data:
                        metadata['structured_author'] = json_data['author']
                    if 'datePublished' in json_data:
                        metadata['date_published'] = json_data['datePublished']
                        
                except json.JSONDecodeError:
                    continue
                    
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from HTML: {e}")
            return {}
            
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """Extract metadata from URL structure."""
        try:
            parsed_url = urlparse(url)
            metadata = {
                'domain': parsed_url.netloc,
                'path': parsed_url.path,
                'scheme': parsed_url.scheme
            }
            
            # Extract date from URL if present
            date_pattern = r'/(\d{4})/(\d{1,2})/(\d{1,2})/'
            date_match = re.search(date_pattern, url)
            if date_match:
                year, month, day = date_match.groups()
                try:
                    metadata['url_date'] = datetime(int(year), int(month), int(day))
                except ValueError:
                    pass
                    
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from URL {url}: {e}")
            return {}


class CitationFormatter:
    """Formats citations in various academic styles."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CitationFormatter")
        
    def format_citation(self, citation: Citation, style: CitationStyle) -> str:
        """
        Format citation in specified style.
        
        Args:
            citation: Citation to format
            style: Citation style to use
            
        Returns:
            Formatted citation string
        """
        formatters = {
            CitationStyle.APA: self._format_apa,
            CitationStyle.MLA: self._format_mla,
            CitationStyle.CHICAGO: self._format_chicago,
            CitationStyle.IEEE: self._format_ieee,
            CitationStyle.HARVARD: self._format_harvard,
            CitationStyle.VANCOUVER: self._format_vancouver
        }
        
        formatter = formatters.get(style)
        if not formatter:
            raise ValueError(f"Unsupported citation style: {style}")
            
        return formatter(citation)
        
    def _format_apa(self, citation: Citation) -> str:
        """Format citation in APA style."""
        parts = []
        
        # Authors
        if citation.authors:
            if len(citation.authors) == 1:
                author_str = citation.authors[0].get_formatted_name("last_first")
            elif len(citation.authors) <= 7:
                author_names = [author.get_formatted_name("last_first") for author in citation.authors[:-1]]
                author_str = ", ".join(author_names) + f", & {citation.authors[-1].get_formatted_name('last_first')}"
            else:
                # More than 7 authors
                author_names = [author.get_formatted_name("last_first") for author in citation.authors[:6]]
                author_str = ", ".join(author_names) + f", ... {citation.authors[-1].get_formatted_name('last_first')}"
            parts.append(author_str)
            
        # Date
        if citation.publication_date:
            parts.append(f"({citation.publication_date.year})")
            
        # Title
        if citation.source_type in [SourceType.JOURNAL_ARTICLE, SourceType.NEWS_ARTICLE]:
            parts.append(citation.title)
        else:
            parts.append(f"*{citation.title}*")
            
        # Journal/Publisher info
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", {citation.volume}"
                if citation.issue:
                    journal_part += f"({citation.issue})"
            if citation.pages:
                journal_part += f", {citation.pages}"
            parts.append(journal_part)
        elif citation.publisher:
            parts.append(citation.publisher)
            
        # URL and access date
        if citation.url:
            url_part = citation.url
            if citation.access_date:
                url_part += f" (accessed {citation.access_date.strftime('%B %d, %Y')})"
            parts.append(url_part)
            
        return ". ".join(parts) + "."
        
    def _format_mla(self, citation: Citation) -> str:
        """Format citation in MLA style."""
        parts = []
        
        # Author
        if citation.authors:
            if len(citation.authors) == 1:
                parts.append(citation.authors[0].get_formatted_name("last_first"))
            else:
                first_author = citation.authors[0].get_formatted_name("last_first")
                parts.append(f"{first_author}, et al")
                
        # Title
        if citation.source_type in [SourceType.JOURNAL_ARTICLE, SourceType.NEWS_ARTICLE]:
            parts.append(f'"{citation.title}"')
        else:
            parts.append(f"*{citation.title}*")
            
        # Container (journal, website, etc.)
        if citation.journal:
            container_part = f"*{citation.journal}*"
            if citation.volume:
                container_part += f", vol. {citation.volume}"
            if citation.issue:
                container_part += f", no. {citation.issue}"
            if citation.publication_date:
                container_part += f", {citation.publication_date.strftime('%d %b %Y')}"
            if citation.pages:
                container_part += f", pp. {citation.pages}"
            parts.append(container_part)
            
        # URL and access date
        if citation.url:
            parts.append(citation.url)
            if citation.access_date:
                parts.append(f"Accessed {citation.access_date.strftime('%d %b %Y')}")
                
        return ". ".join(parts) + "."
        
    def _format_chicago(self, citation: Citation) -> str:
        """Format citation in Chicago style (Notes-Bibliography)."""
        # Simplified Chicago format
        return self._format_apa(citation)  # Placeholder - implement full Chicago format
        
    def _format_ieee(self, citation: Citation) -> str:
        """Format citation in IEEE style."""
        parts = []
        
        # Authors
        if citation.authors:
            if len(citation.authors) <= 3:
                author_names = [author.get_formatted_name("initials") for author in citation.authors]
                parts.append(" and ".join(author_names))
            else:
                first_author = citation.authors[0].get_formatted_name("initials")
                parts.append(f"{first_author} et al.")
                
        # Title
        parts.append(f'"{citation.title}"')
        
        # Journal
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", vol. {citation.volume}"
            if citation.issue:
                journal_part += f", no. {citation.issue}"
            if citation.pages:
                journal_part += f", pp. {citation.pages}"
            if citation.publication_date:
                journal_part += f", {citation.publication_date.strftime('%b. %Y')}"
            parts.append(journal_part)
            
        return ", ".join(parts) + "."
        
    def _format_harvard(self, citation: Citation) -> str:
        """Format citation in Harvard style."""
        # Similar to APA but with some differences
        return self._format_apa(citation)  # Placeholder - implement full Harvard format
        
    def _format_vancouver(self, citation: Citation) -> str:
        """Format citation in Vancouver style."""
        parts = []
        
        # Authors
        if citation.authors:
            if len(citation.authors) <= 6:
                author_names = [f"{author.last_name} {author.first_name[0] if author.first_name else ''}" 
                               for author in citation.authors]
                parts.append(", ".join(author_names))
            else:
                first_authors = [f"{author.last_name} {author.first_name[0] if author.first_name else ''}" 
                                for author in citation.authors[:3]]
                parts.append(", ".join(first_authors) + ", et al")
                
        # Title
        parts.append(citation.title)
        
        # Journal
        if citation.journal:
            journal_part = citation.journal
            if citation.publication_date:
                journal_part += f". {citation.publication_date.year}"
            if citation.volume:
                journal_part += f";{citation.volume}"
                if citation.issue:
                    journal_part += f"({citation.issue})"
            if citation.pages:
                journal_part += f":{citation.pages}"
            parts.append(journal_part)
            
        return ". ".join(parts) + "."


class CitationManager:
    """Main citation management system."""
    
    def __init__(self, default_style: CitationStyle = CitationStyle.APA):
        self.citations: Dict[str, Citation] = {}
        self.default_style = default_style
        self.source_detector = SourceDetector()
        self.metadata_extractor = MetadataExtractor()
        self.formatter = CitationFormatter()
        self.logger = logging.getLogger(f"{__name__}.CitationManager")
        
    def add_source(self, url: str, title: str = "", content: str = "",
                  metadata: Dict[str, Any] = None, 
                  manual_metadata: Dict[str, Any] = None) -> Citation:
        """
        Add a source and create citation.
        
        Args:
            url: Source URL
            title: Source title
            content: Source content (for metadata extraction)
            metadata: Existing metadata
            manual_metadata: Manually provided metadata
            
        Returns:
            Created Citation object
        """
        metadata = metadata or {}
        manual_metadata = manual_metadata or {}
        
        # Detect source type
        source_type = self.source_detector.detect_source_type(url, title, metadata)
        
        # Extract metadata from URL
        url_metadata = self.metadata_extractor.extract_from_url(url)
        
        # Extract metadata from content if available
        if content:
            content_metadata = self.metadata_extractor.extract_from_html(content, url)
            metadata.update(content_metadata)
            
        metadata.update(url_metadata)
        metadata.update(manual_metadata)
        
        # Parse authors
        authors = self._parse_authors(metadata)
        
        # Parse publication date
        pub_date = self._parse_date(metadata)
        
        # Create citation
        citation = Citation(
            id="",  # Will be auto-generated
            source_type=source_type,
            title=title or metadata.get('title', 'Untitled'),
            authors=authors,
            url=url,
            publication_date=pub_date,
            access_date=datetime.now(),
            publisher=metadata.get('publisher', ''),
            journal=metadata.get('journal', metadata.get('publication_title', '')),
            volume=metadata.get('volume', ''),
            issue=metadata.get('issue', ''),
            pages=metadata.get('pages', metadata.get('page', '')),
            doi=metadata.get('doi', ''),
            isbn=metadata.get('isbn', ''),
            abstract=metadata.get('description', metadata.get('abstract', '')),
            keywords=metadata.get('keywords', '').split(',') if metadata.get('keywords') else [],
            metadata=metadata
        )
        
        self.citations[citation.id] = citation
        self.logger.info(f"Added citation: {citation.id} - {citation.title}")
        
        return citation
        
    def _parse_authors(self, metadata: Dict[str, Any]) -> List[Author]:
        """Parse authors from metadata."""
        authors = []
        
        # Try different author fields
        author_fields = ['authors', 'author', 'citation_author', 'structured_author']
        
        for field in author_fields:
            if field in metadata:
                author_data = metadata[field]
                
                if isinstance(author_data, list):
                    for author_info in author_data:
                        if isinstance(author_info, str):
                            authors.append(Author(full_name=author_info))
                        elif isinstance(author_info, dict):
                            authors.append(Author(
                                first_name=author_info.get('given', ''),
                                last_name=author_info.get('family', ''),
                                full_name=author_info.get('name', '')
                            ))
                elif isinstance(author_data, str):
                    # Split multiple authors
                    author_names = re.split(r'[,;]|\sand\s|\&', author_data)
                    for name in author_names:
                        if name.strip():
                            authors.append(Author(full_name=name.strip()))
                            
                if authors:  # Stop at first successful parse
                    break
                    
        return authors
        
    def _parse_date(self, metadata: Dict[str, Any]) -> Optional[datetime]:
        """Parse publication date from metadata."""
        date_fields = ['date_published', 'publication_date', 'date', 'url_date']
        
        for field in date_fields:
            if field in metadata:
                date_value = metadata[field]
                
                if isinstance(date_value, datetime):
                    return date_value
                elif isinstance(date_value, str):
                    # Try to parse date string
                    try:
                        # Try ISO format first
                        return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    except ValueError:
                        # Try other common formats
                        date_formats = [
                            '%Y-%m-%d',
                            '%Y/%m/%d',
                            '%d/%m/%Y',
                            '%m/%d/%Y',
                            '%B %d, %Y',
                            '%b %d, %Y',
                            '%Y'
                        ]
                        
                        for fmt in date_formats:
                            try:
                                return datetime.strptime(date_value, fmt)
                            except ValueError:
                                continue
                                
        return None
        
    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Get citation by ID."""
        return self.citations.get(citation_id)
        
    def get_all_citations(self) -> List[Citation]:
        """Get all citations."""
        return list(self.citations.values())
        
    def format_citation(self, citation_id: str, 
                       style: CitationStyle = None) -> str:
        """Format a citation in specified style."""
        citation = self.get_citation(citation_id)
        if not citation:
            return f"Citation not found: {citation_id}"
            
        style = style or self.default_style
        return self.formatter.format_citation(citation, style)
        
    def generate_bibliography(self, style: CitationStyle = None,
                            sort_by: str = "author") -> str:
        """
        Generate bibliography from all citations.
        
        Args:
            style: Citation style to use
            sort_by: Sort method ("author", "date", "title")
            
        Returns:
            Formatted bibliography string
        """
        style = style or self.default_style
        citations = self.get_all_citations()
        
        if not citations:
            return "No citations available."
            
        # Sort citations
        if sort_by == "author":
            citations.sort(key=lambda c: c.authors[0].last_name if c.authors else c.title)
        elif sort_by == "date":
            citations.sort(key=lambda c: c.publication_date or datetime.min)
        elif sort_by == "title":
            citations.sort(key=lambda c: c.title.lower())
            
        # Format bibliography
        bibliography_lines = [f"# Bibliography ({style.value.upper()})"]
        bibliography_lines.append("")
        
        for citation in citations:
            formatted = self.formatter.format_citation(citation, style)
            bibliography_lines.append(formatted)
            bibliography_lines.append("")
            
        return "\n".join(bibliography_lines)
        
    def export_citations(self, format_type: str = "json") -> str:
        """Export citations in various formats."""
        if format_type == "json":
            citations_data = []
            for citation in self.citations.values():
                citation_dict = asdict(citation)
                # Convert datetime objects to strings
                if citation_dict['publication_date']:
                    citation_dict['publication_date'] = citation_dict['publication_date'].isoformat()
                citation_dict['access_date'] = citation_dict['access_date'].isoformat()
                citations_data.append(citation_dict)
                
            return json.dumps(citations_data, indent=2)
            
        elif format_type == "bibtex":
            bibtex_entries = []
            for citation in self.citations.values():
                entry = self._to_bibtex(citation)
                bibtex_entries.append(entry)
            return "\n\n".join(bibtex_entries)
            
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
            
    def _to_bibtex(self, citation: Citation) -> str:
        """Convert citation to BibTeX format."""
        # Determine entry type
        type_mapping = {
            SourceType.JOURNAL_ARTICLE: "article",
            SourceType.BOOK: "book",
            SourceType.CONFERENCE_PAPER: "inproceedings",
            SourceType.THESIS: "phdthesis",
            SourceType.REPORT: "techreport",
            SourceType.WEBSITE: "misc"
        }
        
        entry_type = type_mapping.get(citation.source_type, "misc")
        
        # Generate BibTeX key
        bibtex_key = citation.id
        
        lines = [f"@{entry_type}{{{bibtex_key},"]
        
        # Add fields
        if citation.title:
            lines.append(f'  title = "{citation.title}",')
            
        if citation.authors:
            authors_str = " and ".join(author.full_name for author in citation.authors)
            lines.append(f'  author = "{authors_str}",')
            
        if citation.journal:
            lines.append(f'  journal = "{citation.journal}",')
            
        if citation.publisher:
            lines.append(f'  publisher = "{citation.publisher}",')
            
        if citation.publication_date:
            lines.append(f'  year = "{citation.publication_date.year}",')
            
        if citation.volume:
            lines.append(f'  volume = "{citation.volume}",')
            
        if citation.issue:
            lines.append(f'  number = "{citation.issue}",')
            
        if citation.pages:
            lines.append(f'  pages = "{citation.pages}",')
            
        if citation.doi:
            lines.append(f'  doi = "{citation.doi}",')
            
        if citation.url:
            lines.append(f'  url = "{citation.url}",')
            
        lines.append("}")
        
        return "\n".join(lines)
        
    def search_citations(self, query: str) -> List[Citation]:
        """Search citations by query."""
        query_lower = query.lower()
        matching_citations = []
        
        for citation in self.citations.values():
            # Search in title, authors, and abstract
            if (query_lower in citation.title.lower() or
                any(query_lower in author.full_name.lower() for author in citation.authors) or
                query_lower in citation.abstract.lower()):
                matching_citations.append(citation)
                
        return matching_citations


# Example usage
async def main():
    """Example usage of citation manager."""
    # Create citation manager
    manager = CitationManager(CitationStyle.APA)
    
    # Add some example sources
    citation1 = manager.add_source(
        url="https://www.nature.com/articles/s41586-021-03819-2",
        title="Highly accurate protein structure prediction with AlphaFold",
        manual_metadata={
            'authors': [
                {'given': 'John', 'family': 'Jumper'},
                {'given': 'Richard', 'family': 'Evans'}
            ],
            'journal': 'Nature',
            'volume': '596',
            'pages': '583-589',
            'date_published': '2021-08-15'
        }
    )
    
    citation2 = manager.add_source(
        url="https://arxiv.org/abs/2103.00020",
        title="Language Models are Few-Shot Learners",
        manual_metadata={
            'authors': [
                {'given': 'Tom', 'family': 'Brown'},
                {'given': 'Benjamin', 'family': 'Mann'}
            ],
            'date_published': '2021-03-01'
        }
    )
    
    # Format individual citations
    print("APA Citation:")
    print(manager.format_citation(citation1.id, CitationStyle.APA))
    print()
    
    print("IEEE Citation:")
    print(manager.format_citation(citation1.id, CitationStyle.IEEE))
    print()
    
    # Generate bibliography
    print("Bibliography:")
    print(manager.generate_bibliography(CitationStyle.APA))
    
    # Export citations
    print("JSON Export:")
    print(manager.export_citations("json"))


if __name__ == "__main__":
    asyncio.run(main())