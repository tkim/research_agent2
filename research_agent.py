"""
Research Agent 2 - An intelligent assistant for conducting thorough research and analysis.

This module provides the core Research Agent 2 class with capabilities for web search,
information gathering, research methodology, and API integration.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class Source:
    """Represents a research source with metadata."""
    url: str
    title: str
    content: str
    timestamp: datetime
    relevance_score: float = 0.0
    source_type: str = "web"


@dataclass
class ResearchQuery:
    """Represents a research query with context and parameters."""
    question: str
    sub_questions: List[str]
    keywords: List[str]
    search_strategy: str
    context: Dict[str, Any]


@dataclass
class ResearchResult:
    """Represents the results of a research operation."""
    query: ResearchQuery
    sources: List[Source]
    synthesis: str
    confidence_level: float
    gaps_identified: List[str]
    timestamp: datetime


class ResearchAgent:
    """
    Core Research Agent class for conducting intelligent research and analysis.
    
    This agent provides capabilities for:
    - Web search and information gathering
    - Research methodology and query decomposition
    - Source verification and citation
    - API integration and tool usage
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Research Agent.
        
        Args:
            config: Configuration dictionary with API keys, search preferences, etc.
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        self.search_engines = self._initialize_search_engines()
        self.api_clients = {}
        self.research_history: List[ResearchResult] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the research agent."""
        logger = logging.getLogger("research_agent2")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_search_engines(self) -> Dict[str, Any]:
        """Initialize available search engines based on configuration."""
        engines = {}
        
        # Default search engine configurations
        if "google_api_key" in self.config:
            engines["google"] = {
                "api_key": self.config["google_api_key"],
                "search_engine_id": self.config.get("google_search_engine_id"),
                "base_url": "https://www.googleapis.com/customsearch/v1"
            }
            
        if "bing_api_key" in self.config:
            engines["bing"] = {
                "api_key": self.config["bing_api_key"],
                "base_url": "https://api.cognitive.microsoft.com/bing/v7.0/search"
            }
            
        return engines
    
    def decompose_query(self, question: str, context: Dict[str, Any] = None) -> ResearchQuery:
        """
        Break down a complex research question into manageable sub-questions.
        
        Args:
            question: The main research question
            context: Additional context for the research
            
        Returns:
            ResearchQuery object with decomposed questions and strategy
        """
        context = context or {}
        
        # Simple decomposition logic - can be enhanced with NLP
        sub_questions = []
        keywords = []
        
        # Extract key terms and generate sub-questions
        words = question.lower().split()
        
        # Identify question types and generate appropriate sub-questions
        if any(word in question.lower() for word in ["what", "define", "definition"]):
            sub_questions.append(f"What is the definition of key terms in: {question}")
            
        if any(word in question.lower() for word in ["how", "process", "method"]):
            sub_questions.append(f"What are the methods/processes related to: {question}")
            
        if any(word in question.lower() for word in ["why", "reason", "cause"]):
            sub_questions.append(f"What are the causes/reasons for: {question}")
            
        if any(word in question.lower() for word in ["when", "history", "timeline"]):
            sub_questions.append(f"What is the historical context of: {question}")
            
        # Extract keywords (simple approach - can be enhanced)
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for", "of", "as", "by"}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words][:10]
        
        # Determine search strategy
        strategy = "comprehensive"  # Default strategy
        if len(keywords) > 7:
            strategy = "focused"
        elif any(word in question.lower() for word in ["recent", "latest", "current", "2024", "2023"]):
            strategy = "current_events"
            
        return ResearchQuery(
            question=question,
            sub_questions=sub_questions,
            keywords=keywords,
            search_strategy=strategy,
            context=context
        )
    
    def search_web(self, query: str, num_results: int = 10, engine: str = "default") -> List[Source]:
        """
        Perform web search using available search engines.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            engine: Preferred search engine
            
        Returns:
            List of Source objects with search results
        """
        sources = []
        
        # For now, implement a basic web search simulation
        # In a real implementation, this would use actual search APIs
        self.logger.info(f"Searching for: {query}")
        
        # Placeholder results - replace with actual search implementation
        for i in range(min(num_results, 3)):
            source = Source(
                url=f"https://example-source-{i+1}.com/article",
                title=f"Article {i+1} about {query}",
                content=f"This is placeholder content for search result {i+1} about {query}. "
                       f"In a real implementation, this would contain actual web content.",
                timestamp=datetime.now(),
                relevance_score=0.8 - (i * 0.1),
                source_type="web"
            )
            sources.append(source)
            
        return sources
    
    def verify_sources(self, sources: List[Source]) -> List[Source]:
        """
        Verify and rank sources based on credibility and relevance.
        
        Args:
            sources: List of sources to verify
            
        Returns:
            List of verified and ranked sources
        """
        verified_sources = []
        
        for source in sources:
            # Basic URL validation
            try:
                parsed_url = urlparse(source.url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    continue
                    
                # Simple credibility scoring based on domain
                domain = parsed_url.netloc.lower()
                credibility_score = 0.5  # Base score
                
                # Boost score for known credible domains
                credible_domains = [
                    'edu', 'gov', 'org', 'wikipedia.org', 'scholar.google.com',
                    'pubmed.ncbi.nlm.nih.gov', 'arxiv.org'
                ]
                
                if any(credible in domain for credible in credible_domains):
                    credibility_score += 0.3
                    
                # Update relevance score
                source.relevance_score = min(1.0, source.relevance_score + credibility_score)
                verified_sources.append(source)
                
            except Exception as e:
                self.logger.warning(f"Failed to verify source {source.url}: {e}")
                continue
                
        # Sort by relevance score
        verified_sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return verified_sources
    
    def synthesize_information(self, sources: List[Source], query: ResearchQuery) -> Tuple[str, float, List[str]]:
        """
        Synthesize information from multiple sources into a coherent response.
        
        Args:
            sources: List of sources to synthesize
            query: Original research query
            
        Returns:
            Tuple of (synthesis text, confidence level, identified gaps)
        """
        if not sources:
            return "No relevant sources found.", 0.0, ["No sources available"]
            
        # Simple synthesis approach - can be enhanced with NLP
        synthesis_parts = []
        confidence_scores = []
        gaps = []
        
        # Group sources by relevance
        high_relevance = [s for s in sources if s.relevance_score > 0.7]
        medium_relevance = [s for s in sources if 0.4 <= s.relevance_score <= 0.7]
        
        if high_relevance:
            synthesis_parts.append("Based on highly relevant sources:")
            for source in high_relevance[:3]:  # Top 3 high-relevance sources
                synthesis_parts.append(f"- {source.title}: {source.content[:200]}...")
                confidence_scores.append(source.relevance_score)
                
        if medium_relevance:
            synthesis_parts.append("\nAdditional information from other sources:")
            for source in medium_relevance[:2]:  # Top 2 medium-relevance sources
                synthesis_parts.append(f"- {source.title}: {source.content[:150]}...")
                confidence_scores.append(source.relevance_score * 0.8)
                
        # Identify potential gaps
        if len(sources) < 3:
            gaps.append("Limited number of sources available")
            
        if not high_relevance:
            gaps.append("No highly relevant sources found")
            
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        synthesis = "\n".join(synthesis_parts) if synthesis_parts else "Unable to synthesize information from available sources."
        
        return synthesis, overall_confidence, gaps
    
    async def conduct_research(self, question: str, context: Dict[str, Any] = None) -> ResearchResult:
        """
        Conduct comprehensive research on a given question.
        
        Args:
            question: The research question
            context: Additional context for the research
            
        Returns:
            ResearchResult object with comprehensive findings
        """
        self.logger.info(f"Starting research on: {question}")
        
        # Step 1: Decompose the query
        query = self.decompose_query(question, context)
        
        # Step 2: Conduct searches for main question and sub-questions
        all_sources = []
        
        # Search for main question
        main_sources = self.search_web(query.question)
        all_sources.extend(main_sources)
        
        # Search for sub-questions
        for sub_question in query.sub_questions:
            sub_sources = self.search_web(sub_question, num_results=5)
            all_sources.extend(sub_sources)
            
        # Search using keywords
        keyword_query = " ".join(query.keywords[:5])  # Top 5 keywords
        keyword_sources = self.search_web(keyword_query, num_results=5)
        all_sources.extend(keyword_sources)
        
        # Step 3: Verify and rank sources
        verified_sources = self.verify_sources(all_sources)
        
        # Step 4: Synthesize information
        synthesis, confidence, gaps = self.synthesize_information(verified_sources, query)
        
        # Step 5: Create research result
        result = ResearchResult(
            query=query,
            sources=verified_sources,
            synthesis=synthesis,
            confidence_level=confidence,
            gaps_identified=gaps,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.research_history.append(result)
        
        self.logger.info(f"Research completed with confidence level: {confidence:.2f}")
        return result
    
    def add_api_client(self, name: str, client_config: Dict[str, Any]):
        """
        Add a new API client for extended functionality.
        
        Args:
            name: Name of the API client
            client_config: Configuration for the API client
        """
        self.api_clients[name] = client_config
        self.logger.info(f"Added API client: {name}")
    
    def get_research_history(self) -> List[ResearchResult]:
        """Get the history of research operations."""
        return self.research_history.copy()
    
    def export_results(self, result: ResearchResult, format: str = "json") -> str:
        """
        Export research results in various formats.
        
        Args:
            result: ResearchResult to export
            format: Export format (json, markdown, etc.)
            
        Returns:
            Formatted string representation
        """
        if format.lower() == "json":
            return json.dumps({
                "question": result.query.question,
                "sub_questions": result.query.sub_questions,
                "keywords": result.query.keywords,
                "synthesis": result.synthesis,
                "confidence_level": result.confidence_level,
                "gaps_identified": result.gaps_identified,
                "sources": [
                    {
                        "url": s.url,
                        "title": s.title,
                        "relevance_score": s.relevance_score,
                        "source_type": s.source_type
                    }
                    for s in result.sources
                ],
                "timestamp": result.timestamp.isoformat()
            }, indent=2)
            
        elif format.lower() == "markdown":
            md_lines = [
                f"# Research Report: {result.query.question}",
                f"\n**Generated:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Confidence Level:** {result.confidence_level:.2f}",
                "\n## Research Question",
                result.query.question,
                "\n## Sub-Questions",
            ]
            
            for i, sq in enumerate(result.query.sub_questions, 1):
                md_lines.append(f"{i}. {sq}")
                
            md_lines.extend([
                "\n## Keywords",
                ", ".join(result.query.keywords),
                "\n## Synthesis",
                result.synthesis,
                "\n## Sources"
            ])
            
            for i, source in enumerate(result.sources, 1):
                md_lines.append(f"{i}. [{source.title}]({source.url}) (Relevance: {source.relevance_score:.2f})")
                
            if result.gaps_identified:
                md_lines.extend(["\n## Identified Gaps"])
                for gap in result.gaps_identified:
                    md_lines.append(f"- {gap}")
                    
            return "\n".join(md_lines)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Example usage and testing functions
def main():
    """Example usage of the Research Agent 2."""
    # Initialize agent
    config = {
        # Add your API keys here
        # "google_api_key": "your_google_api_key",
        # "bing_api_key": "your_bing_api_key"
    }
    
    agent = ResearchAgent(config)
    
    # Example research question
    question = "What are the latest developments in artificial intelligence for medical diagnosis?"
    
    # Conduct research
    result = asyncio.run(agent.conduct_research(question))
    
    # Display results
    print("Research Results:")
    print("=" * 50)
    print(f"Question: {result.query.question}")
    print(f"Confidence: {result.confidence_level:.2f}")
    print(f"Sources found: {len(result.sources)}")
    print("\nSynthesis:")
    print(result.synthesis)
    
    # Export results
    json_export = agent.export_results(result, "json")
    markdown_export = agent.export_results(result, "markdown")
    
    print("\n" + "=" * 50)
    print("Export formats available: JSON, Markdown")


if __name__ == "__main__":
    main()