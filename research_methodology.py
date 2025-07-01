"""
Research Methodology Module for Research Agent

This module provides advanced research methodologies, query analysis,
and information synthesis capabilities.
"""

import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import Counter
import json


class ResearchType(Enum):
    """Types of research approaches."""
    EXPLORATORY = "exploratory"
    DESCRIPTIVE = "descriptive"
    EXPLANATORY = "explanatory"
    COMPARATIVE = "comparative"
    HISTORICAL = "historical"
    CURRENT_EVENTS = "current_events"
    TECHNICAL = "technical"
    ACADEMIC = "academic"


class ConfidenceLevel(Enum):
    """Confidence levels for research findings."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class ResearchContext:
    """Context for research operations."""
    domain: str = ""
    time_period: Optional[Tuple[datetime, datetime]] = None
    geographic_scope: str = ""
    target_audience: str = ""
    purpose: str = ""
    constraints: List[str] = field(default_factory=list)
    priority_keywords: List[str] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Analysis of a research query."""
    original_query: str
    research_type: ResearchType
    intent: str
    key_concepts: List[str]
    entities: List[str]
    temporal_indicators: List[str]
    complexity_score: float
    ambiguity_score: float
    search_strategies: List[str]
    expected_source_types: List[str]


@dataclass
class SourceCredibility:
    """Credibility assessment for a source."""
    domain_score: float
    content_quality_score: float
    recency_score: float
    authority_score: float
    bias_indicators: List[str]
    credibility_factors: Dict[str, float]
    overall_score: float
    confidence: float


@dataclass
class InformationGap:
    """Represents a gap in available information."""
    gap_type: str
    description: str
    severity: str
    potential_sources: List[str]
    search_suggestions: List[str]


class QueryAnalyzer:
    """Analyzes research queries to determine optimal search strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QueryAnalyzer")
        self._load_patterns()
        
    def _load_patterns(self):
        """Load patterns for query analysis."""
        # Question type patterns
        self.question_patterns = {
            'what': r'\b(what|define|definition|meaning)\b',
            'how': r'\b(how|process|method|procedure|steps)\b',
            'why': r'\b(why|reason|cause|because|explanation)\b',
            'when': r'\b(when|time|date|timeline|history)\b',
            'where': r'\b(where|location|place|geography)\b',
            'who': r'\b(who|person|people|organization|company)\b',
            'which': r'\b(which|compare|comparison|versus|vs)\b'
        }
        
        # Research type indicators
        self.research_type_patterns = {
            ResearchType.EXPLORATORY: [
                'explore', 'investigate', 'overview', 'survey', 'landscape'
            ],
            ResearchType.DESCRIPTIVE: [
                'describe', 'characteristics', 'features', 'properties', 'attributes'
            ],
            ResearchType.EXPLANATORY: [
                'explain', 'why', 'cause', 'reason', 'mechanism', 'theory'
            ],
            ResearchType.COMPARATIVE: [
                'compare', 'versus', 'vs', 'difference', 'similarity', 'contrast'
            ],
            ResearchType.HISTORICAL: [
                'history', 'historical', 'evolution', 'development', 'timeline', 'past'
            ],
            ResearchType.CURRENT_EVENTS: [
                'latest', 'recent', 'current', 'news', 'update', '2024', '2023'
            ],
            ResearchType.TECHNICAL: [
                'technical', 'implementation', 'algorithm', 'specification', 'protocol'
            ],
            ResearchType.ACADEMIC: [
                'research', 'study', 'paper', 'journal', 'academic', 'peer-reviewed'
            ]
        }
        
        # Temporal indicators
        self.temporal_patterns = {
            'current': r'\b(current|present|now|today|latest|recent)\b',
            'past': r'\b(history|historical|past|previous|former|ago)\b',
            'future': r'\b(future|upcoming|planned|expected|forecast)\b',
            'specific_year': r'\b(19|20)\d{2}\b',
            'time_range': r'\b(since|from|between|during)\b'
        }
        
        # Entity patterns (simplified)
        self.entity_patterns = {
            'organization': r'\b[A-Z][a-z]+ (?:Inc|Corp|Company|Organization|Institute|University|Agency)\b',
            'location': r'\b[A-Z][a-z]+ (?:City|State|Country|Region|Province)\b',
            'technology': r'\b[A-Z]{2,}|(?:[A-Z][a-z]+){2,}\b'
        }
        
    def analyze_query(self, query: str, context: ResearchContext = None) -> QueryAnalysis:
        """
        Analyze a research query to determine optimal search approach.
        
        Args:
            query: The research query to analyze
            context: Optional research context
            
        Returns:
            QueryAnalysis object with detailed analysis
        """
        context = context or ResearchContext()
        query_lower = query.lower()
        
        # Determine research type
        research_type = self._determine_research_type(query_lower)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Find temporal indicators
        temporal_indicators = self._find_temporal_indicators(query_lower)
        
        # Calculate complexity and ambiguity
        complexity_score = self._calculate_complexity(query, key_concepts)
        ambiguity_score = self._calculate_ambiguity(query, key_concepts)
        
        # Determine search strategies
        search_strategies = self._determine_search_strategies(
            research_type, complexity_score, temporal_indicators
        )
        
        # Determine expected source types
        expected_source_types = self._determine_source_types(research_type, query_lower)
        
        # Determine intent
        intent = self._determine_intent(query_lower, research_type)
        
        return QueryAnalysis(
            original_query=query,
            research_type=research_type,
            intent=intent,
            key_concepts=key_concepts,
            entities=entities,
            temporal_indicators=temporal_indicators,
            complexity_score=complexity_score,
            ambiguity_score=ambiguity_score,
            search_strategies=search_strategies,
            expected_source_types=expected_source_types
        )
        
    def _determine_research_type(self, query_lower: str) -> ResearchType:
        """Determine the type of research based on query content."""
        type_scores = {}
        
        for research_type, keywords in self.research_type_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                type_scores[research_type] = score
                
        if type_scores:
            return max(type_scores, key=type_scores.get)
        else:
            return ResearchType.EXPLORATORY  # Default
            
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query."""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'about',
            'what', 'how', 'why', 'when', 'where', 'who', 'which'
        }
        
        # Simple tokenization and filtering
        words = re.findall(r'\b\w+\b', query.lower())
        concepts = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Find multi-word concepts (simple approach)
        multi_word_concepts = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', query)
        concepts.extend([concept.lower() for concept in multi_word_concepts])
        
        return list(set(concepts))
        
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query)
            entities.extend(matches)
            
        # Find capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized_words)
        
        return list(set(entities))
        
    def _find_temporal_indicators(self, query_lower: str) -> List[str]:
        """Find temporal indicators in the query."""
        indicators = []
        
        for temporal_type, pattern in self.temporal_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                indicators.extend([f"{temporal_type}: {match}" for match in matches])
                
        return indicators
        
    def _calculate_complexity(self, query: str, key_concepts: List[str]) -> float:
        """Calculate query complexity score."""
        # Factors affecting complexity
        word_count = len(query.split())
        concept_count = len(key_concepts)
        question_count = len(re.findall(r'[?]', query))
        conjunction_count = len(re.findall(r'\b(and|or|but)\b', query.lower()))
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (
            (word_count / 20) * 0.3 +
            (concept_count / 10) * 0.4 +
            (question_count / 3) * 0.2 +
            (conjunction_count / 3) * 0.1
        ))
        
        return complexity
        
    def _calculate_ambiguity(self, query: str, key_concepts: List[str]) -> float:
        """Calculate query ambiguity score."""
        # Factors affecting ambiguity
        vague_terms = ['thing', 'stuff', 'something', 'anything', 'some', 'any']
        vague_count = sum(1 for term in vague_terms if term in query.lower())
        
        # Multiple meanings for concepts (simplified)
        ambiguous_concepts = ['bank', 'bank', 'cell', 'court', 'interest', 'rock']
        ambiguous_count = sum(1 for concept in key_concepts if concept in ambiguous_concepts)
        
        # Pronoun usage
        pronoun_count = len(re.findall(r'\b(it|this|that|they|them)\b', query.lower()))
        
        # Normalize to 0-1 scale
        ambiguity = min(1.0, (
            (vague_count / 3) * 0.4 +
            (ambiguous_count / 3) * 0.4 +
            (pronoun_count / 3) * 0.2
        ))
        
        return ambiguity
        
    def _determine_search_strategies(self, research_type: ResearchType, 
                                   complexity: float, temporal_indicators: List[str]) -> List[str]:
        """Determine optimal search strategies."""
        strategies = []
        
        # Base strategies by research type
        type_strategies = {
            ResearchType.EXPLORATORY: ['broad_search', 'multi_source', 'survey_approach'],
            ResearchType.DESCRIPTIVE: ['detailed_search', 'factual_sources', 'structured_approach'],
            ResearchType.EXPLANATORY: ['causal_search', 'expert_sources', 'theory_based'],
            ResearchType.COMPARATIVE: ['comparative_search', 'multiple_perspectives', 'side_by_side'],
            ResearchType.HISTORICAL: ['chronological_search', 'historical_sources', 'timeline_based'],
            ResearchType.CURRENT_EVENTS: ['recent_search', 'news_sources', 'real_time'],
            ResearchType.TECHNICAL: ['technical_search', 'documentation', 'specification_based'],
            ResearchType.ACADEMIC: ['scholarly_search', 'peer_reviewed', 'citation_based']
        }
        
        strategies.extend(type_strategies.get(research_type, ['general_search']))
        
        # Complexity-based strategies
        if complexity > 0.7:
            strategies.append('decomposition_approach')
        if complexity < 0.3:
            strategies.append('direct_search')
            
        # Temporal strategies
        if any('current' in indicator for indicator in temporal_indicators):
            strategies.append('time_filtered_recent')
        if any('past' in indicator for indicator in temporal_indicators):
            strategies.append('historical_focus')
            
        return list(set(strategies))
        
    def _determine_source_types(self, research_type: ResearchType, query_lower: str) -> List[str]:
        """Determine expected source types."""
        source_types = []
        
        # Research type based sources
        type_sources = {
            ResearchType.ACADEMIC: ['academic_journals', 'research_papers', 'university_sites'],
            ResearchType.TECHNICAL: ['documentation', 'technical_specs', 'developer_sites'],
            ResearchType.CURRENT_EVENTS: ['news_sites', 'press_releases', 'social_media'],
            ResearchType.HISTORICAL: ['archives', 'historical_sites', 'timeline_resources'],
            ResearchType.COMPARATIVE: ['comparison_sites', 'review_sites', 'analytical_reports']
        }
        
        source_types.extend(type_sources.get(research_type, ['general_web']))
        
        # Query-based source indicators
        if any(word in query_lower for word in ['study', 'research', 'paper']):
            source_types.append('academic_sources')
        if any(word in query_lower for word in ['news', 'breaking', 'latest']):
            source_types.append('news_sources')
        if any(word in query_lower for word in ['official', 'government', 'policy']):
            source_types.append('official_sources')
            
        return list(set(source_types))
        
    def _determine_intent(self, query_lower: str, research_type: ResearchType) -> str:
        """Determine the user's intent behind the query."""
        # Intent patterns
        intent_patterns = {
            'learn': ['learn', 'understand', 'know', 'find out'],
            'compare': ['compare', 'versus', 'difference', 'better'],
            'solve': ['solve', 'fix', 'resolve', 'solution'],
            'decide': ['should', 'choose', 'decision', 'recommend'],
            'verify': ['verify', 'confirm', 'check', 'validate'],
            'explore': ['explore', 'discover', 'investigate', 'research']
        }
        
        for intent, keywords in intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
                
        # Fallback based on research type
        type_intents = {
            ResearchType.EXPLORATORY: 'explore',
            ResearchType.COMPARATIVE: 'compare',
            ResearchType.EXPLANATORY: 'learn',
            ResearchType.CURRENT_EVENTS: 'learn'
        }
        
        return type_intents.get(research_type, 'learn')


class SourceCredibilityAnalyzer:
    """Analyzes the credibility of information sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SourceCredibilityAnalyzer")
        self._load_credibility_data()
        
    def _load_credibility_data(self):
        """Load credibility assessment data."""
        # Domain authority scores (simplified)
        self.domain_scores = {
            'edu': 0.9,
            'gov': 0.95,
            'org': 0.7,
            'com': 0.6,
            'net': 0.5,
            'info': 0.4
        }
        
        # High-credibility domains
        self.credible_domains = {
            'wikipedia.org': 0.8,
            'scholar.google.com': 0.95,
            'pubmed.ncbi.nlm.nih.gov': 0.95,
            'arxiv.org': 0.85,
            'nature.com': 0.9,
            'science.org': 0.9,
            'ieee.org': 0.9,
            'acm.org': 0.9,
            'britannica.com': 0.85,
            'who.int': 0.9,
            'cdc.gov': 0.9,
            'nytimes.com': 0.8,
            'bbc.com': 0.8,
            'reuters.com': 0.8
        }
        
        # Bias indicators
        self.bias_indicators = {
            'strong_language': ['amazing', 'incredible', 'shocking', 'devastating'],
            'emotional_appeals': ['you must', 'everyone knows', 'obviously'],
            'absolute_statements': ['always', 'never', 'all', 'none', 'every'],
            'conspiracy_language': ['they don\'t want you to know', 'hidden truth', 'secret']
        }
        
    def analyze_source(self, url: str, content: str = "", 
                      metadata: Dict[str, Any] = None) -> SourceCredibility:
        """
        Analyze the credibility of a source.
        
        Args:
            url: Source URL
            content: Source content (optional)
            metadata: Additional metadata about the source
            
        Returns:
            SourceCredibility assessment
        """
        metadata = metadata or {}
        
        # Parse URL
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Domain credibility score
        domain_score = self._assess_domain_credibility(domain)
        
        # Content quality score
        content_quality_score = self._assess_content_quality(content) if content else 0.5
        
        # Recency score
        recency_score = self._assess_recency(metadata.get('date_published'), metadata.get('date_modified'))
        
        # Authority score
        authority_score = self._assess_authority(domain, content, metadata)
        
        # Bias assessment
        bias_indicators = self._detect_bias(content) if content else []
        
        # Credibility factors
        credibility_factors = {
            'domain_authority': domain_score,
            'content_quality': content_quality_score,
            'recency': recency_score,
            'author_authority': authority_score,
            'bias_level': 1.0 - (len(bias_indicators) * 0.1)  # Reduce score for bias
        }
        
        # Calculate overall score
        weights = {'domain_authority': 0.3, 'content_quality': 0.25, 'recency': 0.15, 
                  'author_authority': 0.2, 'bias_level': 0.1}
        
        overall_score = sum(credibility_factors[factor] * weights[factor] 
                           for factor in weights if factor in credibility_factors)
        
        # Confidence in assessment
        confidence = self._calculate_confidence(credibility_factors, content, metadata)
        
        return SourceCredibility(
            domain_score=domain_score,
            content_quality_score=content_quality_score,
            recency_score=recency_score,
            authority_score=authority_score,
            bias_indicators=bias_indicators,
            credibility_factors=credibility_factors,
            overall_score=overall_score,
            confidence=confidence
        )
        
    def _assess_domain_credibility(self, domain: str) -> float:
        """Assess credibility based on domain."""
        # Check specific credible domains first
        for credible_domain, score in self.credible_domains.items():
            if credible_domain in domain:
                return score
                
        # Check TLD
        tld = domain.split('.')[-1]
        return self.domain_scores.get(tld, 0.5)
        
    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality based on various factors."""
        if not content:
            return 0.5
            
        score = 0.5  # Base score
        
        # Length factor (moderate length is better)
        word_count = len(content.split())
        if 100 <= word_count <= 2000:
            score += 0.1
        elif word_count > 2000:
            score += 0.05
            
        # Grammar and spelling (simplified check)
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
            score += 0.1
            
        # Citation indicators
        citation_indicators = ['(', ')', '[', ']', 'http', 'doi:', 'isbn:']
        if any(indicator in content.lower() for indicator in citation_indicators):
            score += 0.1
            
        # Professional language indicators
        professional_terms = ['research', 'study', 'analysis', 'data', 'findings', 'conclusion']
        professional_count = sum(1 for term in professional_terms if term in content.lower())
        score += min(0.2, professional_count * 0.05)
        
        return min(1.0, score)
        
    def _assess_recency(self, date_published: str = None, date_modified: str = None) -> float:
        """Assess recency of the source."""
        if not date_published and not date_modified:
            return 0.5  # Neutral if no date info
            
        # Use most recent date
        relevant_date = date_modified or date_published
        
        try:
            if isinstance(relevant_date, str):
                # Simple date parsing (would need more robust parsing in practice)
                from datetime import datetime
                source_date = datetime.fromisoformat(relevant_date.replace('Z', '+00:00'))
            else:
                source_date = relevant_date
                
            days_old = (datetime.now() - source_date).days
            
            # Scoring based on age
            if days_old <= 30:
                return 1.0
            elif days_old <= 90:
                return 0.8
            elif days_old <= 365:
                return 0.6
            elif days_old <= 365 * 2:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5
            
    def _assess_authority(self, domain: str, content: str, metadata: Dict[str, Any]) -> float:
        """Assess authority of the source."""
        score = 0.5
        
        # Author information
        if metadata.get('author'):
            score += 0.2
            
        # Professional domain indicators
        if any(indicator in domain for indicator in ['university', 'institute', 'foundation']):
            score += 0.2
            
        # Content authority indicators
        if content:
            authority_indicators = ['dr.', 'professor', 'phd', 'researcher', 'expert']
            if any(indicator in content.lower() for indicator in authority_indicators):
                score += 0.1
                
        return min(1.0, score)
        
    def _detect_bias(self, content: str) -> List[str]:
        """Detect potential bias indicators in content."""
        if not content:
            return []
            
        detected_bias = []
        content_lower = content.lower()
        
        for bias_type, indicators in self.bias_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                detected_bias.append(bias_type)
                
        return detected_bias
        
    def _calculate_confidence(self, credibility_factors: Dict[str, float], 
                            content: str, metadata: Dict[str, Any]) -> float:
        """Calculate confidence in the credibility assessment."""
        confidence = 0.5
        
        # More factors available = higher confidence
        factor_count = len([f for f in credibility_factors.values() if f != 0.5])
        confidence += min(0.3, factor_count * 0.1)
        
        # Content available = higher confidence
        if content and len(content) > 100:
            confidence += 0.1
            
        # Metadata available = higher confidence
        if metadata and len(metadata) > 2:
            confidence += 0.1
            
        return min(1.0, confidence)


class InformationSynthesizer:
    """Synthesizes information from multiple sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InformationSynthesizer")
        
    def synthesize(self, sources: List[Dict[str, Any]], 
                  query_analysis: QueryAnalysis) -> Tuple[str, List[InformationGap]]:
        """
        Synthesize information from multiple sources.
        
        Args:
            sources: List of source data with content and credibility
            query_analysis: Analysis of the original query
            
        Returns:
            Tuple of (synthesized_content, information_gaps)
        """
        if not sources:
            return "No sources available for synthesis.", [
                InformationGap(
                    gap_type="no_sources",
                    description="No sources found for the query",
                    severity="critical",
                    potential_sources=[],
                    search_suggestions=["Try broader search terms", "Check spelling"]
                )
            ]
            
        # Group sources by credibility
        high_credibility = [s for s in sources if s.get('credibility', {}).get('overall_score', 0) > 0.7]
        medium_credibility = [s for s in sources if 0.4 <= s.get('credibility', {}).get('overall_score', 0) <= 0.7]
        low_credibility = [s for s in sources if s.get('credibility', {}).get('overall_score', 0) < 0.4]
        
        # Start synthesis
        synthesis_parts = []
        
        # Handle main findings from high-credibility sources
        if high_credibility:
            synthesis_parts.append("## Key Findings from Authoritative Sources")
            for source in high_credibility[:3]:  # Top 3
                content = source.get('content', '')[:300]
                title = source.get('title', 'Unknown Source')
                synthesis_parts.append(f"**{title}**: {content}...")
                
        # Add supporting information from medium credibility sources
        if medium_credibility:
            synthesis_parts.append("\n## Additional Information")
            for source in medium_credibility[:2]:  # Top 2
                content = source.get('content', '')[:200]
                title = source.get('title', 'Unknown Source')
                synthesis_parts.append(f"**{title}**: {content}...")
                
        # Identify conflicting information
        conflicts = self._identify_conflicts(sources)
        if conflicts:
            synthesis_parts.append("\n## Conflicting Information")
            for conflict in conflicts:
                synthesis_parts.append(f"- {conflict}")
                
        # Identify gaps
        gaps = self._identify_gaps(sources, query_analysis)
        
        # Create final synthesis
        if synthesis_parts:
            final_synthesis = "\n".join(synthesis_parts)
        else:
            final_synthesis = "Unable to synthesize coherent information from available sources."
            
        return final_synthesis, gaps
        
    def _identify_conflicts(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Identify conflicting information between sources."""
        # Simplified conflict detection
        conflicts = []
        
        # Look for contradictory statements (simplified)
        statements = []
        for source in sources:
            content = source.get('content', '').lower()
            # Extract potential factual statements
            sentences = content.split('.')
            for sentence in sentences:
                if any(indicator in sentence for indicator in ['is', 'are', 'was', 'were']):
                    statements.append((sentence.strip(), source.get('title', 'Unknown')))
                    
        # Simple conflict detection (would need more sophisticated NLP in practice)
        contradiction_pairs = [
            ('increase', 'decrease'), ('rise', 'fall'), ('grow', 'shrink'),
            ('improve', 'worsen'), ('positive', 'negative')
        ]
        
        for i, (stmt1, source1) in enumerate(statements):
            for j, (stmt2, source2) in enumerate(statements[i+1:], i+1):
                for pos, neg in contradiction_pairs:
                    if pos in stmt1 and neg in stmt2:
                        conflicts.append(f"Contradiction between {source1} and {source2}")
                        break
                        
        return conflicts
        
    def _identify_gaps(self, sources: List[Dict[str, Any]], 
                      query_analysis: QueryAnalysis) -> List[InformationGap]:
        """Identify information gaps based on query analysis and available sources."""
        gaps = []
        
        # Check if key concepts are covered
        key_concepts = query_analysis.key_concepts
        covered_concepts = set()
        
        for source in sources:
            content = source.get('content', '').lower()
            for concept in key_concepts:
                if concept.lower() in content:
                    covered_concepts.add(concept)
                    
        uncovered_concepts = set(key_concepts) - covered_concepts
        if uncovered_concepts:
            gaps.append(InformationGap(
                gap_type="missing_concepts",
                description=f"Limited information about: {', '.join(uncovered_concepts)}",
                severity="medium",
                potential_sources=["Academic databases", "Specialized websites"],
                search_suggestions=[f"Search specifically for '{concept}'" for concept in uncovered_concepts]
            ))
            
        # Check source diversity
        source_types = [source.get('source_type', 'unknown') for source in sources]
        if len(set(source_types)) < 2:
            gaps.append(InformationGap(
                gap_type="source_diversity",
                description="Limited diversity in source types",
                severity="low",
                potential_sources=["Academic sources", "News sources", "Official sources"],
                search_suggestions=["Search in different types of sources"]
            ))
            
        # Check temporal coverage
        if any(indicator for indicator in query_analysis.temporal_indicators):
            # Should check if sources match temporal requirements
            gaps.append(InformationGap(
                gap_type="temporal_coverage",
                description="May not cover the requested time period comprehensively",
                severity="medium",
                potential_sources=["Historical archives", "Recent news sources"],
                search_suggestions=["Add time-specific search terms"]
            ))
            
        return gaps


# Example usage
async def main():
    """Example usage of research methodology components."""
    # Initialize components
    analyzer = QueryAnalyzer()
    credibility_analyzer = SourceCredibilityAnalyzer()
    synthesizer = InformationSynthesizer()
    
    # Example query
    query = "What are the latest developments in AI for medical diagnosis in 2024?"
    
    # Analyze query
    analysis = analyzer.analyze_query(query)
    print(f"Query Analysis for: {query}")
    print(f"Research Type: {analysis.research_type}")
    print(f"Key Concepts: {analysis.key_concepts}")
    print(f"Search Strategies: {analysis.search_strategies}")
    print(f"Complexity: {analysis.complexity_score:.2f}")
    print()
    
    # Example source credibility analysis
    test_url = "https://www.nature.com/articles/example"
    test_content = "This research study analyzes the implementation of artificial intelligence in medical diagnosis systems..."
    
    credibility = credibility_analyzer.analyze_source(test_url, test_content)
    print(f"Source Credibility Analysis:")
    print(f"Overall Score: {credibility.overall_score:.2f}")
    print(f"Domain Score: {credibility.domain_score:.2f}")
    print(f"Content Quality: {credibility.content_quality_score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())