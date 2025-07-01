"""
Enhanced Research Agent 2 - Integrated Framework

This module integrates all the advanced frameworks to create a comprehensive,
intelligent research agent with function calling, learning, safety, and
structured response capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Import all framework components
from function_calling_framework import (
    FunctionCallFramework, FunctionSpec, ParameterSpec, ParameterType,
    FunctionCallStatus, FunctionCallResult
)
from web_search_protocol import (
    WebSearchProtocol, QueryFormulation, SourceEvaluation, 
    InformationSynthesis, SearchResult
)
from learning_adaptation_framework import (
    LearningAdaptationFramework, LearningType, AdaptationLevel
)
from response_templates import (
    ResponseFormatter, ResponseType, Source, LimitationNote,
    FollowUpSuggestion, ConfidenceLevel
)
from safety_ethics_guidelines import (
    SafetyEthicsFramework, SafetyLevel, PrivacyLevel,
    InformationReliability
)

# Import existing components
from research_methodology import (
    QueryAnalyzer, SourceCredibilityAnalyzer, InformationSynthesizer
)
from api_integrations import APIIntegrationManager
from citation_manager import CitationManager, CitationStyle


@dataclass
class ResearchSession:
    """Represents a research session with context and history."""
    session_id: str
    user_id: str = "default"
    start_time: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    query_history: List[str] = field(default_factory=list)
    results_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchRequest:
    """Structured research request."""
    query: str
    session_id: str
    research_type: str = "comprehensive"
    max_sources: int = 15
    preferred_citation_style: str = "apa"
    time_constraints: Optional[Dict[str, Any]] = None
    domain_filters: List[str] = field(default_factory=list)
    quality_threshold: float = 0.6
    include_learning: bool = True
    safety_check: bool = True


class EnhancedResearchAgent:
    """
    Enhanced Research Agent with integrated advanced capabilities.
    
    This agent combines:
    - Function calling framework for API management
    - Web search protocol for intelligent search
    - Learning and adaptation for continuous improvement
    - Structured response templates for consistent output
    - Safety and ethics guidelines for responsible operation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Enhanced Research Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.EnhancedResearchAgent")
        
        # Initialize all framework components
        self._initialize_frameworks()
        
        # Session management
        self.active_sessions: Dict[str, ResearchSession] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_response_time": 0.0,
            "learning_events": 0,
            "safety_blocks": 0
        }
        
    def _initialize_frameworks(self):
        """Initialize all framework components."""
        try:
            # Core frameworks
            self.function_framework = FunctionCallFramework()
            self.web_search_protocol = WebSearchProtocol()
            self.learning_framework = LearningAdaptationFramework()
            self.response_formatter = ResponseFormatter()
            self.safety_framework = SafetyEthicsFramework()
            
            # Existing components
            self.query_analyzer = QueryAnalyzer()
            self.source_analyzer = SourceCredibilityAnalyzer()
            self.info_synthesizer = InformationSynthesizer()
            self.api_manager = APIIntegrationManager()
            self.citation_manager = CitationManager()
            
            # Register research functions
            self._register_research_functions()
            
            self.logger.info("All frameworks initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Framework initialization failed: {e}")
            raise
            
    def _register_research_functions(self):
        """Register research-related functions with the function framework."""
        
        # Web search function
        async def enhanced_web_search(query: str, max_results: int = 10, 
                                    domain_filters: List[str] = None) -> Dict[str, Any]:
            """Enhanced web search with protocol integration."""
            domain_filters = domain_filters or []
            context = {"domain_filters": domain_filters}
            
            query_form, results, synthesis = await self.web_search_protocol.conduct_enhanced_search(
                query, context, max_results
            )
            
            return {
                "query_formulation": {
                    "original": query_form.original_query,
                    "type": query_form.search_type.value,
                    "keywords": query_form.primary_keywords
                },
                "results_count": len(results),
                "synthesis": synthesis.summary,
                "confidence": synthesis.confidence_level,
                "sources": [{"title": r.title, "url": r.url} for r in results]
            }
            
        # Citation generation function
        def generate_citations(source_urls: List[str], style: str = "apa") -> Dict[str, Any]:
            """Generate citations for given sources."""
            citations = []
            for url in source_urls:
                citation = self.citation_manager.add_source(url=url, title=f"Source: {url}")
                formatted = self.citation_manager.format_citation(citation.id, CitationStyle(style))
                citations.append({"id": citation.id, "formatted": formatted})
                
            return {"citations": citations, "style": style, "count": len(citations)}
            
        # Information validation function
        def validate_information(content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Validate information using safety and ethics framework."""
            validation = self.safety_framework.validate_research_output(content, sources)
            return validation
            
        # Register functions with specifications
        search_spec = FunctionSpec(
            name="enhanced_web_search",
            description="Perform enhanced web search with intelligent protocols",
            parameters=[
                ParameterSpec("query", ParameterType.REQUIRED, str, description="Search query"),
                ParameterSpec("max_results", ParameterType.OPTIONAL, int, 10, "Maximum results"),
                ParameterSpec("domain_filters", ParameterType.OPTIONAL, list, [], "Domain filters")
            ],
            return_type=Dict[str, Any],
            tags=["search", "research", "web"]
        )
        
        citation_spec = FunctionSpec(
            name="generate_citations",
            description="Generate formatted citations for sources",
            parameters=[
                ParameterSpec("source_urls", ParameterType.REQUIRED, list, description="List of source URLs"),
                ParameterSpec("style", ParameterType.OPTIONAL, str, "apa", "Citation style")
            ],
            return_type=Dict[str, Any],
            tags=["citation", "formatting", "academic"]
        )
        
        validation_spec = FunctionSpec(
            name="validate_information",
            description="Validate information for accuracy and ethics",
            parameters=[
                ParameterSpec("content", ParameterType.REQUIRED, str, description="Content to validate"),
                ParameterSpec("sources", ParameterType.REQUIRED, list, description="Source information")
            ],
            return_type=Dict[str, Any],
            tags=["validation", "ethics", "safety"]
        )
        
        # Register functions
        self.function_framework.register_function(enhanced_web_search, search_spec)
        self.function_framework.register_function(generate_citations, citation_spec)
        self.function_framework.register_function(validate_information, validation_spec)
        
    async def conduct_enhanced_research(self, request: ResearchRequest) -> Dict[str, Any]:
        """
        Conduct enhanced research with all integrated capabilities.
        
        Args:
            request: Structured research request
            
        Returns:
            Comprehensive research response
        """
        start_time = datetime.now()
        session = self._get_or_create_session(request.session_id)
        
        try:
            # Step 1: Safety and ethics check
            if request.safety_check:
                safety_assessment = await self._conduct_safety_check(request.query, session)
                if not safety_assessment["safe_to_proceed"]:
                    return self._format_blocked_response(request.query, safety_assessment)
                    
            # Step 2: Query analysis and adaptation
            adapted_approach = await self._adapt_research_approach(request, session)
            
            # Step 3: Enhanced web search
            search_results = await self._conduct_enhanced_search(request, adapted_approach)
            
            # Step 4: Information synthesis and validation
            synthesis_results = await self._synthesize_and_validate(
                request.query, search_results, session
            )
            
            # Step 5: Citation generation
            citations = await self._generate_citations(
                search_results["sources"], request.preferred_citation_style
            )
            
            # Step 6: Learning and adaptation
            if request.include_learning:
                await self._record_learning_events(request, search_results, synthesis_results)
                
            # Step 7: Format comprehensive response
            response = await self._format_comprehensive_response(
                request, search_results, synthesis_results, citations, start_time
            )
            
            # Update session and metrics
            session.query_history.append(request.query)
            session.results_history.append(response)
            self._update_performance_metrics(True, start_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            self._update_performance_metrics(False, start_time)
            
            return self.response_formatter.format_response(
                ResponseType.ERROR_RESPONSE,
                error_type="research_failure",
                error_message=str(e),
                context={"query": request.query, "session_id": request.session_id},
                alternative_approaches=["Try a simpler query", "Check network connectivity"],
                recovery_suggestions=["Retry with different parameters", "Contact support"]
            )
            
    async def _conduct_safety_check(self, query: str, session: ResearchSession) -> Dict[str, Any]:
        """Conduct comprehensive safety and ethics check."""
        context = {
            "session_id": session.session_id,
            "user_preferences": session.preferences,
            "query_history": session.query_history[-5:]  # Last 5 queries for context
        }
        
        assessment = self.safety_framework.comprehensive_assessment(query, context)
        
        if assessment["blocked"]:
            self.performance_metrics["safety_blocks"] += 1
            
        return assessment
        
    async def _adapt_research_approach(self, request: ResearchRequest, 
                                     session: ResearchSession) -> Dict[str, Any]:
        """Adapt research approach based on learning and context."""
        context = {
            "query_type": request.research_type,
            "user_preferences": session.preferences,
            "session_history": len(session.query_history)
        }
        
        # Get adaptive recommendations
        search_adaptation = self.learning_framework.adapt_search_approach(request.query, context)
        
        # Merge with request parameters
        adapted_approach = {
            "original_request": request,
            "adaptations": search_adaptation,
            "effective_max_sources": request.max_sources,
            "effective_quality_threshold": request.quality_threshold,
            "recommended_keywords": search_adaptation.get("recommended_keywords", []),
            "preferred_sources": search_adaptation.get("preferred_sources", [])
        }
        
        return adapted_approach
        
    async def _conduct_enhanced_search(self, request: ResearchRequest, 
                                     adapted_approach: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct enhanced search using the web search protocol."""
        context = {
            "research_type": request.research_type,
            "domain_filters": request.domain_filters,
            "time_constraints": request.time_constraints,
            "quality_threshold": request.quality_threshold,
            "adaptations": adapted_approach["adaptations"]
        }
        
        # Use function calling framework for search
        search_result = await self.function_framework.call_function(
            "enhanced_web_search",
            {
                "query": request.query,
                "max_results": adapted_approach["effective_max_sources"],
                "domain_filters": request.domain_filters
            },
            context=f"Research query: {request.research_type}"
        )
        
        if search_result.status == FunctionCallStatus.SUCCESS:
            return search_result.result
        else:
            raise Exception(f"Search failed: {search_result.error}")
            
    async def _synthesize_and_validate(self, query: str, search_results: Dict[str, Any], 
                                     session: ResearchSession) -> Dict[str, Any]:
        """Synthesize information and validate for accuracy and ethics."""
        # Extract sources for validation
        sources = search_results.get("sources", [])
        synthesis_content = search_results.get("synthesis", "")
        
        # Validate using safety framework
        validation_result = await self.function_framework.call_function(
            "validate_information",
            {
                "content": synthesis_content,
                "sources": sources
            },
            context="Information synthesis validation"
        )
        
        if validation_result.status == FunctionCallStatus.SUCCESS:
            validation = validation_result.result
        else:
            validation = {"approved": False, "confidence_in_output": 0.3}
            
        # Combine with synthesis results
        synthesis_results = {
            "synthesis": synthesis_content,
            "confidence": search_results.get("confidence", 0.5),
            "validation": validation,
            "source_count": len(sources),
            "quality_assessment": self._assess_result_quality(search_results)
        }
        
        return synthesis_results
        
    async def _generate_citations(self, sources: List[Dict[str, Any]], 
                                style: str) -> Dict[str, Any]:
        """Generate citations using the citation framework."""
        source_urls = [source.get("url", "") for source in sources if source.get("url")]
        
        citation_result = await self.function_framework.call_function(
            "generate_citations",
            {
                "source_urls": source_urls,
                "style": style
            },
            context="Citation generation"
        )
        
        if citation_result.status == FunctionCallStatus.SUCCESS:
            return citation_result.result
        else:
            return {"citations": [], "style": style, "count": 0, "error": citation_result.error}
            
    async def _record_learning_events(self, request: ResearchRequest,
                                    search_results: Dict[str, Any],
                                    synthesis_results: Dict[str, Any]):
        """Record learning events for future adaptation."""
        # Record search strategy success
        self.learning_framework.record_learning_event(
            LearningType.SEARCH_STRATEGY,
            {
                "query_type": request.research_type,
                "query": request.query,
                "sources_found": search_results.get("results_count", 0),
                "confidence_achieved": synthesis_results.get("confidence", 0.5),
                "quality_threshold": request.quality_threshold
            },
            "success" if synthesis_results.get("confidence", 0) > 0.6 else "partial",
            {
                "session_id": request.session_id,
                "research_type": request.research_type
            },
            confidence=synthesis_results.get("confidence", 0.5)
        )
        
        # Record function usage learning
        self.learning_framework.learn_from_api_usage(
            api_name="enhanced_web_search",
            endpoint="/search",
            parameters={
                "query": request.query,
                "max_results": request.max_sources
            },
            success=True,
            response_time=1.0  # Would be actual response time
        )
        
        self.performance_metrics["learning_events"] += 1
        
    async def _format_comprehensive_response(self, request: ResearchRequest,
                                           search_results: Dict[str, Any],
                                           synthesis_results: Dict[str, Any],
                                           citations: Dict[str, Any],
                                           start_time: datetime) -> Dict[str, Any]:
        """Format comprehensive response using response templates."""
        # Prepare sources for response template
        sources = []
        for source in search_results.get("sources", []):
            sources.append(Source(
                title=source.get("title", "Unknown"),
                url=source.get("url", ""),
                credibility_score=source.get("credibility_score", 0.5),
                source_type=source.get("source_type", "web")
            ))
            
        # Prepare limitations
        limitations = []
        validation = synthesis_results.get("validation", {})
        if not validation.get("approved", True):
            limitations.append(LimitationNote(
                type="validation",
                description="Information validation detected potential issues",
                severity="medium",
                suggestions=["Verify information from additional sources"]
            ))
            
        # Prepare follow-ups
        follow_ups = []
        if synthesis_results.get("confidence", 0.5) < 0.7:
            follow_ups.append(FollowUpSuggestion(
                title="Additional Verification",
                description="Seek additional sources to improve confidence",
                priority="high",
                estimated_effort="medium"
            ))
            
        # Format using appropriate template
        if request.research_type == "comprehensive":
            response = self.response_formatter.format_response(
                ResponseType.DETAILED_ANALYSIS,
                query=request.query,
                methodology={
                    "approach": "Enhanced multi-framework research",
                    "type": request.research_type,
                    "search_strategy": search_results.get("query_formulation", {}).get("keywords", []),
                    "evaluation_criteria": ["Source credibility", "Information validation", "Ethics compliance"]
                },
                findings={
                    "primary_synthesis": synthesis_results.get("synthesis", ""),
                    "confidence_level": synthesis_results.get("confidence", 0.5),
                    "source_analysis": search_results.get("source_distribution", {})
                },
                analysis={
                    "quality_assessment": synthesis_results.get("quality_assessment", {}),
                    "validation_results": validation,
                    "learning_insights": "Adaptive recommendations applied"
                },
                sources=sources,
                metadata={
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "frameworks_used": ["Function Calling", "Web Search Protocol", "Learning Framework", "Safety Framework"],
                    "citations": citations,
                    "session_id": request.session_id
                }
            )
        else:
            response = self.response_formatter.format_response(
                ResponseType.RESEARCH_SUMMARY,
                query=request.query,
                key_findings=[synthesis_results.get("synthesis", "")],
                sources=sources,
                confidence_level=synthesis_results.get("confidence", 0.5),
                limitations=limitations,
                follow_ups=follow_ups
            )
            
        return response
        
    def _format_blocked_response(self, query: str, safety_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Format response for blocked queries."""
        return self.response_formatter.format_response(
            ResponseType.ERROR_RESPONSE,
            error_type="safety_block",
            error_message="Query blocked due to safety or ethics concerns",
            context={
                "query": query,
                "safety_level": safety_assessment["safety_assessment"]["level"],
                "privacy_level": safety_assessment["privacy_assessment"]["level"]
            },
            alternative_approaches=safety_assessment["safety_assessment"].get("alternatives", []),
            recovery_suggestions=safety_assessment["guidelines"]
        )
        
    def _get_or_create_session(self, session_id: str) -> ResearchSession:
        """Get existing session or create new one."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ResearchSession(session_id=session_id)
        return self.active_sessions[session_id]
        
    def _assess_result_quality(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of search results."""
        return {
            "source_count": search_results.get("results_count", 0),
            "confidence_level": search_results.get("confidence", 0.5),
            "query_formulation_quality": "good" if search_results.get("query_formulation") else "basic"
        }
        
    def _update_performance_metrics(self, success: bool, start_time: datetime):
        """Update performance metrics."""
        self.performance_metrics["total_queries"] += 1
        if success:
            self.performance_metrics["successful_queries"] += 1
            
        response_time = (datetime.now() - start_time).total_seconds()
        current_avg = self.performance_metrics["average_response_time"]
        total_queries = self.performance_metrics["total_queries"]
        
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
        
    async def get_function_recommendations(self, query: str, context: str = "") -> List[Tuple[str, float]]:
        """Get function recommendations for a query."""
        return self.function_framework.get_function_recommendations(context, query)
        
    async def call_specific_function(self, function_name: str, 
                                   parameters: Dict[str, Any],
                                   context: str = "") -> FunctionCallResult:
        """Call a specific function with parameters."""
        return await self.function_framework.call_function(function_name, parameters, context)
        
    def get_learning_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Get learning and adaptation report."""
        return self.learning_framework.generate_learning_report(days_back)
        
    def get_safety_compliance_report(self) -> Dict[str, Any]:
        """Get safety and ethics compliance report."""
        return self.safety_framework.generate_compliance_report()
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
        
    def get_available_functions(self) -> List[str]:
        """Get list of available functions."""
        return self.function_framework.registry.list_functions()
        
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        return {
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": self.get_performance_metrics(),
            "learning_report": self.get_learning_report(7),  # Last 7 days
            "safety_report": self.get_safety_compliance_report(),
            "available_functions": len(self.get_available_functions()),
            "active_sessions": len(self.active_sessions),
            "frameworks_loaded": [
                "Function Calling Framework",
                "Web Search Protocol", 
                "Learning Adaptation Framework",
                "Response Templates",
                "Safety Ethics Guidelines"
            ]
        }


# Convenience function for easy usage
async def conduct_research(query: str, 
                          research_type: str = "comprehensive",
                          max_sources: int = 15,
                          citation_style: str = "apa",
                          session_id: str = "default") -> Dict[str, Any]:
    """
    Convenience function for conducting research.
    
    Args:
        query: Research query
        research_type: Type of research to conduct
        max_sources: Maximum sources to consider
        citation_style: Preferred citation style
        session_id: Session identifier
        
    Returns:
        Comprehensive research results
    """
    agent = EnhancedResearchAgent()
    
    request = ResearchRequest(
        query=query,
        session_id=session_id,
        research_type=research_type,
        max_sources=max_sources,
        preferred_citation_style=citation_style
    )
    
    return await agent.conduct_enhanced_research(request)


# Example usage
async def main():
    """Example usage of the Enhanced Research Agent."""
    # Initialize agent
    agent = EnhancedResearchAgent()
    
    # Create research request
    request = ResearchRequest(
        query="What are the latest developments in AI for medical diagnosis?",
        session_id="demo_session",
        research_type="comprehensive",
        max_sources=10,
        preferred_citation_style="apa"
    )
    
    print("Enhanced Research Agent - Demo")
    print("=" * 50)
    
    # Conduct research
    result = await agent.conduct_enhanced_research(request)
    
    # Display results
    print("Research completed!")
    print(f"Response Type: {result.get('response_type')}")
    print(f"Query: {result.get('query', {}).get('original')}")
    
    if result.get('response_type') == 'detailed_analysis':
        print(f"Methodology: {result.get('methodology', {}).get('approach')}")
        print(f"Sources: {len(result.get('sources', {}).get('complete_source_list', []))}")
        
    # Get system report
    print("\n" + "=" * 50)
    print("System Report:")
    system_report = agent.generate_system_report()
    print(f"Status: {system_report['system_status']}")
    print(f"Performance: {system_report['performance_metrics']['successful_queries']}/{system_report['performance_metrics']['total_queries']} successful")
    print(f"Available Functions: {system_report['available_functions']}")
    print(f"Frameworks: {len(system_report['frameworks_loaded'])}")


if __name__ == "__main__":
    asyncio.run(main())