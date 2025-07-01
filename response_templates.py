"""
Response Structure Templates for Research Agent 2

This module provides structured response templates for different types of research
queries, API usage, and user interactions with consistent formatting and organization.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import textwrap


class ResponseType(Enum):
    """Types of responses."""
    RESEARCH_SUMMARY = "research_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    API_RESULT = "api_result"
    SEARCH_RESULT = "search_result"
    ERROR_RESPONSE = "error_response"
    PROGRESS_UPDATE = "progress_update"
    RECOMMENDATION = "recommendation"
    CITATION_LIST = "citation_list"


class ConfidenceLevel(Enum):
    """Confidence levels for information."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9


@dataclass
class Source:
    """Represents a source with metadata."""
    title: str
    url: str
    publication_date: Optional[str] = None
    author: Optional[str] = None
    credibility_score: float = 0.5
    source_type: str = "web"
    relevance_score: float = 0.5


@dataclass
class LimitationNote:
    """Represents a limitation or caveat."""
    type: str  # scope, data_quality, temporal, methodological
    description: str
    severity: str  # low, medium, high
    suggestions: List[str] = field(default_factory=list)


@dataclass
class FollowUpSuggestion:
    """Represents a follow-up research suggestion."""
    title: str
    description: str
    priority: str  # high, medium, low
    estimated_effort: str  # low, medium, high
    potential_sources: List[str] = field(default_factory=list)


@dataclass
class ProgressStatus:
    """Represents progress status for long-running tasks."""
    task_id: str
    current_step: str
    total_steps: int
    completed_steps: int
    estimated_remaining_time: Optional[int] = None  # seconds
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)


class ResearchSummaryTemplate:
    """Template for research summary responses."""
    
    def __init__(self):
        self.template_name = "Research Summary"
        
    def generate(self, 
                query: str,
                key_findings: List[str],
                sources: List[Source],
                confidence_level: float,
                limitations: List[LimitationNote] = None,
                follow_ups: List[FollowUpSuggestion] = None) -> Dict[str, Any]:
        """
        Generate a research summary response.
        
        Args:
            query: Original research query
            key_findings: List of key findings
            sources: List of sources consulted
            confidence_level: Overall confidence in findings
            limitations: Known limitations
            follow_ups: Suggested follow-up research
            
        Returns:
            Structured response dictionary
        """
        limitations = limitations or []
        follow_ups = follow_ups or []
        
        # Categorize sources by credibility
        high_credibility = [s for s in sources if s.credibility_score >= 0.7]
        medium_credibility = [s for s in sources if 0.4 <= s.credibility_score < 0.7]
        low_credibility = [s for s in sources if s.credibility_score < 0.4]
        
        # Generate confidence assessment
        confidence_desc = self._get_confidence_description(confidence_level)
        
        response = {
            "response_type": ResponseType.RESEARCH_SUMMARY.value,
            "timestamp": datetime.now().isoformat(),
            "query": {
                "original": query,
                "processed": query  # Could be enhanced with query processing
            },
            "summary": {
                "overview": self._generate_overview(key_findings, confidence_level),
                "key_findings": [
                    {
                        "finding": finding,
                        "supporting_sources": self._get_supporting_sources(finding, sources)
                    }
                    for finding in key_findings
                ],
                "confidence": {
                    "level": confidence_level,
                    "description": confidence_desc,
                    "factors": self._get_confidence_factors(sources, limitations)
                }
            },
            "sources": {
                "total_consulted": len(sources),
                "by_credibility": {
                    "high": len(high_credibility),
                    "medium": len(medium_credibility),
                    "low": len(low_credibility)
                },
                "high_credibility_sources": [asdict(s) for s in high_credibility[:5]],
                "all_sources": [asdict(s) for s in sources]
            },
            "limitations": [asdict(l) for l in limitations],
            "follow_up_suggestions": [asdict(f) for f in follow_ups]
        }
        
        return response
        
    def _generate_overview(self, findings: List[str], confidence: float) -> str:
        """Generate an overview summary."""
        if not findings:
            return "No significant findings were identified from the available sources."
            
        confidence_phrase = "with high confidence" if confidence > 0.7 else \
                           "with moderate confidence" if confidence > 0.5 else \
                           "with limited confidence"
                           
        overview = f"Research {confidence_phrase} identified {len(findings)} key findings. "
        
        if len(findings) == 1:
            overview += f"The primary finding indicates: {findings[0][:100]}..."
        else:
            overview += f"The findings span multiple aspects of the query, with primary focus on: {findings[0][:80]}..."
            
        return overview
        
    def _get_supporting_sources(self, finding: str, sources: List[Source]) -> List[str]:
        """Get sources that support a particular finding."""
        # Simple matching - could be enhanced with semantic similarity
        supporting = []
        finding_lower = finding.lower()
        
        for source in sources:
            if any(word in source.title.lower() for word in finding_lower.split()[:3]):
                supporting.append(source.url)
                
        return supporting[:3]  # Limit to top 3
        
    def _get_confidence_description(self, confidence: float) -> str:
        """Get human-readable confidence description."""
        if confidence >= 0.9:
            return "Very high confidence - findings are well-supported by authoritative sources"
        elif confidence >= 0.7:
            return "High confidence - findings are supported by credible sources with minimal conflicts"
        elif confidence >= 0.5:
            return "Moderate confidence - findings have reasonable support but may have some limitations"
        elif confidence >= 0.3:
            return "Low confidence - findings are preliminary and require additional verification"
        else:
            return "Very low confidence - findings are speculative and need substantial verification"
            
    def _get_confidence_factors(self, sources: List[Source], 
                              limitations: List[LimitationNote]) -> List[str]:
        """Get factors affecting confidence."""
        factors = []
        
        # Source quality factors
        high_cred_count = sum(1 for s in sources if s.credibility_score >= 0.7)
        if high_cred_count >= 3:
            factors.append(f"Multiple high-credibility sources ({high_cred_count})")
        elif high_cred_count > 0:
            factors.append(f"Some high-credibility sources ({high_cred_count})")
        else:
            factors.append("Limited high-credibility sources")
            
        # Limitation factors
        if limitations:
            factors.append(f"{len(limitations)} known limitations identified")
            
        # Source diversity
        source_types = set(s.source_type for s in sources)
        if len(source_types) > 2:
            factors.append("Diverse source types consulted")
            
        return factors


class DetailedAnalysisTemplate:
    """Template for detailed analysis responses."""
    
    def __init__(self):
        self.template_name = "Detailed Analysis"
        
    def generate(self,
                query: str,
                methodology: Dict[str, Any],
                findings: Dict[str, Any],
                analysis: Dict[str, Any],
                sources: List[Source],
                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a detailed analysis response.
        
        Args:
            query: Original research query
            methodology: Research methodology used
            findings: Detailed findings organized by category
            analysis: Analysis and interpretation
            sources: Sources with full details
            metadata: Additional metadata
            
        Returns:
            Structured detailed analysis response
        """
        metadata = metadata or {}
        
        response = {
            "response_type": ResponseType.DETAILED_ANALYSIS.value,
            "timestamp": datetime.now().isoformat(),
            "query": {
                "original": query,
                "research_type": methodology.get("type", "comprehensive"),
                "scope": methodology.get("scope", "general")
            },
            "methodology": {
                "approach": methodology.get("approach", "multi-source synthesis"),
                "search_strategy": methodology.get("search_strategy", []),
                "evaluation_criteria": methodology.get("evaluation_criteria", []),
                "time_period": methodology.get("time_period"),
                "geographic_scope": methodology.get("geographic_scope")
            },
            "detailed_findings": findings,
            "analysis": analysis,
            "sources": {
                "methodology": {
                    "total_sources_reviewed": len(sources),
                    "inclusion_criteria": metadata.get("inclusion_criteria", []),
                    "exclusion_criteria": metadata.get("exclusion_criteria", [])
                },
                "source_analysis": self._analyze_sources(sources),
                "complete_source_list": [self._format_detailed_source(s) for s in sources]
            },
            "metadata": metadata
        }
        
        return response
        
    def _analyze_sources(self, sources: List[Source]) -> Dict[str, Any]:
        """Analyze source characteristics."""
        if not sources:
            return {}
            
        # Publication date analysis
        dated_sources = [s for s in sources if s.publication_date]
        recent_sources = sum(1 for s in dated_sources 
                           if s.publication_date and "2023" in s.publication_date or "2024" in s.publication_date)
        
        # Source type distribution
        type_dist = {}
        for source in sources:
            type_dist[source.source_type] = type_dist.get(source.source_type, 0) + 1
            
        # Credibility distribution
        cred_dist = {
            "very_high": sum(1 for s in sources if s.credibility_score >= 0.9),
            "high": sum(1 for s in sources if 0.7 <= s.credibility_score < 0.9),
            "medium": sum(1 for s in sources if 0.4 <= s.credibility_score < 0.7),
            "low": sum(1 for s in sources if s.credibility_score < 0.4)
        }
        
        return {
            "temporal_distribution": {
                "total_with_dates": len(dated_sources),
                "recent_sources": recent_sources,
                "percentage_recent": (recent_sources / len(dated_sources) * 100) if dated_sources else 0
            },
            "source_type_distribution": type_dist,
            "credibility_distribution": cred_dist,
            "average_credibility": sum(s.credibility_score for s in sources) / len(sources),
            "average_relevance": sum(s.relevance_score for s in sources) / len(sources)
        }
        
    def _format_detailed_source(self, source: Source) -> Dict[str, Any]:
        """Format source with detailed information."""
        return {
            "title": source.title,
            "url": source.url,
            "author": source.author,
            "publication_date": source.publication_date,
            "source_type": source.source_type,
            "credibility_assessment": {
                "score": source.credibility_score,
                "level": self._get_credibility_level(source.credibility_score)
            },
            "relevance_score": source.relevance_score
        }
        
    def _get_credibility_level(self, score: float) -> str:
        """Convert credibility score to level."""
        if score >= 0.9:
            return "very_high"
        elif score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"


class APIResultTemplate:
    """Template for API call results."""
    
    def __init__(self):
        self.template_name = "API Result"
        
    def generate(self,
                action_taken: str,
                api_details: Dict[str, Any],
                results: Any,
                integration_context: str,
                next_steps: List[str] = None,
                errors: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an API result response.
        
        Args:
            action_taken: Description of the action performed
            api_details: Details about the API call
            results: The actual results obtained
            integration_context: How this fits into broader research
            next_steps: Suggested next steps
            errors: Any errors encountered
            
        Returns:
            Structured API result response
        """
        next_steps = next_steps or []
        errors = errors or []
        
        response = {
            "response_type": ResponseType.API_RESULT.value,
            "timestamp": datetime.now().isoformat(),
            "action_taken": {
                "description": action_taken,
                "api_name": api_details.get("api_name"),
                "endpoint": api_details.get("endpoint"),
                "method": api_details.get("method", "GET"),
                "parameters": api_details.get("parameters", {}),
                "execution_time": api_details.get("execution_time")
            },
            "results": {
                "success": len(errors) == 0,
                "data": results,
                "data_summary": self._summarize_results(results),
                "quality_assessment": self._assess_result_quality(results)
            },
            "integration": {
                "context": integration_context,
                "relevance_to_query": api_details.get("relevance", "high"),
                "data_reliability": api_details.get("reliability", "medium")
            },
            "next_steps": next_steps,
            "errors": errors,
            "metadata": {
                "result_count": self._count_results(results),
                "data_types": self._identify_data_types(results),
                "processing_notes": api_details.get("processing_notes", [])
            }
        }
        
        return response
        
    def _summarize_results(self, results: Any) -> Dict[str, Any]:
        """Summarize the results data."""
        summary = {}
        
        if isinstance(results, dict):
            summary["type"] = "object"
            summary["keys"] = list(results.keys())[:10]  # First 10 keys
            summary["total_keys"] = len(results.keys()) if hasattr(results, 'keys') else 0
            
        elif isinstance(results, list):
            summary["type"] = "list"
            summary["length"] = len(results)
            if results and isinstance(results[0], dict):
                summary["item_structure"] = list(results[0].keys())[:5]
                
        elif isinstance(results, str):
            summary["type"] = "string"
            summary["length"] = len(results)
            summary["preview"] = results[:100] + "..." if len(results) > 100 else results
            
        else:
            summary["type"] = type(results).__name__
            summary["value"] = str(results)[:100]
            
        return summary
        
    def _assess_result_quality(self, results: Any) -> Dict[str, Any]:
        """Assess the quality of results."""
        assessment = {
            "completeness": "unknown",
            "data_richness": "unknown",
            "structure_quality": "unknown"
        }
        
        if isinstance(results, list):
            if len(results) == 0:
                assessment["completeness"] = "empty"
            elif len(results) < 5:
                assessment["completeness"] = "limited"
            else:
                assessment["completeness"] = "good"
                
            if results and isinstance(results[0], dict):
                avg_fields = sum(len(item.keys()) for item in results[:5]) / min(5, len(results))
                assessment["data_richness"] = "rich" if avg_fields > 5 else "moderate"
                
        elif isinstance(results, dict):
            field_count = len(results.keys())
            assessment["data_richness"] = "rich" if field_count > 10 else "moderate"
            assessment["structure_quality"] = "good"
            
        return assessment
        
    def _count_results(self, results: Any) -> int:
        """Count the number of results."""
        if isinstance(results, list):
            return len(results)
        elif isinstance(results, dict):
            return len(results.keys())
        else:
            return 1
            
    def _identify_data_types(self, results: Any) -> List[str]:
        """Identify the types of data in results."""
        types = set()
        
        if isinstance(results, list):
            for item in results[:5]:  # Sample first 5 items
                types.add(type(item).__name__)
                if isinstance(item, dict):
                    for value in item.values():
                        types.add(type(value).__name__)
                        
        elif isinstance(results, dict):
            for value in results.values():
                types.add(type(value).__name__)
                
        else:
            types.add(type(results).__name__)
            
        return list(types)


class ErrorResponseTemplate:
    """Template for error responses."""
    
    def __init__(self):
        self.template_name = "Error Response"
        
    def generate(self,
                error_type: str,
                error_message: str,
                context: Dict[str, Any],
                alternative_approaches: List[str] = None,
                partial_results: Any = None,
                recovery_suggestions: List[str] = None) -> Dict[str, Any]:
        """
        Generate an error response.
        
        Args:
            error_type: Type of error encountered
            error_message: Detailed error message
            context: Context where error occurred
            alternative_approaches: Suggested alternative approaches
            partial_results: Any partial results obtained
            recovery_suggestions: Suggestions for recovery
            
        Returns:
            Structured error response
        """
        alternative_approaches = alternative_approaches or []
        recovery_suggestions = recovery_suggestions or []
        
        response = {
            "response_type": ResponseType.ERROR_RESPONSE.value,
            "timestamp": datetime.now().isoformat(),
            "error": {
                "type": error_type,
                "message": error_message,
                "severity": self._assess_error_severity(error_type),
                "context": context,
                "recoverable": len(alternative_approaches) > 0 or len(recovery_suggestions) > 0
            },
            "impact_assessment": {
                "query_completion": "failed" if not partial_results else "partial",
                "data_availability": "none" if not partial_results else "limited",
                "confidence_impact": "high"
            },
            "partial_results": partial_results,
            "alternative_approaches": alternative_approaches,
            "recovery_suggestions": recovery_suggestions,
            "next_actions": self._generate_next_actions(error_type, alternative_approaches, recovery_suggestions)
        }
        
        return response
        
    def _assess_error_severity(self, error_type: str) -> str:
        """Assess the severity of the error."""
        high_severity = ["authentication", "permission", "quota_exceeded", "service_unavailable"]
        medium_severity = ["timeout", "rate_limit", "validation", "not_found"]
        
        if error_type.lower() in high_severity:
            return "high"
        elif error_type.lower() in medium_severity:
            return "medium"
        else:
            return "low"
            
    def _generate_next_actions(self, error_type: str, 
                             alternatives: List[str], 
                             suggestions: List[str]) -> List[str]:
        """Generate recommended next actions."""
        actions = []
        
        if alternatives:
            actions.append(f"Try alternative approach: {alternatives[0]}")
            
        if suggestions:
            actions.append(f"Follow recovery suggestion: {suggestions[0]}")
            
        # Error-specific actions
        if "timeout" in error_type.lower():
            actions.append("Retry with increased timeout values")
        elif "rate_limit" in error_type.lower():
            actions.append("Wait and retry after rate limit reset")
        elif "authentication" in error_type.lower():
            actions.append("Verify and refresh authentication credentials")
            
        if not actions:
            actions.append("Manual intervention may be required")
            
        return actions


class ProgressUpdateTemplate:
    """Template for progress updates on long-running tasks."""
    
    def __init__(self):
        self.template_name = "Progress Update"
        
    def generate(self,
                task_id: str,
                current_step: str,
                progress_percentage: float,
                intermediate_results: List[Dict[str, Any]] = None,
                estimated_completion: Optional[datetime] = None,
                issues_encountered: List[str] = None) -> Dict[str, Any]:
        """
        Generate a progress update response.
        
        Args:
            task_id: Unique task identifier
            current_step: Description of current step
            progress_percentage: Completion percentage (0-100)
            intermediate_results: Any intermediate results
            estimated_completion: Estimated completion time
            issues_encountered: Any issues found so far
            
        Returns:
            Structured progress update response
        """
        intermediate_results = intermediate_results or []
        issues_encountered = issues_encountered or []
        
        response = {
            "response_type": ResponseType.PROGRESS_UPDATE.value,
            "timestamp": datetime.now().isoformat(),
            "task": {
                "id": task_id,
                "current_step": current_step,
                "progress_percentage": min(100, max(0, progress_percentage)),
                "status": "in_progress" if progress_percentage < 100 else "completed"
            },
            "timing": {
                "estimated_completion": estimated_completion.isoformat() if estimated_completion else None,
                "estimated_remaining_minutes": self._calculate_remaining_time(progress_percentage, estimated_completion)
            },
            "intermediate_results": {
                "count": len(intermediate_results),
                "summary": self._summarize_intermediate_results(intermediate_results),
                "details": intermediate_results
            },
            "issues": {
                "count": len(issues_encountered),
                "list": issues_encountered,
                "impact_on_completion": "minimal" if len(issues_encountered) < 2 else "moderate"
            }
        }
        
        return response
        
    def _calculate_remaining_time(self, progress: float, 
                                estimated_completion: Optional[datetime]) -> Optional[int]:
        """Calculate estimated remaining time in minutes."""
        if estimated_completion:
            remaining = (estimated_completion - datetime.now()).total_seconds() / 60
            return max(0, int(remaining))
        return None
        
    def _summarize_intermediate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize intermediate results."""
        if not results:
            return {"status": "no_results_yet"}
            
        summary = {
            "total_items": len(results),
            "types": list(set(r.get("type", "unknown") for r in results)),
            "latest_result": results[-1] if results else None
        }
        
        return summary


class ResponseFormatter:
    """Main formatter for generating structured responses."""
    
    def __init__(self):
        self.templates = {
            ResponseType.RESEARCH_SUMMARY: ResearchSummaryTemplate(),
            ResponseType.DETAILED_ANALYSIS: DetailedAnalysisTemplate(),
            ResponseType.API_RESULT: APIResultTemplate(),
            ResponseType.ERROR_RESPONSE: ErrorResponseTemplate(),
            ResponseType.PROGRESS_UPDATE: ProgressUpdateTemplate()
        }
        
    def format_response(self, response_type: ResponseType, **kwargs) -> Dict[str, Any]:
        """
        Format a response using the appropriate template.
        
        Args:
            response_type: Type of response to format
            **kwargs: Arguments specific to the response type
            
        Returns:
            Formatted response dictionary
        """
        template = self.templates.get(response_type)
        if not template:
            raise ValueError(f"No template available for response type: {response_type}")
            
        return template.generate(**kwargs)
        
    def format_as_text(self, response: Dict[str, Any]) -> str:
        """
        Convert structured response to formatted text.
        
        Args:
            response: Structured response dictionary
            
        Returns:
            Human-readable text format
        """
        response_type = ResponseType(response.get("response_type"))
        
        if response_type == ResponseType.RESEARCH_SUMMARY:
            return self._format_research_summary_text(response)
        elif response_type == ResponseType.DETAILED_ANALYSIS:
            return self._format_detailed_analysis_text(response)
        elif response_type == ResponseType.API_RESULT:
            return self._format_api_result_text(response)
        elif response_type == ResponseType.ERROR_RESPONSE:
            return self._format_error_response_text(response)
        elif response_type == ResponseType.PROGRESS_UPDATE:
            return self._format_progress_update_text(response)
        else:
            return json.dumps(response, indent=2)
            
    def _format_research_summary_text(self, response: Dict[str, Any]) -> str:
        """Format research summary as readable text."""
        lines = [
            f"# Research Summary",
            f"**Query:** {response['query']['original']}",
            f"**Confidence:** {response['summary']['confidence']['description']}",
            "",
            "## Overview",
            response['summary']['overview'],
            "",
            "## Key Findings"
        ]
        
        for i, finding in enumerate(response['summary']['key_findings'], 1):
            lines.append(f"{i}. {finding['finding']}")
            
        lines.extend([
            "",
            f"## Sources ({response['sources']['total_consulted']} total)",
            f"- High credibility: {response['sources']['by_credibility']['high']}",
            f"- Medium credibility: {response['sources']['by_credibility']['medium']}",
            f"- Low credibility: {response['sources']['by_credibility']['low']}"
        ])
        
        if response.get('limitations'):
            lines.extend(["", "## Limitations"])
            for limitation in response['limitations']:
                lines.append(f"- {limitation['description']}")
                
        return "\n".join(lines)
        
    def _format_detailed_analysis_text(self, response: Dict[str, Any]) -> str:
        """Format detailed analysis as readable text."""
        lines = [
            f"# Detailed Analysis",
            f"**Query:** {response['query']['original']}",
            f"**Research Type:** {response['query']['research_type']}",
            "",
            "## Methodology",
            f"**Approach:** {response['methodology']['approach']}"
        ]
        
        if response['methodology'].get('search_strategy'):
            lines.append(f"**Search Strategy:** {', '.join(response['methodology']['search_strategy'])}")
            
        lines.extend(["", "## Findings"])
        
        # Format findings (structure depends on the specific findings format)
        findings = response.get('detailed_findings', {})
        for category, content in findings.items():
            lines.append(f"### {category.replace('_', ' ').title()}")
            if isinstance(content, list):
                for item in content:
                    lines.append(f"- {item}")
            else:
                lines.append(str(content))
            lines.append("")
            
        return "\n".join(lines)
        
    def _format_api_result_text(self, response: Dict[str, Any]) -> str:
        """Format API result as readable text."""
        lines = [
            f"# API Result",
            f"**Action:** {response['action_taken']['description']}",
            f"**API:** {response['action_taken']['api_name']}",
            f"**Success:** {'Yes' if response['results']['success'] else 'No'}",
            ""
        ]
        
        if response['results']['success']:
            lines.extend([
                "## Results Summary",
                f"**Type:** {response['results']['data_summary'].get('type', 'unknown')}",
                f"**Count:** {response['metadata']['result_count']}"
            ])
            
            if response['results']['data_summary'].get('preview'):
                lines.extend([
                    "",
                    "## Preview",
                    response['results']['data_summary']['preview']
                ])
        else:
            lines.extend([
                "## Errors",
                *[f"- {error.get('message', 'Unknown error')}" for error in response.get('errors', [])]
            ])
            
        lines.extend([
            "",
            "## Integration Context",
            response['integration']['context']
        ])
        
        if response.get('next_steps'):
            lines.extend([
                "",
                "## Next Steps",
                *[f"- {step}" for step in response['next_steps']]
            ])
            
        return "\n".join(lines)
        
    def _format_error_response_text(self, response: Dict[str, Any]) -> str:
        """Format error response as readable text."""
        lines = [
            f"# Error Response",
            f"**Type:** {response['error']['type']}",
            f"**Severity:** {response['error']['severity']}",
            "",
            "## Error Details",
            response['error']['message']
        ]
        
        if response.get('alternative_approaches'):
            lines.extend([
                "",
                "## Alternative Approaches",
                *[f"- {approach}" for approach in response['alternative_approaches']]
            ])
            
        if response.get('recovery_suggestions'):
            lines.extend([
                "",
                "## Recovery Suggestions",
                *[f"- {suggestion}" for suggestion in response['recovery_suggestions']]
            ])
            
        if response.get('next_actions'):
            lines.extend([
                "",
                "## Recommended Actions",
                *[f"- {action}" for action in response['next_actions']]
            ])
            
        return "\n".join(lines)
        
    def _format_progress_update_text(self, response: Dict[str, Any]) -> str:
        """Format progress update as readable text."""
        progress = response['task']['progress_percentage']
        
        lines = [
            f"# Progress Update",
            f"**Task:** {response['task']['id']}",
            f"**Progress:** {progress:.1f}%",
            f"**Current Step:** {response['task']['current_step']}"
        ]
        
        if response['timing']['estimated_remaining_minutes']:
            lines.append(f"**Est. Remaining:** {response['timing']['estimated_remaining_minutes']} minutes")
            
        if response['intermediate_results']['count'] > 0:
            lines.extend([
                "",
                f"## Intermediate Results ({response['intermediate_results']['count']})",
                f"Found: {', '.join(response['intermediate_results']['summary'].get('types', []))}"
            ])
            
        if response['issues']['count'] > 0:
            lines.extend([
                "",
                f"## Issues Encountered ({response['issues']['count']})",
                *[f"- {issue}" for issue in response['issues']['list']]
            ])
            
        return "\n".join(lines)


# Example usage
def main():
    """Example usage of response templates."""
    formatter = ResponseFormatter()
    
    # Example research summary
    sources = [
        Source("AI in Healthcare", "https://example.com/ai-health", "2024-01-15", "Dr. Smith", 0.9, "academic"),
        Source("Medical AI Applications", "https://example.com/med-ai", "2023-12-10", "Jane Doe", 0.7, "news")
    ]
    
    limitations = [
        LimitationNote("temporal", "Limited to English-language sources", "medium", ["Include multilingual sources"])
    ]
    
    follow_ups = [
        FollowUpSuggestion("Regulatory Analysis", "Examine regulatory frameworks", "high", "medium", ["FDA.gov", "EMA.europa.eu"])
    ]
    
    summary_response = formatter.format_response(
        ResponseType.RESEARCH_SUMMARY,
        query="AI applications in healthcare",
        key_findings=["AI shows promise in diagnostic imaging", "Regulatory challenges remain significant"],
        sources=sources,
        confidence_level=0.8,
        limitations=limitations,
        follow_ups=follow_ups
    )
    
    print("Structured Response:")
    print(json.dumps(summary_response, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    print("Text Format:")
    print(formatter.format_as_text(summary_response))


if __name__ == "__main__":
    main()