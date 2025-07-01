# Research Agent 2 - Usage Patterns and Examples

This guide provides comprehensive examples of how to effectively use Research Agent 2 for various research scenarios, API integrations, and continuous learning patterns.

## Core Research Patterns

### Basic Research Query

**Example**: "Research the latest developments in quantum computing"

#### Process Flow:
1. **Query Analysis**: Identify research type (current events + technical)
2. **Search Strategy**: Target recent publications and industry announcements
3. **Source Evaluation**: Prioritize academic papers, tech company announcements, credible news
4. **Information Synthesis**: Organize by categories (hardware, software, applications)
5. **Citation Generation**: Provide proper academic citations for all sources

#### Implementation:
```python
import asyncio
from enhanced_research_agent import conduct_research

async def quantum_computing_research():
    result = await conduct_research(
        query="Research the latest developments in quantum computing",
        research_type="comprehensive",
        max_sources=15,
        citation_style="apa"
    )
    
    return result

# Run the research
result = asyncio.run(quantum_computing_research())
print(f"Confidence: {result.get('confidence_level', 0):.2f}")
print(f"Sources: {result.get('sources_found', 0)}")
```

#### Expected Output Structure:
```json
{
  "response_type": "detailed_analysis",
  "query": {
    "original": "Research the latest developments in quantum computing",
    "research_type": "comprehensive"
  },
  "detailed_findings": {
    "hardware_advances": "Recent breakthroughs in quantum processors...",
    "software_developments": "New quantum programming frameworks...",
    "commercial_applications": "Enterprise quantum computing solutions..."
  },
  "sources": {
    "total_consulted": 15,
    "by_credibility": {
      "high": 8,
      "medium": 5,
      "low": 2
    }
  }
}
```

## Advanced Research Patterns

### Domain-Specific Research with Time Constraints

**Example**: "Latest medical AI breakthroughs in 2024"

```python
from enhanced_research_agent import EnhancedResearchAgent, ResearchRequest

async def medical_ai_research():
    agent = EnhancedResearchAgent()
    
    request = ResearchRequest(
        query="Latest medical AI breakthroughs in 2024",
        session_id="medical_research_session",
        research_type="current_events",
        max_sources=20,
        time_constraints={"year": "2024", "recent": True},
        domain_filters=["medical", "healthcare", "ai"],
        quality_threshold=0.7,
        preferred_citation_style="apa"
    )
    
    result = await agent.conduct_enhanced_research(request)
    return result
```

### Comparative Analysis Research

**Example**: "Compare renewable energy sources efficiency 2024"

```python
async def comparative_energy_research():
    request = ResearchRequest(
        query="Compare renewable energy sources efficiency 2024",
        research_type="comparative",
        max_sources=25,
        domain_filters=["energy", "environment", "technology"],
        quality_threshold=0.8
    )
    
    agent = EnhancedResearchAgent()
    result = await agent.conduct_enhanced_research(request)
    
    # Extract comparison data
    if result.get('response_type') == 'detailed_analysis':
        findings = result.get('detailed_findings', {})
        return {
            'solar_efficiency': findings.get('solar_energy', {}),
            'wind_efficiency': findings.get('wind_energy', {}),
            'hydro_efficiency': findings.get('hydro_energy', {}),
            'comparison_summary': findings.get('efficiency_comparison', '')
        }
```

## API Integration Patterns

### Weather Service Integration Example

**API Documentation**: `GET /weather?location={city}&units={metric/imperial}`

#### Step 1: API Learning Process

```python
from function_calling_framework import FunctionSpec, ParameterSpec, ParameterType
from enhanced_research_agent import EnhancedResearchAgent

# Define the weather API function
async def get_weather_data(location: str, units: str = "metric") -> dict:
    """Get weather data for a specific location."""
    import aiohttp
    
    url = f"https://api.weather.service/weather"
    params = {"location": location, "units": units}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "location": data.get("location"),
                    "temperature": data.get("temperature"),
                    "humidity": data.get("humidity"),
                    "conditions": data.get("conditions"),
                    "units": units
                }
            else:
                raise Exception(f"Weather API failed: {response.status}")

# Register with the function calling framework
agent = EnhancedResearchAgent()

weather_spec = FunctionSpec(
    name="get_weather_data",
    description="Get current weather data for a location",
    parameters=[
        ParameterSpec("location", ParameterType.REQUIRED, str, 
                     description="City name or coordinates"),
        ParameterSpec("units", ParameterType.OPTIONAL, str, "metric",
                     description="Temperature units (metric/imperial)",
                     validation_rules=["regex:^(metric|imperial)$"])
    ],
    return_type=dict,
    examples=[
        {"location": "London", "units": "metric"},
        {"location": "New York", "units": "imperial"}
    ],
    tags=["weather", "environment", "data"]
)

agent.function_framework.register_function(get_weather_data, weather_spec)
```

#### Step 2: Integration with Research Queries

```python
async def location_based_research():
    """Research that incorporates weather data for context."""
    
    # Research query about agriculture
    query = "Impact of climate on crop yields in California"
    
    # Conduct primary research
    result = await agent.conduct_enhanced_research(
        ResearchRequest(
            query=query,
            research_type="comprehensive",
            max_sources=15
        )
    )
    
    # Enhance with current weather data
    weather_result = await agent.call_specific_function(
        "get_weather_data",
        {"location": "California", "units": "metric"},
        context="Agricultural research context"
    )
    
    # Combine results
    enhanced_result = {
        "research_findings": result,
        "current_conditions": weather_result.result if weather_result.success else None,
        "integration_notes": "Weather data provides current context for climate impact analysis"
    }
    
    return enhanced_result
```

#### Step 3: Learning and Optimization

```python
# The system automatically learns from API usage
async def demonstrate_learning():
    # Multiple API calls to learn patterns
    locations = ["London", "Tokyo", "New York", "Sydney"]
    
    for location in locations:
        result = await agent.call_specific_function(
            "get_weather_data",
            {"location": location, "units": "metric"}
        )
        
        # System automatically records:
        # - Success/failure rates
        # - Response times
        # - Parameter patterns
        # - Error conditions
    
    # Get learning insights
    learning_report = agent.get_learning_report(days_back=7)
    
    api_learning = learning_report.get('api_usage_patterns', {})
    weather_learning = api_learning.get('get_weather_data', {})
    
    print(f"Weather API Success Rate: {weather_learning.get('success_rate', 0):.2%}")
    print(f"Average Response Time: {weather_learning.get('avg_response_time', 0):.2f}s")
    print(f"Optimal Parameters: {weather_learning.get('optimal_params', {})}")
```

## Continuous Improvement Patterns

### Learning from Successful Research Patterns

```python
async def demonstrate_pattern_learning():
    agent = EnhancedResearchAgent()
    
    # Conduct multiple research queries in the same domain
    tech_queries = [
        "artificial intelligence applications in healthcare",
        "machine learning in drug discovery",
        "AI-powered medical imaging breakthroughs"
    ]
    
    for query in tech_queries:
        result = await agent.conduct_enhanced_research(
            ResearchRequest(
                query=query,
                research_type="comprehensive",
                max_sources=10,
                domain_filters=["healthcare", "technology", "ai"]
            )
        )
        
        # System learns:
        # - Effective keywords for healthcare+AI research
        # - Best source types for this domain
        # - Optimal search strategies
        # - User preferences for this topic area
    
    # Get adaptive recommendations for new query
    recommendations = await agent.get_function_recommendations(
        "AI in medical diagnosis accuracy",
        context="healthcare technology research"
    )
    
    return recommendations
```

### Domain Expertise Development

```python
async def domain_expertise_example():
    agent = EnhancedResearchAgent()
    
    # System tracks domain-specific patterns
    domains = {
        "technology": {
            "preferred_sources": ["arxiv.org", "ieee.org", "acm.org"],
            "key_terms": ["innovation", "development", "breakthrough"],
            "citation_style": "ieee"
        },
        "healthcare": {
            "preferred_sources": ["pubmed.ncbi.nlm.nih.gov", "who.int"],
            "key_terms": ["clinical", "treatment", "patient"],
            "citation_style": "ama"
        },
        "business": {
            "preferred_sources": ["harvard.edu", "wharton.upenn.edu"],
            "key_terms": ["strategy", "market", "revenue"],
            "citation_style": "harvard"
        }
    }
    
    # System adapts search strategy based on detected domain
    query = "blockchain applications in supply chain management"
    
    # Detected domains: technology + business
    adapted_approach = agent.learning_framework.adapt_search_approach(
        query, 
        context={
            "detected_domains": ["technology", "business"],
            "user_expertise_level": "intermediate"
        }
    )
    
    return adapted_approach
```

### Performance Optimization Learning

```python
async def performance_optimization():
    agent = EnhancedResearchAgent()
    
    # System tracks and optimizes:
    performance_metrics = {
        "response_time_optimization": {
            "concurrent_searches": "Learn optimal parallelization",
            "source_prioritization": "Rank sources by relevance and speed",
            "caching_strategies": "Cache frequently accessed information"
        },
        "accuracy_improvement": {
            "source_reliability": "Learn which sources provide accurate info",
            "cross_validation": "Identify patterns in conflicting information",
            "confidence_calibration": "Improve confidence level accuracy"
        },
        "user_satisfaction": {
            "preference_learning": "Adapt to user research preferences",
            "result_formatting": "Optimize presentation based on feedback",
            "follow_up_suggestions": "Improve relevance of suggested research"
        }
    }
    
    # Get current performance metrics
    current_metrics = agent.get_performance_metrics()
    
    return {
        "optimization_areas": performance_metrics,
        "current_performance": current_metrics,
        "improvement_suggestions": [
            "Increase concurrent API calls for faster research",
            "Improve source credibility filtering",
            "Enhance domain-specific search strategies"
        ]
    }
```

## Interactive Usage Patterns

### CLI Interactive Mode

```bash
# Start interactive mode
python main.py --interactive

# Example session:
> research latest developments in quantum computing
Enhanced Mode: Yes
Response Type: detailed_analysis
Research Approach: Enhanced multi-framework research
Sources Analyzed: 15
Confidence: 0.85

> search renewable energy efficiency comparison
Found 12 results for renewable energy efficiency comparison
1. Solar Panel Efficiency Trends 2024 (nature.com)
2. Wind Turbine Technology Advances (ieee.org)
...

> bibliography
# Bibliography (APA)
Generated: 2024-01-15 14:30:22
Confidence Level: 0.85

## Sources
1. Smith, J. (2024). Quantum Computing Breakthroughs. Nature Physics...
2. Johnson, M. (2024). Renewable Energy Efficiency Report. IEEE Transactions...
```

### Programmatic Usage with Session Management

```python
async def session_based_research():
    agent = EnhancedResearchAgent()
    
    # Create persistent session
    session_id = "research_project_2024"
    
    # Sequential research building on previous context
    queries = [
        "Overview of renewable energy technologies",
        "Solar energy efficiency improvements 2024", 
        "Wind energy cost reduction strategies",
        "Integration challenges for renewable energy grids"
    ]
    
    results = []
    for query in queries:
        request = ResearchRequest(
            query=query,
            session_id=session_id,
            research_type="comprehensive"
        )
        
        result = await agent.conduct_enhanced_research(request)
        results.append(result)
        
        # System learns from session context:
        # - User's research progression
        # - Domain focus areas
        # - Preferred information depth
        # - Citation requirements
    
    # Generate comprehensive report from session
    session_report = {
        "session_id": session_id,
        "total_queries": len(queries),
        "research_progression": [r.get('query') for r in results],
        "cumulative_sources": sum(r.get('sources_found', 0) for r in results),
        "learning_insights": agent.get_learning_report(days_back=1)
    }
    
    return session_report
```

## Best Practices and Guidelines

### 1. Query Formulation Best Practices

```python
# Good: Specific, actionable queries
good_queries = [
    "Latest FDA-approved cancer treatments in 2024",
    "Comparison of electric vehicle battery technologies",
    "Machine learning applications in financial fraud detection"
]

# Avoid: Vague or overly broad queries
avoid_queries = [
    "Tell me about technology",
    "What is science?",
    "Research everything about AI"
]
```

### 2. Source Quality Optimization

```python
# Configure for high-quality research
request = ResearchRequest(
    query="Climate change impact on agriculture",
    quality_threshold=0.8,  # High quality sources only
    domain_filters=["academic", "government", "research"],
    preferred_citation_style="apa",
    max_sources=20  # More sources for better coverage
)
```

### 3. Learning Optimization

```python
# Enable all learning features
request = ResearchRequest(
    query="Artificial intelligence ethics frameworks",
    include_learning=True,  # Enable learning from this query
    safety_check=True,      # Ensure ethical compliance
    session_id="ethics_research_session"  # Track session context
)
```

### 4. Error Handling and Recovery

```python
async def robust_research():
    try:
        result = await agent.conduct_enhanced_research(request)
        
        if not result.get('enhanced'):
            print("Using fallback basic research mode")
            
        return result
        
    except Exception as e:
        print(f"Research failed: {e}")
        
        # Fallback to simpler query
        simplified_request = ResearchRequest(
            query="simplified version of query",
            max_sources=5,
            research_type="basic"
        )
        
        return await agent.conduct_enhanced_research(simplified_request)
```

## Summary

Research Agent 2 is designed to be a powerful, adaptive research assistant that:

- **Learns continuously** from usage patterns and outcomes
- **Adapts strategies** based on domain expertise and user preferences  
- **Ensures quality** through comprehensive source evaluation and validation
- **Maintains ethics** with built-in safety and privacy protections
- **Provides transparency** with clear citations and confidence levels
- **Optimizes performance** through intelligent caching and parallel processing

The key to effective usage is:
1. **Clear, specific queries** that define research scope
2. **Appropriate quality thresholds** for the research context
3. **Consistent session usage** to enable learning and adaptation
4. **Regular review** of learning reports and performance metrics
5. **Ethical awareness** of information limitations and uncertainties

Remember: Always verify critical information from primary sources and understand the limitations of automated research tools.