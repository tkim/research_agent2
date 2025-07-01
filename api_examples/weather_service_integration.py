"""
Weather Service API Integration Example

This module demonstrates how to integrate a new API service into Research Agent 2,
using weather data as a practical example. It shows the complete learning process
from API discovery to optimization.
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from function_calling_framework import (
    FunctionSpec, ParameterSpec, ParameterType, FunctionCallFramework
)
from enhanced_research_agent import EnhancedResearchAgent, ResearchRequest


@dataclass
class WeatherData:
    """Structured weather data."""
    location: str
    temperature: float
    humidity: float
    conditions: str
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[str] = None
    units: str = "metric"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class WeatherServiceAPI:
    """
    Weather Service API client demonstrating API integration learning process.
    
    This class shows how to:
    1. Understand API parameters and responses
    2. Implement error handling and retries
    3. Learn optimal usage patterns
    4. Integrate with research workflows
    """
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.weather.service"):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(f"{__name__}.WeatherServiceAPI")
        
        # Learning metrics
        self.call_history: List[Dict[str, Any]] = []
        self.error_patterns: List[Dict[str, Any]] = []
        self.optimal_parameters: Dict[str, Any] = {}
        
    async def get_weather(self, location: str, units: str = "metric", 
                         include_forecast: bool = False) -> WeatherData:
        """
        Get current weather data for a location.
        
        API Documentation Learning Process:
        - GET /weather?location={city}&units={metric/imperial}
        - Optional: &forecast=true for extended data
        - Returns: JSON with temperature, humidity, conditions
        
        Args:
            location: City name, coordinates, or postal code
            units: Temperature units (metric/imperial)
            include_forecast: Include forecast data
            
        Returns:
            WeatherData object with current conditions
        """
        start_time = datetime.now()
        
        try:
            # Validate parameters (learning from documentation)
            if not location:
                raise ValueError("Location parameter is required")
                
            if units not in ["metric", "imperial"]:
                raise ValueError("Units must be 'metric' or 'imperial'")
            
            # Build request
            params = {
                "location": location,
                "units": units
            }
            
            if include_forecast:
                params["forecast"] = "true"
                
            if self.api_key:
                params["api_key"] = self.api_key
                
            # Make API call
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/weather"
                
                async with session.get(url, params=params) as response:
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    if response.status == 200:
                        data = await response.json()
                        weather_data = self._parse_weather_response(data, units)
                        
                        # Record successful call for learning
                        self._record_successful_call(location, units, response_time, data)
                        
                        return weather_data
                        
                    elif response.status == 429:
                        # Rate limit - learn optimal timing
                        error_info = {
                            "type": "rate_limit",
                            "status": response.status,
                            "parameters": params,
                            "timestamp": datetime.now()
                        }
                        self.error_patterns.append(error_info)
                        raise Exception("Rate limit exceeded - reduce request frequency")
                        
                    elif response.status == 401:
                        # Authentication error
                        error_info = {
                            "type": "authentication",
                            "status": response.status,
                            "message": "Invalid or missing API key"
                        }
                        self.error_patterns.append(error_info)
                        raise Exception("Authentication failed - check API key")
                        
                    else:
                        # Other errors
                        error_text = await response.text()
                        error_info = {
                            "type": "api_error",
                            "status": response.status,
                            "message": error_text,
                            "parameters": params
                        }
                        self.error_patterns.append(error_info)
                        raise Exception(f"Weather API error {response.status}: {error_text}")
                        
        except aiohttp.ClientError as e:
            # Network errors
            error_info = {
                "type": "network_error",
                "message": str(e),
                "parameters": {"location": location, "units": units}
            }
            self.error_patterns.append(error_info)
            raise Exception(f"Network error: {e}")
            
        except Exception as e:
            # Record failure for learning
            self._record_failed_call(location, units, str(e))
            raise
            
    def _parse_weather_response(self, data: Dict[str, Any], units: str) -> WeatherData:
        """
        Parse API response into structured weather data.
        
        Learning Process:
        - Understand response structure through testing
        - Handle missing optional fields gracefully  
        - Extract relevant data for research context
        """
        try:
            return WeatherData(
                location=data.get("location", "Unknown"),
                temperature=float(data.get("temperature", 0)),
                humidity=float(data.get("humidity", 0)),
                conditions=data.get("conditions", "Unknown"),
                pressure=data.get("pressure"),
                wind_speed=data.get("wind", {}).get("speed"),
                wind_direction=data.get("wind", {}).get("direction"),
                units=units
            )
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error parsing weather response: {e}")
            # Return basic weather data with available information
            return WeatherData(
                location=data.get("location", "Unknown"),
                temperature=0,
                humidity=0,
                conditions="Data unavailable",
                units=units
            )
            
    def _record_successful_call(self, location: str, units: str, 
                              response_time: float, data: Dict[str, Any]):
        """Record successful API call for learning."""
        call_record = {
            "timestamp": datetime.now(),
            "location": location,
            "units": units,
            "response_time": response_time,
            "data_quality": self._assess_data_quality(data),
            "success": True
        }
        
        self.call_history.append(call_record)
        self._update_optimal_parameters(location, units, response_time)
        
    def _record_failed_call(self, location: str, units: str, error: str):
        """Record failed API call for learning."""
        call_record = {
            "timestamp": datetime.now(),
            "location": location, 
            "units": units,
            "error": error,
            "success": False
        }
        
        self.call_history.append(call_record)
        
    def _assess_data_quality(self, data: Dict[str, Any]) -> float:
        """Assess the quality/completeness of weather data."""
        required_fields = ["location", "temperature", "humidity", "conditions"]
        optional_fields = ["pressure", "wind"]
        
        score = 0.0
        
        # Check required fields
        for field in required_fields:
            if field in data and data[field] is not None:
                score += 0.2
                
        # Check optional fields
        for field in optional_fields:
            if field in data and data[field] is not None:
                score += 0.1
                
        return min(1.0, score)
        
    def _update_optimal_parameters(self, location: str, units: str, response_time: float):
        """Update optimal parameters based on successful calls."""
        # Learn location patterns
        if "locations" not in self.optimal_parameters:
            self.optimal_parameters["locations"] = {}
            
        location_key = location.lower()
        if location_key not in self.optimal_parameters["locations"]:
            self.optimal_parameters["locations"][location_key] = {
                "success_count": 0,
                "avg_response_time": 0.0,
                "preferred_format": location
            }
            
        loc_data = self.optimal_parameters["locations"][location_key]
        loc_data["success_count"] += 1
        
        # Update average response time
        current_avg = loc_data["avg_response_time"]
        count = loc_data["success_count"]
        loc_data["avg_response_time"] = ((current_avg * (count - 1)) + response_time) / count
        
        # Learn unit preferences
        if "preferred_units" not in self.optimal_parameters:
            self.optimal_parameters["preferred_units"] = {}
            
        if units not in self.optimal_parameters["preferred_units"]:
            self.optimal_parameters["preferred_units"][units] = 0
        self.optimal_parameters["preferred_units"][units] += 1
        
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from API usage learning."""
        if not self.call_history:
            return {"message": "No API calls recorded yet"}
            
        successful_calls = [call for call in self.call_history if call["success"]]
        failed_calls = [call for call in self.call_history if not call["success"]]
        
        insights = {
            "total_calls": len(self.call_history),
            "success_rate": len(successful_calls) / len(self.call_history),
            "average_response_time": sum(call.get("response_time", 0) for call in successful_calls) / len(successful_calls) if successful_calls else 0,
            "error_patterns": self._analyze_error_patterns(),
            "optimal_parameters": self.optimal_parameters,
            "recommendations": self._generate_recommendations()
        }
        
        return insights
        
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in API errors."""
        if not self.error_patterns:
            return {"message": "No errors recorded"}
            
        error_types = [error["type"] for error in self.error_patterns]
        error_counts = {}
        for error_type in error_types:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
        return {
            "most_common_errors": error_counts,
            "recent_errors": self.error_patterns[-5:],  # Last 5 errors
            "error_frequency": len(self.error_patterns) / len(self.call_history) if self.call_history else 0
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on learning."""
        recommendations = []
        
        # Success rate recommendations
        success_rate = len([call for call in self.call_history if call["success"]]) / len(self.call_history) if self.call_history else 0
        
        if success_rate < 0.8:
            recommendations.append("Consider reviewing API parameters - success rate below optimal")
            
        # Response time recommendations
        successful_calls = [call for call in self.call_history if call["success"]]
        if successful_calls:
            avg_time = sum(call.get("response_time", 0) for call in successful_calls) / len(successful_calls)
            if avg_time > 2.0:
                recommendations.append("API response times are slow - consider caching frequently requested locations")
                
        # Error pattern recommendations
        error_types = [error["type"] for error in self.error_patterns]
        if "rate_limit" in error_types:
            recommendations.append("Implement request throttling to avoid rate limits")
        if "authentication" in error_types:
            recommendations.append("Verify API key configuration")
            
        return recommendations if recommendations else ["API usage patterns are optimal"]


class WeatherResearchIntegration:
    """
    Integration of weather data into research workflows.
    
    This demonstrates how to:
    1. Register weather functions with the research agent
    2. Use weather data to enhance research context
    3. Learn from integrated usage patterns
    """
    
    def __init__(self, weather_api: WeatherServiceAPI):
        self.weather_api = weather_api
        self.agent = EnhancedResearchAgent()
        self.logger = logging.getLogger(f"{__name__}.WeatherResearchIntegration")
        
        # Register weather functions
        self._register_weather_functions()
        
    def _register_weather_functions(self):
        """Register weather-related functions with the research agent."""
        
        # Basic weather function
        async def get_weather_data(location: str, units: str = "metric") -> Dict[str, Any]:
            """Get current weather data for research context."""
            try:
                weather_data = await self.weather_api.get_weather(location, units)
                return {
                    "location": weather_data.location,
                    "temperature": weather_data.temperature,
                    "humidity": weather_data.humidity,
                    "conditions": weather_data.conditions,
                    "units": weather_data.units,
                    "timestamp": weather_data.timestamp.isoformat()
                }
            except Exception as e:
                return {"error": str(e), "location": location}
                
        # Enhanced weather function with research integration
        async def get_weather_for_research(location: str, research_context: str = "") -> Dict[str, Any]:
            """Get weather data with research context enhancement."""
            weather_data = await get_weather_data(location)
            
            if "error" not in weather_data:
                # Add research-relevant context
                research_enhancement = {
                    "agriculture_relevance": self._assess_agriculture_relevance(weather_data),
                    "energy_relevance": self._assess_energy_relevance(weather_data),
                    "health_relevance": self._assess_health_relevance(weather_data),
                    "research_context": research_context
                }
                weather_data["research_enhancement"] = research_enhancement
                
            return weather_data
            
        # Register functions with specifications
        weather_spec = FunctionSpec(
            name="get_weather_data",
            description="Get current weather data for a specific location",
            parameters=[
                ParameterSpec("location", ParameterType.REQUIRED, str,
                            description="City name, coordinates, or postal code"),
                ParameterSpec("units", ParameterType.OPTIONAL, str, "metric",
                            description="Temperature units (metric/imperial)",
                            validation_rules=["regex:^(metric|imperial)$"])
            ],
            return_type=Dict[str, Any],
            examples=[
                {"location": "London", "units": "metric"},
                {"location": "New York", "units": "imperial"},
                {"location": "37.7749,-122.4194", "units": "metric"}  # Coordinates
            ],
            tags=["weather", "environment", "data", "context"]
        )
        
        enhanced_weather_spec = FunctionSpec(
            name="get_weather_for_research", 
            description="Get weather data enhanced for research context",
            parameters=[
                ParameterSpec("location", ParameterType.REQUIRED, str,
                            description="Location for weather data"),
                ParameterSpec("research_context", ParameterType.OPTIONAL, str, "",
                            description="Research context for enhanced relevance")
            ],
            return_type=Dict[str, Any],
            examples=[
                {"location": "California", "research_context": "agricultural impact study"},
                {"location": "Texas", "research_context": "renewable energy analysis"}
            ],
            tags=["weather", "research", "context", "enhanced"]
        )
        
        # Register with the agent's function framework
        self.agent.function_framework.register_function(get_weather_data, weather_spec)
        self.agent.function_framework.register_function(get_weather_for_research, enhanced_weather_spec)
        
        self.logger.info("Weather functions registered with research agent")
        
    def _assess_agriculture_relevance(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess relevance of weather data for agricultural research."""
        temp = weather_data.get("temperature", 0)
        humidity = weather_data.get("humidity", 0)
        conditions = weather_data.get("conditions", "").lower()
        
        relevance = {
            "growing_conditions": "optimal" if 15 <= temp <= 25 and humidity > 40 else "suboptimal",
            "irrigation_needs": "high" if humidity < 30 else "moderate" if humidity < 60 else "low",
            "pest_risk": "high" if 20 <= temp <= 30 and humidity > 70 else "moderate",
            "harvest_suitability": "good" if "clear" in conditions or "sunny" in conditions else "poor"
        }
        
        return relevance
        
    def _assess_energy_relevance(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess relevance of weather data for energy research."""
        temp = weather_data.get("temperature", 0)
        conditions = weather_data.get("conditions", "").lower()
        
        relevance = {
            "solar_potential": "high" if "sunny" in conditions or "clear" in conditions else "low",
            "wind_potential": "data_needed",  # Would need wind speed data
            "cooling_demand": "high" if temp > 25 else "moderate" if temp > 15 else "low",
            "heating_demand": "high" if temp < 10 else "moderate" if temp < 20 else "low"
        }
        
        return relevance
        
    def _assess_health_relevance(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess relevance of weather data for health research."""
        temp = weather_data.get("temperature", 0)
        humidity = weather_data.get("humidity", 0)
        
        relevance = {
            "heat_stress_risk": "high" if temp > 35 else "moderate" if temp > 30 else "low",
            "respiratory_conditions": "concern" if humidity > 80 or humidity < 20 else "normal",
            "disease_vector_activity": "high" if 20 <= temp <= 30 and humidity > 60 else "low",
            "air_quality_factors": "monitor" if temp > 30 else "normal"
        }
        
        return relevance
        
    async def demonstrate_integrated_research(self) -> Dict[str, Any]:
        """Demonstrate weather data integration in research workflows."""
        examples = []
        
        # Example 1: Climate impact on agriculture
        agriculture_research = await self.agent.conduct_enhanced_research(
            ResearchRequest(
                query="Impact of climate change on crop yields in California",
                session_id="weather_integration_demo",
                research_type="comprehensive",
                max_sources=10
            )
        )
        
        # Enhance with current weather context
        california_weather = await self.agent.call_specific_function(
            "get_weather_for_research",
            {"location": "California", "research_context": "agricultural impact study"}
        )
        
        examples.append({
            "research_type": "agriculture_climate",
            "research_result": agriculture_research,
            "weather_context": california_weather.result if california_weather.success else None,
            "integration_value": "Current weather provides context for climate impact analysis"
        })
        
        # Example 2: Renewable energy potential
        energy_research = await self.agent.conduct_enhanced_research(
            ResearchRequest(
                query="Solar energy potential in desert regions",
                session_id="weather_integration_demo",
                research_type="comprehensive",
                max_sources=8
            )
        )
        
        # Enhance with weather data for multiple desert locations
        desert_locations = ["Phoenix", "Las Vegas", "Albuquerque"]
        weather_contexts = []
        
        for location in desert_locations:
            weather_result = await self.agent.call_specific_function(
                "get_weather_for_research",
                {"location": location, "research_context": "solar energy analysis"}
            )
            if weather_result.success:
                weather_contexts.append(weather_result.result)
                
        examples.append({
            "research_type": "renewable_energy",
            "research_result": energy_research,
            "weather_contexts": weather_contexts,
            "integration_value": "Real-time solar conditions inform energy potential analysis"
        })
        
        return {
            "integration_examples": examples,
            "weather_api_learning": self.weather_api.get_learning_insights(),
            "function_usage": self.agent.function_framework.registry.usage_patterns,
            "recommendations": [
                "Weather context significantly enhances location-based research",
                "Integrate multiple weather data points for regional analysis",
                "Consider seasonal patterns for long-term research projects"
            ]
        }


# Example usage and testing
async def main():
    """Demonstrate weather API integration and learning process."""
    print("Weather Service API Integration Demo")
    print("=" * 50)
    
    # Initialize weather API (would use real API key in production)
    weather_api = WeatherServiceAPI(api_key="demo_key")
    
    # Test basic API calls to learn patterns
    test_locations = ["London", "New York", "Tokyo", "Sydney"]
    
    print("Learning API usage patterns...")
    for location in test_locations:
        try:
            weather_data = await weather_api.get_weather(location, "metric")
            print(f"✓ {location}: {weather_data.temperature}°C, {weather_data.conditions}")
        except Exception as e:
            print(f"✗ {location}: {e}")
            
    # Get learning insights
    insights = weather_api.get_learning_insights()
    print(f"\nAPI Learning Insights:")
    print(f"Success Rate: {insights['success_rate']:.2%}")
    print(f"Avg Response Time: {insights['average_response_time']:.2f}s")
    print(f"Recommendations: {insights['recommendations']}")
    
    # Demonstrate research integration
    print("\n" + "=" * 50)
    print("Research Integration Demo")
    
    integration = WeatherResearchIntegration(weather_api)
    demo_results = await integration.demonstrate_integrated_research()
    
    print(f"Integration Examples: {len(demo_results['integration_examples'])}")
    for example in demo_results['integration_examples']:
        print(f"- {example['research_type']}: {example['integration_value']}")
        
    print(f"\nFunction Framework Learning:")
    framework_insights = demo_results['weather_api_learning']
    print(f"Total API Calls: {framework_insights.get('total_calls', 0)}")
    print(f"Success Rate: {framework_insights.get('success_rate', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())