"""
Function Calling Framework for Research Agent 2

This module provides a comprehensive framework for function calling, API management,
parameter validation, and response handling with learning capabilities.
"""

import asyncio
import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from enum import Enum
import traceback


class FunctionCallStatus(Enum):
    """Status of function call execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class ParameterType(Enum):
    """Types of function parameters."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    KEYWORD = "keyword"
    VARIABLE = "variable"


@dataclass
class ParameterSpec:
    """Specification for a function parameter."""
    name: str
    param_type: ParameterType
    data_type: Type
    default_value: Any = None
    description: str = ""
    validation_rules: List[str] = field(default_factory=list)
    examples: List[Any] = field(default_factory=list)


@dataclass
class FunctionSpec:
    """Complete specification for a function or API endpoint."""
    name: str
    description: str
    parameters: List[ParameterSpec]
    return_type: Type
    examples: List[Dict[str, Any]] = field(default_factory=list)
    error_conditions: List[str] = field(default_factory=list)
    rate_limits: Optional[Dict[str, Any]] = None
    timeout: int = 30
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class FunctionCallResult:
    """Result of a function call execution."""
    function_name: str
    status: FunctionCallStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsagePattern:
    """Pattern of successful function usage."""
    function_name: str
    parameter_combinations: List[Dict[str, Any]]
    success_rate: float
    average_execution_time: float
    common_errors: List[str]
    usage_contexts: List[str]
    last_updated: datetime = field(default_factory=datetime.now)


class ParameterValidator:
    """Validates function parameters against specifications."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ParameterValidator")
        
    def validate_parameters(self, parameters: Dict[str, Any], 
                          function_spec: FunctionSpec) -> Tuple[bool, List[str]]:
        """
        Validate parameters against function specification.
        
        Args:
            parameters: Dictionary of parameter values
            function_spec: Function specification
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required parameters
        required_params = {p.name for p in function_spec.parameters 
                         if p.param_type == ParameterType.REQUIRED}
        missing_params = required_params - set(parameters.keys())
        
        if missing_params:
            errors.append(f"Missing required parameters: {', '.join(missing_params)}")
            
        # Check each provided parameter
        for param_name, param_value in parameters.items():
            param_spec = self._find_parameter_spec(param_name, function_spec)
            
            if not param_spec:
                errors.append(f"Unknown parameter: {param_name}")
                continue
                
            # Type validation
            type_errors = self._validate_type(param_value, param_spec)
            errors.extend(type_errors)
            
            # Rule validation
            rule_errors = self._validate_rules(param_value, param_spec)
            errors.extend(rule_errors)
            
        return len(errors) == 0, errors
        
    def _find_parameter_spec(self, param_name: str, 
                           function_spec: FunctionSpec) -> Optional[ParameterSpec]:
        """Find parameter specification by name."""
        for param_spec in function_spec.parameters:
            if param_spec.name == param_name:
                return param_spec
        return None
        
    def _validate_type(self, value: Any, param_spec: ParameterSpec) -> List[str]:
        """Validate parameter type."""
        errors = []
        
        try:
            # Handle None values for optional parameters
            if value is None and param_spec.param_type == ParameterType.OPTIONAL:
                return errors
                
            # Basic type checking
            if not isinstance(value, param_spec.data_type):
                # Try type conversion for common cases
                if param_spec.data_type == int and isinstance(value, str):
                    try:
                        int(value)
                    except ValueError:
                        errors.append(f"Parameter '{param_spec.name}' must be {param_spec.data_type.__name__}")
                elif param_spec.data_type == float and isinstance(value, (int, str)):
                    try:
                        float(value)
                    except ValueError:
                        errors.append(f"Parameter '{param_spec.name}' must be {param_spec.data_type.__name__}")
                else:
                    errors.append(f"Parameter '{param_spec.name}' must be {param_spec.data_type.__name__}")
                    
        except Exception as e:
            errors.append(f"Type validation error for '{param_spec.name}': {e}")
            
        return errors
        
    def _validate_rules(self, value: Any, param_spec: ParameterSpec) -> List[str]:
        """Validate parameter against custom rules."""
        errors = []
        
        for rule in param_spec.validation_rules:
            try:
                # Simple rule evaluation (can be extended)
                if rule.startswith("min_length:"):
                    min_len = int(rule.split(":")[1])
                    if hasattr(value, '__len__') and len(value) < min_len:
                        errors.append(f"Parameter '{param_spec.name}' must have minimum length {min_len}")
                        
                elif rule.startswith("max_length:"):
                    max_len = int(rule.split(":")[1])
                    if hasattr(value, '__len__') and len(value) > max_len:
                        errors.append(f"Parameter '{param_spec.name}' must have maximum length {max_len}")
                        
                elif rule.startswith("range:"):
                    range_parts = rule.split(":")[1].split(",")
                    min_val, max_val = float(range_parts[0]), float(range_parts[1])
                    if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                        errors.append(f"Parameter '{param_spec.name}' must be between {min_val} and {max_val}")
                        
                elif rule.startswith("regex:"):
                    import re
                    pattern = rule.split(":", 1)[1]
                    if isinstance(value, str) and not re.match(pattern, value):
                        errors.append(f"Parameter '{param_spec.name}' must match pattern {pattern}")
                        
            except Exception as e:
                errors.append(f"Rule validation error for '{param_spec.name}': {e}")
                
        return errors


class FunctionRegistry:
    """Registry for managing available functions and their specifications."""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.specifications: Dict[str, FunctionSpec] = {}
        self.usage_patterns: Dict[str, UsagePattern] = {}
        self.logger = logging.getLogger(f"{__name__}.FunctionRegistry")
        
    def register_function(self, func: Callable, spec: FunctionSpec):
        """Register a function with its specification."""
        self.functions[spec.name] = func
        self.specifications[spec.name] = spec
        self.logger.info(f"Registered function: {spec.name}")
        
    def auto_register_function(self, func: Callable, 
                             description: str = "", 
                             tags: List[str] = None) -> FunctionSpec:
        """
        Automatically register a function by inspecting its signature.
        
        Args:
            func: Function to register
            description: Function description
            tags: Function tags
            
        Returns:
            Generated FunctionSpec
        """
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            # Determine parameter type
            if param.default == inspect.Parameter.empty:
                param_type = ParameterType.REQUIRED
                default_value = None
            else:
                param_type = ParameterType.OPTIONAL
                default_value = param.default
                
            # Determine data type
            if param.annotation != inspect.Parameter.empty:
                data_type = param.annotation
            else:
                data_type = type(default_value) if default_value is not None else str
                
            param_spec = ParameterSpec(
                name=param_name,
                param_type=param_type,
                data_type=data_type,
                default_value=default_value,
                description=f"Parameter {param_name}"
            )
            parameters.append(param_spec)
            
        # Determine return type
        return_type = sig.return_annotation if sig.return_annotation != inspect.Parameter.empty else Any
        
        spec = FunctionSpec(
            name=func.__name__,
            description=description or func.__doc__ or f"Function {func.__name__}",
            parameters=parameters,
            return_type=return_type,
            tags=tags or []
        )
        
        self.register_function(func, spec)
        return spec
        
    def get_function(self, name: str) -> Optional[Callable]:
        """Get function by name."""
        return self.functions.get(name)
        
    def get_specification(self, name: str) -> Optional[FunctionSpec]:
        """Get function specification by name."""
        return self.specifications.get(name)
        
    def list_functions(self, tags: List[str] = None) -> List[str]:
        """List available functions, optionally filtered by tags."""
        if not tags:
            return list(self.functions.keys())
            
        filtered_functions = []
        for name, spec in self.specifications.items():
            if any(tag in spec.tags for tag in tags):
                filtered_functions.append(name)
                
        return filtered_functions
        
    def update_usage_pattern(self, function_name: str, 
                           parameters: Dict[str, Any],
                           success: bool, execution_time: float,
                           context: str = ""):
        """Update usage patterns based on function call results."""
        if function_name not in self.usage_patterns:
            self.usage_patterns[function_name] = UsagePattern(
                function_name=function_name,
                parameter_combinations=[],
                success_rate=0.0,
                average_execution_time=0.0,
                common_errors=[],
                usage_contexts=[]
            )
            
        pattern = self.usage_patterns[function_name]
        
        # Update parameter combinations
        param_combo = {k: type(v).__name__ for k, v in parameters.items()}
        if param_combo not in pattern.parameter_combinations:
            pattern.parameter_combinations.append(param_combo)
            
        # Update execution time
        pattern.average_execution_time = (
            (pattern.average_execution_time + execution_time) / 2
        )
        
        # Update context
        if context and context not in pattern.usage_contexts:
            pattern.usage_contexts.append(context)
            
        pattern.last_updated = datetime.now()


class FunctionCallExecutor:
    """Executes function calls with validation, error handling, and retries."""
    
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
        self.validator = ParameterValidator()
        self.logger = logging.getLogger(f"{__name__}.FunctionCallExecutor")
        self.call_history: List[FunctionCallResult] = []
        
    async def execute_function(self, function_name: str, 
                             parameters: Dict[str, Any],
                             context: str = "") -> FunctionCallResult:
        """
        Execute a function with full validation and error handling.
        
        Args:
            function_name: Name of function to execute
            parameters: Function parameters
            context: Context of the function call
            
        Returns:
            FunctionCallResult with execution details
        """
        start_time = datetime.now()
        result = FunctionCallResult(
            function_name=function_name,
            status=FunctionCallStatus.PENDING,
            start_time=start_time,
            parameters=parameters.copy()
        )
        
        try:
            # Step 1: Validate function exists
            func = self.registry.get_function(function_name)
            if not func:
                result.status = FunctionCallStatus.FAILED
                result.error = f"Function '{function_name}' not found"
                return result
                
            spec = self.registry.get_specification(function_name)
            if not spec:
                result.status = FunctionCallStatus.FAILED
                result.error = f"No specification found for '{function_name}'"
                return result
                
            # Step 2: Validate parameters
            is_valid, errors = self.validator.validate_parameters(parameters, spec)
            if not is_valid:
                result.status = FunctionCallStatus.FAILED
                result.error = "Parameter validation failed"
                result.error_details = {"validation_errors": errors}
                return result
                
            # Step 3: Execute with retry logic
            result.status = FunctionCallStatus.IN_PROGRESS
            
            retry_policy = spec.retry_policy
            max_retries = retry_policy.get('max_retries', 3)
            backoff_factor = retry_policy.get('backoff_factor', 1.0)
            
            for attempt in range(max_retries + 1):
                try:
                    # Execute function
                    if asyncio.iscoroutinefunction(func):
                        func_result = await asyncio.wait_for(
                            func(**parameters), 
                            timeout=spec.timeout
                        )
                    else:
                        func_result = func(**parameters)
                        
                    # Success
                    result.status = FunctionCallStatus.SUCCESS
                    result.result = func_result
                    break
                    
                except asyncio.TimeoutError:
                    if attempt == max_retries:
                        result.status = FunctionCallStatus.TIMEOUT
                        result.error = f"Function timed out after {spec.timeout} seconds"
                    else:
                        result.status = FunctionCallStatus.RETRYING
                        result.retry_count = attempt + 1
                        await asyncio.sleep(backoff_factor * (2 ** attempt))
                        
                except Exception as e:
                    if attempt == max_retries:
                        result.status = FunctionCallStatus.FAILED
                        result.error = str(e)
                        result.error_details = {
                            "exception_type": type(e).__name__,
                            "traceback": traceback.format_exc()
                        }
                    else:
                        result.status = FunctionCallStatus.RETRYING
                        result.retry_count = attempt + 1
                        await asyncio.sleep(backoff_factor * (2 ** attempt))
                        
        except Exception as e:
            result.status = FunctionCallStatus.FAILED
            result.error = f"Unexpected error: {e}"
            result.error_details = {
                "exception_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            
        finally:
            # Finalize result
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
            
            # Update usage patterns
            self.registry.update_usage_pattern(
                function_name=function_name,
                parameters=parameters,
                success=(result.status == FunctionCallStatus.SUCCESS),
                execution_time=result.execution_time,
                context=context
            )
            
            # Store in history
            self.call_history.append(result)
            
            # Log result
            if result.status == FunctionCallStatus.SUCCESS:
                self.logger.info(f"Function '{function_name}' executed successfully in {result.execution_time:.2f}s")
            else:
                self.logger.error(f"Function '{function_name}' failed: {result.error}")
                
        return result
        
    def get_call_history(self, function_name: str = None, 
                        limit: int = 100) -> List[FunctionCallResult]:
        """Get function call history."""
        history = self.call_history
        
        if function_name:
            history = [r for r in history if r.function_name == function_name]
            
        return history[-limit:] if limit else history
        
    def get_success_rate(self, function_name: str) -> float:
        """Get success rate for a function."""
        history = self.get_call_history(function_name)
        if not history:
            return 0.0
            
        successful = sum(1 for r in history if r.status == FunctionCallStatus.SUCCESS)
        return successful / len(history)


class FunctionCallFramework:
    """Main framework for managing function calls in Research Agent 2."""
    
    def __init__(self):
        self.registry = FunctionRegistry()
        self.executor = FunctionCallExecutor(self.registry)
        self.logger = logging.getLogger(f"{__name__}.FunctionCallFramework")
        
    def register_function(self, func: Callable, spec: FunctionSpec = None):
        """Register a function with optional specification."""
        if spec:
            self.registry.register_function(func, spec)
        else:
            self.registry.auto_register_function(func)
            
    async def call_function(self, function_name: str, 
                          parameters: Dict[str, Any],
                          context: str = "") -> FunctionCallResult:
        """Call a function with validation and error handling."""
        return await self.executor.execute_function(function_name, parameters, context)
        
    def analyze_available_functions(self) -> Dict[str, Any]:
        """Analyze available functions and their usage patterns."""
        functions = self.registry.list_functions()
        analysis = {
            "total_functions": len(functions),
            "functions_by_tag": {},
            "usage_statistics": {},
            "most_used_functions": [],
            "least_reliable_functions": []
        }
        
        # Group by tags
        for func_name in functions:
            spec = self.registry.get_specification(func_name)
            if spec:
                for tag in spec.tags:
                    if tag not in analysis["functions_by_tag"]:
                        analysis["functions_by_tag"][tag] = []
                    analysis["functions_by_tag"][tag].append(func_name)
                    
        # Usage statistics
        for func_name in functions:
            history = self.executor.get_call_history(func_name)
            if history:
                analysis["usage_statistics"][func_name] = {
                    "total_calls": len(history),
                    "success_rate": self.executor.get_success_rate(func_name),
                    "average_execution_time": sum(r.execution_time for r in history) / len(history)
                }
                
        return analysis
        
    def get_function_recommendations(self, context: str, 
                                   user_query: str) -> List[Tuple[str, float]]:
        """
        Recommend functions based on context and query.
        
        Args:
            context: Current context or task
            user_query: User's query or request
            
        Returns:
            List of (function_name, relevance_score) tuples
        """
        recommendations = []
        query_lower = user_query.lower()
        context_lower = context.lower()
        
        for func_name in self.registry.list_functions():
            spec = self.registry.get_specification(func_name)
            if not spec:
                continue
                
            score = 0.0
            
            # Check description relevance
            if any(word in spec.description.lower() for word in query_lower.split()):
                score += 0.3
                
            # Check tag relevance
            for tag in spec.tags:
                if tag.lower() in query_lower or tag.lower() in context_lower:
                    score += 0.2
                    
            # Check usage patterns
            pattern = self.registry.usage_patterns.get(func_name)
            if pattern:
                for usage_context in pattern.usage_contexts:
                    if usage_context.lower() in context_lower:
                        score += 0.1
                        
                # Boost score for reliable functions
                if pattern.success_rate > 0.8:
                    score += 0.1
                    
            if score > 0:
                recommendations.append((func_name, score))
                
        # Sort by relevance score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
        
    def generate_function_documentation(self) -> str:
        """Generate documentation for all registered functions."""
        docs = ["# Available Functions\n"]
        
        for func_name in sorted(self.registry.list_functions()):
            spec = self.registry.get_specification(func_name)
            if not spec:
                continue
                
            docs.append(f"## {func_name}")
            docs.append(f"\n**Description:** {spec.description}\n")
            
            if spec.tags:
                docs.append(f"**Tags:** {', '.join(spec.tags)}\n")
                
            docs.append("**Parameters:**")
            for param in spec.parameters:
                param_info = f"- `{param.name}` ({param.data_type.__name__})"
                if param.param_type == ParameterType.REQUIRED:
                    param_info += " - **Required**"
                else:
                    param_info += f" - Optional (default: {param.default_value})"
                    
                if param.description:
                    param_info += f" - {param.description}"
                    
                docs.append(param_info)
                
            if spec.examples:
                docs.append("\n**Examples:**")
                for i, example in enumerate(spec.examples, 1):
                    docs.append(f"```python")
                    docs.append(f"# Example {i}")
                    docs.append(f"result = await call_function('{func_name}', {json.dumps(example, indent=2)})")
                    docs.append("```")
                    
            docs.append("\n---\n")
            
        return "\n".join(docs)


# Example usage and testing
async def example_function(query: str, max_results: int = 10, 
                         include_metadata: bool = False) -> Dict[str, Any]:
    """Example function for testing the framework."""
    await asyncio.sleep(0.1)  # Simulate async operation
    
    return {
        "query": query,
        "results": [f"Result {i}" for i in range(max_results)],
        "metadata": {"timestamp": datetime.now().isoformat()} if include_metadata else None
    }


async def main():
    """Example usage of the Function Calling Framework."""
    # Initialize framework
    framework = FunctionCallFramework()
    
    # Register example function
    spec = FunctionSpec(
        name="example_function",
        description="Example function for demonstration",
        parameters=[
            ParameterSpec("query", ParameterType.REQUIRED, str, description="Search query"),
            ParameterSpec("max_results", ParameterType.OPTIONAL, int, 10, "Maximum results to return"),
            ParameterSpec("include_metadata", ParameterType.OPTIONAL, bool, False, "Include metadata in response")
        ],
        return_type=Dict[str, Any],
        examples=[
            {"query": "test search", "max_results": 5},
            {"query": "another search", "include_metadata": True}
        ],
        tags=["search", "example"]
    )
    
    framework.register_function(example_function, spec)
    
    # Test function call
    result = await framework.call_function(
        "example_function",
        {"query": "test", "max_results": 3},
        context="testing framework"
    )
    
    print(f"Function call result: {result.status}")
    print(f"Result data: {result.result}")
    print(f"Execution time: {result.execution_time:.3f}s")
    
    # Get recommendations
    recommendations = framework.get_function_recommendations(
        context="search task",
        user_query="I need to search for information"
    )
    
    print(f"\nRecommended functions: {recommendations}")
    
    # Generate documentation
    docs = framework.generate_function_documentation()
    print(f"\nFunction Documentation:\n{docs[:500]}...")


if __name__ == "__main__":
    asyncio.run(main())