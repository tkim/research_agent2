"""
Learning and Adaptation Framework for Research Agent 2

This module implements dynamic learning capabilities, pattern recognition,
API learning, and adaptive behavior for the research agent.
"""

import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from enum import Enum
import hashlib
import asyncio
from collections import defaultdict, Counter
import statistics


class LearningType(Enum):
    """Types of learning patterns."""
    API_USAGE = "api_usage"
    SEARCH_STRATEGY = "search_strategy"
    USER_PREFERENCE = "user_preference"
    ERROR_PATTERN = "error_pattern"
    SUCCESS_PATTERN = "success_pattern"
    DOMAIN_EXPERTISE = "domain_expertise"


class AdaptationLevel(Enum):
    """Levels of adaptation confidence."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class LearningEvent:
    """Records a learning event."""
    event_id: str
    learning_type: LearningType
    event_data: Dict[str, Any]
    outcome: str  # success, failure, partial
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APILearningRecord:
    """Records learning about API usage."""
    api_name: str
    endpoint: str
    successful_patterns: List[Dict[str, Any]]
    failed_patterns: List[Dict[str, Any]]
    parameter_insights: Dict[str, Any]
    error_patterns: List[Dict[str, Any]]
    optimal_parameters: Dict[str, Any]
    usage_frequency: int
    success_rate: float
    average_response_time: float
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SearchStrategyPattern:
    """Records successful search strategies."""
    query_type: str
    domain: str
    successful_keywords: List[str]
    effective_sources: List[str]
    optimal_filters: Dict[str, Any]
    success_metrics: Dict[str, float]
    usage_count: int
    contexts: List[str]
    last_used: datetime = field(default_factory=datetime.now)


@dataclass
class UserPreferenceProfile:
    """User preference and behavior profile."""
    user_id: str
    preferred_citation_style: str
    preferred_source_types: List[str]
    complexity_preference: str  # simple, detailed, comprehensive
    domain_interests: List[str]
    interaction_patterns: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]
    adaptation_level: AdaptationLevel
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationRule:
    """Rules for adapting behavior based on learning."""
    rule_id: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int
    confidence: float
    success_count: int
    failure_count: int
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None


class LearningDatabase:
    """Persistent storage for learning data."""
    
    def __init__(self, db_path: str = "learning_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.LearningDatabase")
        self._init_database()
        
    def _init_database(self):
        """Initialize the learning database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Learning events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_events (
                event_id TEXT PRIMARY KEY,
                learning_type TEXT,
                event_data TEXT,
                outcome TEXT,
                context TEXT,
                timestamp TEXT,
                confidence REAL,
                metadata TEXT
            )
        """)
        
        # API learning records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_learning (
                api_name TEXT,
                endpoint TEXT,
                record_data TEXT,
                last_updated TEXT,
                PRIMARY KEY (api_name, endpoint)
            )
        """)
        
        # Search strategy patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_patterns (
                pattern_id TEXT PRIMARY KEY,
                query_type TEXT,
                domain TEXT,
                pattern_data TEXT,
                success_metrics TEXT,
                usage_count INTEGER,
                last_used TEXT
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT,
                last_updated TEXT
            )
        """)
        
        # Adaptation rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adaptation_rules (
                rule_id TEXT PRIMARY KEY,
                condition TEXT,
                action TEXT,
                priority INTEGER,
                confidence REAL,
                success_count INTEGER,
                failure_count INTEGER,
                created_at TEXT,
                last_applied TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
    def store_learning_event(self, event: LearningEvent):
        """Store a learning event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO learning_events 
            (event_id, learning_type, event_data, outcome, context, timestamp, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.learning_type.value,
            json.dumps(event.event_data),
            event.outcome,
            json.dumps(event.context),
            event.timestamp.isoformat(),
            event.confidence,
            json.dumps(event.metadata)
        ))
        
        conn.commit()
        conn.close()
        
    def get_learning_events(self, learning_type: LearningType = None,
                           days_back: int = 30) -> List[LearningEvent]:
        """Retrieve learning events."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        if learning_type:
            cursor.execute("""
                SELECT * FROM learning_events 
                WHERE learning_type = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (learning_type.value, cutoff_date))
        else:
            cursor.execute("""
                SELECT * FROM learning_events 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff_date,))
            
        events = []
        for row in cursor.fetchall():
            event = LearningEvent(
                event_id=row[0],
                learning_type=LearningType(row[1]),
                event_data=json.loads(row[2]),
                outcome=row[3],
                context=json.loads(row[4]),
                timestamp=datetime.fromisoformat(row[5]),
                confidence=row[6],
                metadata=json.loads(row[7])
            )
            events.append(event)
            
        conn.close()
        return events
        
    def store_api_learning(self, record: APILearningRecord):
        """Store API learning record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO api_learning 
            (api_name, endpoint, record_data, last_updated)
            VALUES (?, ?, ?, ?)
        """, (
            record.api_name,
            record.endpoint,
            json.dumps(asdict(record)),
            record.last_updated.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def get_api_learning(self, api_name: str, endpoint: str = None) -> List[APILearningRecord]:
        """Retrieve API learning records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if endpoint:
            cursor.execute("""
                SELECT record_data FROM api_learning 
                WHERE api_name = ? AND endpoint = ?
            """, (api_name, endpoint))
        else:
            cursor.execute("""
                SELECT record_data FROM api_learning 
                WHERE api_name = ?
            """, (api_name,))
            
        records = []
        for row in cursor.fetchall():
            data = json.loads(row[0])
            record = APILearningRecord(**data)
            records.append(record)
            
        conn.close()
        return records


class PatternRecognition:
    """Recognizes patterns in user behavior and system performance."""
    
    def __init__(self, db: LearningDatabase):
        self.db = db
        self.logger = logging.getLogger(f"{__name__}.PatternRecognition")
        
    def analyze_search_patterns(self, days_back: int = 30) -> List[SearchStrategyPattern]:
        """Analyze successful search patterns."""
        events = self.db.get_learning_events(LearningType.SEARCH_STRATEGY, days_back)
        
        # Group events by query type and domain
        patterns = defaultdict(list)
        
        for event in events:
            if event.outcome == "success":
                key = (
                    event.event_data.get("query_type", "unknown"),
                    event.event_data.get("domain", "general")
                )
                patterns[key].append(event)
                
        # Analyze patterns
        strategy_patterns = []
        
        for (query_type, domain), pattern_events in patterns.items():
            if len(pattern_events) < 3:  # Need minimum occurrences
                continue
                
            # Extract common keywords
            all_keywords = []
            all_sources = []
            all_filters = []
            
            for event in pattern_events:
                all_keywords.extend(event.event_data.get("keywords", []))
                all_sources.extend(event.event_data.get("sources", []))
                all_filters.append(event.event_data.get("filters", {}))
                
            # Find most common elements
            keyword_counts = Counter(all_keywords)
            source_counts = Counter(all_sources)
            
            # Calculate success metrics
            success_metrics = {
                "average_confidence": statistics.mean(
                    event.confidence for event in pattern_events
                ),
                "response_time": statistics.mean(
                    event.metadata.get("response_time", 1.0) for event in pattern_events
                ),
                "source_count": statistics.mean(
                    event.metadata.get("source_count", 5) for event in pattern_events
                )
            }
            
            pattern = SearchStrategyPattern(
                query_type=query_type,
                domain=domain,
                successful_keywords=[kw for kw, count in keyword_counts.most_common(10)],
                effective_sources=[src for src, count in source_counts.most_common(5)],
                optimal_filters=self._merge_filters(all_filters),
                success_metrics=success_metrics,
                usage_count=len(pattern_events),
                contexts=[event.context.get("task", "") for event in pattern_events]
            )
            
            strategy_patterns.append(pattern)
            
        return strategy_patterns
        
    def _merge_filters(self, filters_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple filter dictionaries to find common preferences."""
        merged = {}
        
        if not filters_list:
            return merged
            
        # Find common filter values
        all_keys = set()
        for filters in filters_list:
            all_keys.update(filters.keys())
            
        for key in all_keys:
            values = [filters.get(key) for filters in filters_list if key in filters]
            if values:
                # Use most common value
                value_counts = Counter(values)
                merged[key] = value_counts.most_common(1)[0][0]
                
        return merged
        
    def detect_error_patterns(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Detect patterns in errors and failures."""
        events = self.db.get_learning_events(LearningType.ERROR_PATTERN, days_back)
        
        # Group errors by type and context
        error_groups = defaultdict(list)
        
        for event in events:
            error_type = event.event_data.get("error_type", "unknown")
            context = event.context.get("operation", "unknown")
            key = f"{error_type}_{context}"
            error_groups[key].append(event)
            
        patterns = []
        
        for error_key, error_events in error_groups.items():
            if len(error_events) < 2:  # Need at least 2 occurrences
                continue
                
            # Analyze common factors
            common_parameters = self._find_common_parameters(error_events)
            
            pattern = {
                "error_pattern": error_key,
                "frequency": len(error_events),
                "common_parameters": common_parameters,
                "first_seen": min(event.timestamp for event in error_events),
                "last_seen": max(event.timestamp for event in error_events),
                "suggestions": self._generate_error_suggestions(error_events)
            }
            
            patterns.append(pattern)
            
        return patterns
        
    def _find_common_parameters(self, events: List[LearningEvent]) -> Dict[str, Any]:
        """Find common parameters across events."""
        all_params = defaultdict(list)
        
        for event in events:
            for key, value in event.event_data.items():
                if isinstance(value, (str, int, float, bool)):
                    all_params[key].append(value)
                    
        common_params = {}
        for key, values in all_params.items():
            if len(set(values)) == 1:  # All values are the same
                common_params[key] = values[0]
                
        return common_params
        
    def _generate_error_suggestions(self, error_events: List[LearningEvent]) -> List[str]:
        """Generate suggestions based on error patterns."""
        suggestions = []
        
        # Generic suggestions based on error types
        error_types = [event.event_data.get("error_type", "") for event in error_events]
        
        if "timeout" in str(error_types).lower():
            suggestions.append("Consider increasing timeout values")
            suggestions.append("Implement retry logic with exponential backoff")
            
        if "rate_limit" in str(error_types).lower():
            suggestions.append("Implement rate limiting with proper delays")
            suggestions.append("Consider using multiple API keys if available")
            
        if "authentication" in str(error_types).lower():
            suggestions.append("Verify API key validity and permissions")
            suggestions.append("Check token expiration and renewal")
            
        return suggestions


class AdaptiveBehavior:
    """Implements adaptive behavior based on learned patterns."""
    
    def __init__(self, db: LearningDatabase):
        self.db = db
        self.pattern_recognition = PatternRecognition(db)
        self.adaptation_rules: List[AdaptationRule] = []
        self.logger = logging.getLogger(f"{__name__}.AdaptiveBehavior")
        self._load_adaptation_rules()
        
    def _load_adaptation_rules(self):
        """Load adaptation rules from database."""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM adaptation_rules ORDER BY priority DESC")
        
        for row in cursor.fetchall():
            rule = AdaptationRule(
                rule_id=row[0],
                condition=json.loads(row[1]),
                action=json.loads(row[2]),
                priority=row[3],
                confidence=row[4],
                success_count=row[5],
                failure_count=row[6],
                created_at=datetime.fromisoformat(row[7]),
                last_applied=datetime.fromisoformat(row[8]) if row[8] else None
            )
            self.adaptation_rules.append(rule)
            
        conn.close()
        
    def adapt_search_strategy(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt search strategy based on learned patterns."""
        query_type = context.get("query_type", "factual")
        domain = context.get("domain", "general")
        
        # Find relevant patterns
        patterns = self.pattern_recognition.analyze_search_patterns()
        relevant_patterns = [
            p for p in patterns 
            if p.query_type == query_type or p.domain == domain
        ]
        
        if not relevant_patterns:
            return {"adapted": False, "reason": "No relevant patterns found"}
            
        # Select best pattern
        best_pattern = max(relevant_patterns, 
                          key=lambda p: p.success_metrics.get("average_confidence", 0))
        
        # Generate adapted strategy
        adapted_strategy = {
            "adapted": True,
            "source_pattern": {
                "query_type": best_pattern.query_type,
                "domain": best_pattern.domain,
                "usage_count": best_pattern.usage_count
            },
            "recommended_keywords": best_pattern.successful_keywords[:5],
            "preferred_sources": best_pattern.effective_sources[:3],
            "optimal_filters": best_pattern.optimal_filters,
            "expected_confidence": best_pattern.success_metrics.get("average_confidence", 0.5)
        }
        
        # Log adaptation
        self._log_adaptation("search_strategy", context, adapted_strategy)
        
        return adapted_strategy
        
    def adapt_api_usage(self, api_name: str, endpoint: str, 
                       base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt API usage based on learned patterns."""
        records = self.db.get_api_learning(api_name, endpoint)
        
        if not records:
            return {
                "adapted": False,
                "parameters": base_parameters,
                "reason": "No learning data available"
            }
            
        record = records[0]  # Most recent record
        
        # Adapt parameters based on learning
        adapted_parameters = base_parameters.copy()
        
        # Apply optimal parameters
        for key, value in record.optimal_parameters.items():
            if key not in adapted_parameters:
                adapted_parameters[key] = value
                
        # Adjust based on success rate
        if record.success_rate < 0.7:
            # Use more conservative parameters
            if "timeout" in adapted_parameters:
                adapted_parameters["timeout"] = max(
                    adapted_parameters["timeout"] * 1.5,
                    record.average_response_time * 2
                )
                
        adaptation_info = {
            "adapted": True,
            "parameters": adapted_parameters,
            "learning_source": {
                "success_rate": record.success_rate,
                "usage_frequency": record.usage_frequency,
                "average_response_time": record.average_response_time
            },
            "changes_made": [
                f"Added {key}={value}" 
                for key, value in record.optimal_parameters.items()
                if key not in base_parameters
            ]
        }
        
        # Log adaptation
        self._log_adaptation("api_usage", 
                           {"api_name": api_name, "endpoint": endpoint}, 
                           adaptation_info)
        
        return adaptation_info
        
    def generate_adaptation_rules(self) -> List[AdaptationRule]:
        """Generate new adaptation rules based on patterns."""
        new_rules = []
        
        # Analyze error patterns
        error_patterns = self.pattern_recognition.detect_error_patterns()
        
        for pattern in error_patterns:
            if pattern["frequency"] >= 3:  # Frequent enough to warrant a rule
                rule = AdaptationRule(
                    rule_id=f"error_rule_{hashlib.md5(str(pattern).encode()).hexdigest()[:8]}",
                    condition={
                        "error_pattern": pattern["error_pattern"],
                        "parameters": pattern["common_parameters"]
                    },
                    action={
                        "type": "parameter_adjustment",
                        "suggestions": pattern["suggestions"]
                    },
                    priority=min(pattern["frequency"], 10),
                    confidence=min(pattern["frequency"] / 10, 1.0),
                    success_count=0,
                    failure_count=0
                )
                new_rules.append(rule)
                
        # Store new rules
        for rule in new_rules:
            self._store_adaptation_rule(rule)
            
        return new_rules
        
    def _log_adaptation(self, adaptation_type: str, context: Dict[str, Any], 
                       result: Dict[str, Any]):
        """Log an adaptation event."""
        event = LearningEvent(
            event_id=f"adapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{adaptation_type}",
            learning_type=LearningType.SUCCESS_PATTERN,
            event_data={
                "adaptation_type": adaptation_type,
                "result": result
            },
            outcome="applied",
            context=context,
            confidence=0.7
        )
        
        self.db.store_learning_event(event)
        
    def _store_adaptation_rule(self, rule: AdaptationRule):
        """Store adaptation rule in database."""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO adaptation_rules 
            (rule_id, condition, action, priority, confidence, success_count, failure_count, created_at, last_applied)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id,
            json.dumps(rule.condition),
            json.dumps(rule.action),
            rule.priority,
            rule.confidence,
            rule.success_count,
            rule.failure_count,
            rule.created_at.isoformat(),
            rule.last_applied.isoformat() if rule.last_applied else None
        ))
        
        conn.commit()
        conn.close()


class LearningAdaptationFramework:
    """Main framework for learning and adaptation."""
    
    def __init__(self, db_path: str = "research_agent_learning.db"):
        self.db = LearningDatabase(db_path)
        self.adaptive_behavior = AdaptiveBehavior(self.db)
        self.logger = logging.getLogger(f"{__name__}.LearningAdaptationFramework")
        
    def record_learning_event(self, learning_type: LearningType,
                             event_data: Dict[str, Any],
                             outcome: str,
                             context: Dict[str, Any],
                             confidence: float = 0.5):
        """Record a learning event."""
        event = LearningEvent(
            event_id=f"{learning_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            learning_type=learning_type,
            event_data=event_data,
            outcome=outcome,
            context=context,
            confidence=confidence
        )
        
        self.db.store_learning_event(event)
        self.logger.info(f"Recorded learning event: {learning_type.value} - {outcome}")
        
    def learn_from_api_usage(self, api_name: str, endpoint: str,
                           parameters: Dict[str, Any],
                           success: bool, response_time: float,
                           error_info: Optional[Dict[str, Any]] = None):
        """Learn from API usage patterns."""
        # Get or create API learning record
        existing_records = self.db.get_api_learning(api_name, endpoint)
        
        if existing_records:
            record = existing_records[0]
        else:
            record = APILearningRecord(
                api_name=api_name,
                endpoint=endpoint,
                successful_patterns=[],
                failed_patterns=[],
                parameter_insights={},
                error_patterns=[],
                optimal_parameters={},
                usage_frequency=0,
                success_rate=0.0,
                average_response_time=0.0
            )
            
        # Update record
        record.usage_frequency += 1
        
        if success:
            record.successful_patterns.append({
                "parameters": parameters,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update optimal parameters
            for key, value in parameters.items():
                if key not in record.optimal_parameters:
                    record.optimal_parameters[key] = value
                    
        else:
            failed_pattern = {
                "parameters": parameters,
                "timestamp": datetime.now().isoformat()
            }
            if error_info:
                failed_pattern["error"] = error_info
                
            record.failed_patterns.append(failed_pattern)
            
            if error_info:
                record.error_patterns.append(error_info)
                
        # Update success rate
        total_successes = len(record.successful_patterns)
        total_attempts = len(record.successful_patterns) + len(record.failed_patterns)
        record.success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0
        
        # Update average response time
        if record.successful_patterns:
            record.average_response_time = statistics.mean(
                p["response_time"] for p in record.successful_patterns
            )
            
        # Store updated record
        self.db.store_api_learning(record)
        
        # Record learning event
        self.record_learning_event(
            LearningType.API_USAGE,
            {
                "api_name": api_name,
                "endpoint": endpoint,
                "parameters": parameters,
                "response_time": response_time
            },
            "success" if success else "failure",
            {"operation": "api_call"},
            confidence=0.8 if success else 0.3
        )
        
    def adapt_search_approach(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive recommendations for search approach."""
        return self.adaptive_behavior.adapt_search_strategy(query, context)
        
    def adapt_api_parameters(self, api_name: str, endpoint: str,
                           base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive recommendations for API parameters."""
        return self.adaptive_behavior.adapt_api_usage(api_name, endpoint, base_parameters)
        
    def generate_learning_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate a comprehensive learning report."""
        # Get learning events
        all_events = self.db.get_learning_events(days_back=days_back)
        
        # Analyze by type
        events_by_type = defaultdict(list)
        for event in all_events:
            events_by_type[event.learning_type].append(event)
            
        # Calculate success rates
        success_rates = {}
        for learning_type, events in events_by_type.items():
            successful = sum(1 for e in events if e.outcome == "success")
            success_rates[learning_type.value] = successful / len(events) if events else 0
            
        # Get adaptation rules
        adaptation_rules = self.adaptive_behavior.adaptation_rules
        
        # Get pattern analysis
        search_patterns = self.adaptive_behavior.pattern_recognition.analyze_search_patterns(days_back)
        error_patterns = self.adaptive_behavior.pattern_recognition.detect_error_patterns(days_back)
        
        report = {
            "reporting_period": f"Last {days_back} days",
            "total_learning_events": len(all_events),
            "events_by_type": {lt.value: len(events) for lt, events in events_by_type.items()},
            "success_rates": success_rates,
            "search_patterns_discovered": len(search_patterns),
            "error_patterns_detected": len(error_patterns),
            "adaptation_rules_active": len(adaptation_rules),
            "top_search_patterns": [
                {
                    "query_type": p.query_type,
                    "domain": p.domain,
                    "usage_count": p.usage_count,
                    "avg_confidence": p.success_metrics.get("average_confidence", 0)
                }
                for p in sorted(search_patterns, 
                              key=lambda x: x.success_metrics.get("average_confidence", 0), 
                              reverse=True)[:5]
            ],
            "most_common_errors": [
                {
                    "pattern": ep["error_pattern"],
                    "frequency": ep["frequency"],
                    "suggestions": ep["suggestions"][:2]  # Top 2 suggestions
                }
                for ep in sorted(error_patterns, key=lambda x: x["frequency"], reverse=True)[:3]
            ]
        }
        
        return report
        
    def reset_learning_data(self, confirm: bool = False):
        """Reset all learning data (use with caution)."""
        if not confirm:
            self.logger.warning("Reset not confirmed. Use confirm=True to actually reset data.")
            return
            
        # Clear database
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        tables = ["learning_events", "api_learning", "search_patterns", 
                 "user_preferences", "adaptation_rules"]
        
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            
        conn.commit()
        conn.close()
        
        self.logger.info("All learning data has been reset")


# Example usage
async def main():
    """Example usage of the Learning and Adaptation Framework."""
    # Initialize framework
    framework = LearningAdaptationFramework()
    
    # Simulate some learning events
    print("Recording learning events...")
    
    # API usage learning
    framework.learn_from_api_usage(
        api_name="search_api",
        endpoint="/search",
        parameters={"query": "AI research", "limit": 10},
        success=True,
        response_time=1.2
    )
    
    # Search strategy learning
    framework.record_learning_event(
        LearningType.SEARCH_STRATEGY,
        {
            "query_type": "academic",
            "domain": "technology",
            "keywords": ["artificial intelligence", "machine learning"],
            "sources": ["arxiv.org", "scholar.google.com"],
            "filters": {"date_range": "2024", "peer_reviewed": True}
        },
        "success",
        {"task": "literature_review", "user_preference": "comprehensive"},
        confidence=0.9
    )
    
    # Get adaptive recommendations
    print("\nGetting adaptive recommendations...")
    
    search_adaptation = framework.adapt_search_approach(
        "machine learning applications", 
        {"query_type": "academic", "domain": "technology"}
    )
    print(f"Search adaptation: {search_adaptation}")
    
    api_adaptation = framework.adapt_api_parameters(
        "search_api", 
        "/search", 
        {"query": "test", "limit": 5}
    )
    print(f"API adaptation: {api_adaptation}")
    
    # Generate learning report
    print("\nGenerating learning report...")
    report = framework.generate_learning_report()
    
    print("Learning Report:")
    print(f"- Total events: {report['total_learning_events']}")
    print(f"- Search patterns: {report['search_patterns_discovered']}")
    print(f"- Error patterns: {report['error_patterns_detected']}")
    print(f"- Active rules: {report['adaptation_rules_active']}")


if __name__ == "__main__":
    asyncio.run(main())