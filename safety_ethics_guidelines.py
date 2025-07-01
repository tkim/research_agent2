"""
Safety and Ethics Guidelines for Research Agent 2

This module implements comprehensive safety and ethics guidelines covering
information accuracy, privacy protection, responsible disclosure, and ethical
research practices.
"""

import logging
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import json


class SafetyLevel(Enum):
    """Safety assessment levels."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


class PrivacyLevel(Enum):
    """Privacy sensitivity levels."""
    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class InformationReliability(Enum):
    """Information reliability levels."""
    VERIFIED = "verified"
    PRELIMINARY = "preliminary"
    CONFLICTING = "conflicting"
    UNCERTAIN = "uncertain"
    SPECULATIVE = "speculative"


@dataclass
class SafetyAssessment:
    """Assessment of safety for a query or operation."""
    level: SafetyLevel
    reasons: List[str]
    recommendations: List[str]
    blocked_terms: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PrivacyAssessment:
    """Assessment of privacy implications."""
    level: PrivacyLevel
    identified_data_types: List[str]
    risks: List[str]
    protection_measures: List[str]
    compliance_notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InformationValidation:
    """Validation of information accuracy and reliability."""
    reliability: InformationReliability
    verification_methods: List[str]
    confidence_level: float
    sources_checked: int
    conflicting_sources: int
    verification_notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EthicsViolation:
    """Records an ethics violation or concern."""
    violation_type: str
    severity: str
    description: str
    context: Dict[str, Any]
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)


class SafetyFilter:
    """Filters content and queries for safety concerns."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SafetyFilter")
        self._load_safety_patterns()
        
    def _load_safety_patterns(self):
        """Load patterns for safety filtering."""
        # Dangerous query patterns
        self.dangerous_patterns = {
            "malware": [
                r'\b(create|build|make).*virus\b',
                r'\bmalware.*development\b',
                r'\bhacking.*tools\b'
            ],
            "illegal_activities": [
                r'\b(illegal|illicit).*activities\b',
                r'\bdrug.*manufacturing\b',
                r'\bweapon.*creation\b'
            ],
            "harmful_content": [
                r'\b(self.harm|suicide).*methods\b',
                r'\bviolence.*instructions\b',
                r'\bharm.*others\b'
            ],
            "privacy_violations": [
                r'\bpersonal.*information.*extraction\b',
                r'\bprivate.*data.*access\b',
                r'\bunauthorized.*access\b'
            ]
        }
        
        # Warning-level patterns
        self.warning_patterns = {
            "sensitive_topics": [
                r'\bmedical.*diagnosis\b',
                r'\blegal.*advice\b',
                r'\bfinancial.*investment\b'
            ],
            "unverified_claims": [
                r'\bconspiracy.*theory\b',
                r'\bunproven.*treatment\b',
                r'\bmisinformation\b'
            ]
        }
        
        # Blocked terms (absolute no-go)
        self.blocked_terms = [
            "child exploitation",
            "terrorist instructions",
            "identity theft methods",
            "credit card fraud"
        ]
        
        # Safe research domains
        self.safe_domains = [
            "academic research",
            "educational content",
            "public information",
            "scientific studies",
            "historical facts"
        ]
        
    def assess_query_safety(self, query: str, context: Dict[str, Any] = None) -> SafetyAssessment:
        """
        Assess the safety of a research query.
        
        Args:
            query: The research query to assess
            context: Additional context about the query
            
        Returns:
            SafetyAssessment with safety level and recommendations
        """
        context = context or {}
        query_lower = query.lower()
        
        reasons = []
        recommendations = []
        blocked_terms = []
        alternative_approaches = []
        
        # Check for blocked terms first
        for term in self.blocked_terms:
            if term in query_lower:
                blocked_terms.append(term)
                
        if blocked_terms:
            return SafetyAssessment(
                level=SafetyLevel.BLOCKED,
                reasons=[f"Query contains blocked term: {term}" for term in blocked_terms],
                recommendations=["Reformulate query without sensitive terms"],
                blocked_terms=blocked_terms,
                alternative_approaches=["Focus on educational or theoretical aspects"]
            )
            
        # Check dangerous patterns
        danger_level = 0
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    danger_level += 3
                    reasons.append(f"Dangerous pattern detected: {category}")
                    
        # Check warning patterns
        warning_level = 0
        for category, patterns in self.warning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    warning_level += 1
                    reasons.append(f"Warning pattern detected: {category}")
                    
        # Determine safety level
        if danger_level >= 3:
            safety_level = SafetyLevel.DANGEROUS
            recommendations.extend([
                "Avoid this type of research",
                "Consider educational alternatives",
                "Consult appropriate authorities if legitimate need"
            ])
        elif danger_level > 0 or warning_level >= 2:
            safety_level = SafetyLevel.WARNING
            recommendations.extend([
                "Proceed with extreme caution",
                "Verify information from authoritative sources only",
                "Consider ethical implications"
            ])
        elif warning_level > 0:
            safety_level = SafetyLevel.CAUTION
            recommendations.extend([
                "Use authoritative sources",
                "Verify information carefully",
                "Note limitations of findings"
            ])
        else:
            safety_level = SafetyLevel.SAFE
            recommendations.append("Query appears safe for research")
            
        # Generate alternative approaches for problematic queries
        if safety_level in [SafetyLevel.WARNING, SafetyLevel.DANGEROUS]:
            alternative_approaches = self._generate_safe_alternatives(query, context)
            
        return SafetyAssessment(
            level=safety_level,
            reasons=reasons,
            recommendations=recommendations,
            alternative_approaches=alternative_approaches
        )
        
    def _generate_safe_alternatives(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate safe alternative research approaches."""
        alternatives = []
        query_lower = query.lower()
        
        # General safe alternatives
        if "medical" in query_lower:
            alternatives.extend([
                "Research from established medical institutions",
                "Focus on peer-reviewed medical literature",
                "Consult official health organization guidelines"
            ])
            
        if "legal" in query_lower:
            alternatives.extend([
                "Research general legal principles",
                "Consult official legal resources",
                "Focus on educational legal content"
            ])
            
        if "security" in query_lower or "hacking" in query_lower:
            alternatives.extend([
                "Study cybersecurity from defensive perspective",
                "Research official security guidelines",
                "Focus on security awareness and protection"
            ])
            
        return alternatives
        
    def filter_content(self, content: str, source_url: str = "") -> Tuple[str, List[str]]:
        """
        Filter content for safety concerns.
        
        Args:
            content: Content to filter
            source_url: Source URL for context
            
        Returns:
            Tuple of (filtered_content, removed_sections)
        """
        removed_sections = []
        filtered_content = content
        
        # Check for dangerous content patterns
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, filtered_content, re.IGNORECASE)
                for match in matches:
                    start, end = match.span()
                    # Remove sentence containing the dangerous pattern
                    sentence_start = filtered_content.rfind('.', 0, start) + 1
                    sentence_end = filtered_content.find('.', end)
                    if sentence_end == -1:
                        sentence_end = len(filtered_content)
                        
                    removed_section = filtered_content[sentence_start:sentence_end]
                    removed_sections.append(f"Removed {category}: {removed_section[:50]}...")
                    
                    # Replace with placeholder
                    filtered_content = (
                        filtered_content[:sentence_start] + 
                        "[CONTENT REMOVED FOR SAFETY] " +
                        filtered_content[sentence_end:]
                    )
                    
        return filtered_content, removed_sections


class PrivacyProtector:
    """Protects user privacy and identifies sensitive information."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PrivacyProtector")
        self._load_privacy_patterns()
        
    def _load_privacy_patterns(self):
        """Load patterns for privacy detection."""
        self.personal_data_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "personal_name": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        }
        
        self.sensitive_keywords = [
            "personal information", "private data", "confidential",
            "medical records", "financial information", "login credentials",
            "password", "social security", "bank account", "driver license"
        ]
        
        self.privacy_risk_indicators = [
            "collect personal data", "store user information", "track users",
            "share data with", "sell information", "data mining", "surveillance"
        ]
        
    def assess_privacy_risk(self, query: str, context: Dict[str, Any] = None) -> PrivacyAssessment:
        """
        Assess privacy risks in a query or operation.
        
        Args:
            query: Query to assess
            context: Additional context
            
        Returns:
            PrivacyAssessment with risk level and protection measures
        """
        context = context or {}
        query_lower = query.lower()
        
        identified_data_types = []
        risks = []
        protection_measures = []
        compliance_notes = []
        
        # Detect personal data types in query
        for data_type, pattern in self.personal_data_patterns.items():
            if re.search(pattern, query):
                identified_data_types.append(data_type)
                risks.append(f"Query contains {data_type}")
                
        # Check for sensitive keywords
        for keyword in self.sensitive_keywords:
            if keyword in query_lower:
                identified_data_types.append("sensitive_keyword")
                risks.append(f"Query mentions sensitive topic: {keyword}")
                
        # Check for privacy risk indicators
        for indicator in self.privacy_risk_indicators:
            if indicator in query_lower:
                risks.append(f"Privacy risk indicator: {indicator}")
                
        # Determine privacy level
        if any("ssn" in dt or "credit_card" in dt for dt in identified_data_types):
            privacy_level = PrivacyLevel.RESTRICTED
            protection_measures.extend([
                "Do not process or store this information",
                "Immediately purge any collected data",
                "Alert security team"
            ])
        elif any("email" in dt or "phone" in dt for dt in identified_data_types):
            privacy_level = PrivacyLevel.CONFIDENTIAL
            protection_measures.extend([
                "Encrypt any stored data",
                "Limit access to authorized personnel",
                "Implement data retention policies"
            ])
        elif identified_data_types or risks:
            privacy_level = PrivacyLevel.SENSITIVE
            protection_measures.extend([
                "Use privacy-preserving techniques",
                "Anonymize data where possible",
                "Follow data minimization principles"
            ])
        elif any(word in query_lower for word in ["personal", "private", "confidential"]):
            privacy_level = PrivacyLevel.PERSONAL
            protection_measures.extend([
                "Be cautious with data handling",
                "Respect user privacy preferences"
            ])
        else:
            privacy_level = PrivacyLevel.PUBLIC
            protection_measures.append("Standard privacy practices apply")
            
        # Add compliance notes
        if privacy_level in [PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED]:
            compliance_notes.extend([
                "May be subject to GDPR regulations",
                "Consider CCPA compliance requirements",
                "Review organizational data policies"
            ])
            
        return PrivacyAssessment(
            level=privacy_level,
            identified_data_types=identified_data_types,
            risks=risks,
            protection_measures=protection_measures,
            compliance_notes=compliance_notes
        )
        
    def anonymize_data(self, text: str, preserve_structure: bool = True) -> Tuple[str, Dict[str, str]]:
        """
        Anonymize personal data in text.
        
        Args:
            text: Text to anonymize
            preserve_structure: Whether to preserve text structure
            
        Returns:
            Tuple of (anonymized_text, anonymization_map)
        """
        anonymized = text
        anonymization_map = {}
        
        for data_type, pattern in self.personal_data_patterns.items():
            matches = list(re.finditer(pattern, anonymized))
            for i, match in enumerate(matches):
                original = match.group()
                if preserve_structure:
                    if data_type == "email":
                        replacement = f"[EMAIL_{i+1}]"
                    elif data_type == "phone":
                        replacement = f"[PHONE_{i+1}]"
                    elif data_type == "ssn":
                        replacement = "[SSN_REDACTED]"
                    elif data_type == "credit_card":
                        replacement = "[CARD_REDACTED]"
                    else:
                        replacement = f"[{data_type.upper()}_{i+1}]"
                else:
                    replacement = "[REDACTED]"
                    
                anonymized = anonymized.replace(original, replacement, 1)
                anonymization_map[replacement] = original
                
        return anonymized, anonymization_map


class InformationValidator:
    """Validates information accuracy and distinguishes fact from speculation."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InformationValidator")
        self._load_validation_patterns()
        
    def _load_validation_patterns(self):
        """Load patterns for information validation."""
        self.certainty_indicators = {
            "high_certainty": [
                "research shows", "studies demonstrate", "data indicates",
                "proven", "established", "confirmed", "verified"
            ],
            "medium_certainty": [
                "evidence suggests", "studies indicate", "research suggests",
                "likely", "probable", "appears to", "seems to"
            ],
            "low_certainty": [
                "may", "might", "could", "possibly", "potentially",
                "preliminary", "initial findings", "early research"
            ],
            "speculation": [
                "believes", "thinks", "opinion", "speculation", "rumor",
                "unconfirmed", "alleged", "supposedly", "reportedly"
            ]
        }
        
        self.verification_sources = {
            "academic": ["edu", "scholar.google.com", "pubmed", "arxiv.org"],
            "official": ["gov", "who.int", "cdc.gov", "fda.gov"],
            "news": ["reuters.com", "bbc.com", "npr.org", "ap.org"],
            "reference": ["wikipedia.org", "britannica.com"]
        }
        
        self.bias_indicators = [
            "amazing", "incredible", "shocking", "unbelievable",
            "devastating", "revolutionary", "miraculous", "secret"
        ]
        
    def validate_information(self, content: str, sources: List[Dict[str, Any]],
                           cross_reference_count: int = 0) -> InformationValidation:
        """
        Validate information reliability and accuracy.
        
        Args:
            content: Content to validate
            sources: Sources providing the information
            cross_reference_count: Number of cross-references checked
            
        Returns:
            InformationValidation with reliability assessment
        """
        verification_methods = []
        verification_notes = []
        content_lower = content.lower()
        
        # Analyze certainty language
        certainty_scores = {}
        for level, indicators in self.certainty_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > 0:
                certainty_scores[level] = score
                verification_methods.append(f"Language analysis: {level}")
                
        # Analyze source quality
        source_quality_score = 0
        sources_by_type = {"academic": 0, "official": 0, "news": 0, "reference": 0}
        
        for source in sources:
            url = source.get("url", "").lower()
            for source_type, domains in self.verification_sources.items():
                if any(domain in url for domain in domains):
                    sources_by_type[source_type] += 1
                    source_quality_score += {"academic": 3, "official": 3, "news": 2, "reference": 1}[source_type]
                    break
                    
        if source_quality_score > 0:
            verification_methods.append("Source credibility analysis")
            
        # Check for bias indicators
        bias_count = sum(1 for indicator in self.bias_indicators if indicator in content_lower)
        if bias_count > 0:
            verification_notes.append(f"Detected {bias_count} potential bias indicators")
            
        # Determine reliability level
        if certainty_scores.get("speculation", 0) > 0 or bias_count > 2:
            reliability = InformationReliability.SPECULATIVE
            confidence = 0.2
        elif certainty_scores.get("low_certainty", 0) > certainty_scores.get("high_certainty", 0):
            reliability = InformationReliability.UNCERTAIN
            confidence = 0.4
        elif cross_reference_count > 0 and sources_by_type["academic"] + sources_by_type["official"] == 0:
            reliability = InformationReliability.CONFLICTING
            confidence = 0.3
        elif sources_by_type["academic"] > 0 or sources_by_type["official"] > 0:
            if certainty_scores.get("high_certainty", 0) > 0:
                reliability = InformationReliability.VERIFIED
                confidence = 0.9
            else:
                reliability = InformationReliability.PRELIMINARY
                confidence = 0.7
        else:
            reliability = InformationReliability.PRELIMINARY
            confidence = 0.6
            
        # Cross-reference adjustment
        if cross_reference_count >= 3:
            confidence = min(1.0, confidence + 0.1)
            verification_methods.append(f"Cross-referenced with {cross_reference_count} sources")
        elif cross_reference_count > 0:
            verification_methods.append(f"Partial cross-referencing ({cross_reference_count} sources)")
            
        return InformationValidation(
            reliability=reliability,
            verification_methods=verification_methods,
            confidence_level=confidence,
            sources_checked=len(sources),
            conflicting_sources=0,  # Would need conflict detection logic
            verification_notes=verification_notes
        )
        
    def check_fact_accuracy(self, claim: str, evidence: List[str]) -> Dict[str, Any]:
        """
        Check the accuracy of a specific factual claim.
        
        Args:
            claim: Factual claim to check
            evidence: Supporting evidence
            
        Returns:
            Accuracy assessment
        """
        assessment = {
            "claim": claim,
            "evidence_count": len(evidence),
            "accuracy_confidence": 0.5,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "verification_status": "unverified"
        }
        
        # Simple evidence analysis
        claim_lower = claim.lower()
        supporting = 0
        contradicting = 0
        
        for piece in evidence:
            piece_lower = piece.lower()
            # Count supporting vs contradicting evidence
            if any(word in piece_lower for word in claim_lower.split()[:3]):
                supporting += 1
                assessment["supporting_evidence"].append(piece[:100] + "...")
            elif any(word in piece_lower for word in ["not", "false", "incorrect", "wrong"]):
                contradicting += 1
                assessment["contradicting_evidence"].append(piece[:100] + "...")
                
        # Determine verification status
        if supporting >= 2 and contradicting == 0:
            assessment["verification_status"] = "verified"
            assessment["accuracy_confidence"] = 0.8
        elif supporting > contradicting:
            assessment["verification_status"] = "likely_accurate"
            assessment["accuracy_confidence"] = 0.7
        elif contradicting > supporting:
            assessment["verification_status"] = "likely_inaccurate"
            assessment["accuracy_confidence"] = 0.3
        elif contradicting > 0:
            assessment["verification_status"] = "conflicting"
            assessment["accuracy_confidence"] = 0.4
            
        return assessment


class EthicsMonitor:
    """Monitors for ethical violations and maintains ethical standards."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EthicsMonitor")
        self.violations_log: List[EthicsViolation] = []
        self._load_ethics_guidelines()
        
    def _load_ethics_guidelines(self):
        """Load ethical guidelines and violation patterns."""
        self.ethics_principles = [
            "Respect for persons and autonomy",
            "Beneficence and non-maleficence",
            "Justice and fairness",
            "Transparency and accountability",
            "Privacy and confidentiality",
            "Informed consent",
            "Responsible disclosure"
        ]
        
        self.violation_patterns = {
            "misrepresentation": [
                "present speculation as fact",
                "hide uncertainty",
                "overstate confidence"
            ],
            "privacy_violation": [
                "expose personal information",
                "violate confidentiality",
                "unauthorized data collection"
            ],
            "bias_amplification": [
                "present biased information as neutral",
                "ignore conflicting evidence",
                "selective reporting"
            ],
            "harm_potential": [
                "dangerous misinformation",
                "harmful recommendations",
                "unsafe practices"
            ]
        }
        
    def monitor_research_ethics(self, operation: str, context: Dict[str, Any],
                              results: Any) -> List[EthicsViolation]:
        """
        Monitor research operation for ethical violations.
        
        Args:
            operation: Type of operation performed
            context: Context of the operation
            results: Results or outputs
            
        Returns:
            List of detected ethics violations
        """
        violations = []
        
        # Check for common ethical issues
        if operation == "information_synthesis":
            violations.extend(self._check_synthesis_ethics(context, results))
        elif operation == "source_evaluation":
            violations.extend(self._check_evaluation_ethics(context, results))
        elif operation == "data_collection":
            violations.extend(self._check_collection_ethics(context, results))
            
        # Log violations
        for violation in violations:
            self.violations_log.append(violation)
            self.logger.warning(f"Ethics violation detected: {violation.violation_type}")
            
        return violations
        
    def _check_synthesis_ethics(self, context: Dict[str, Any], results: Any) -> List[EthicsViolation]:
        """Check ethics of information synthesis."""
        violations = []
        
        # Check for misrepresentation
        confidence = context.get("confidence_level", 0.5)
        presentation = str(results).lower() if results else ""
        
        if confidence < 0.6 and not any(word in presentation for word in ["uncertain", "preliminary", "limited"]):
            violations.append(EthicsViolation(
                violation_type="misrepresentation",
                severity="medium",
                description="Low-confidence information presented without appropriate uncertainty indicators",
                context=context,
                recommended_action="Add uncertainty qualifiers to presentation"
            ))
            
        # Check for bias amplification
        sources = context.get("sources", [])
        if sources:
            low_credibility_count = sum(1 for s in sources if s.get("credibility_score", 0.5) < 0.4)
            if low_credibility_count / len(sources) > 0.5:
                violations.append(EthicsViolation(
                    violation_type="bias_amplification",
                    severity="medium",
                    description="High proportion of low-credibility sources without adequate disclosure",
                    context=context,
                    recommended_action="Clearly indicate source quality limitations"
                ))
                
        return violations
        
    def _check_evaluation_ethics(self, context: Dict[str, Any], results: Any) -> List[EthicsViolation]:
        """Check ethics of source evaluation."""
        violations = []
        
        # Check for fair evaluation
        if isinstance(results, dict) and "bias_indicators" in results:
            bias_indicators = results.get("bias_indicators", [])
            if len(bias_indicators) > 3:
                violations.append(EthicsViolation(
                    violation_type="bias_amplification",
                    severity="low",
                    description="Source with multiple bias indicators included without sufficient warning",
                    context=context,
                    recommended_action="Provide clear bias warnings or exclude source"
                ))
                
        return violations
        
    def _check_collection_ethics(self, context: Dict[str, Any], results: Any) -> List[EthicsViolation]:
        """Check ethics of data collection."""
        violations = []
        
        # Check for privacy violations
        collection_method = context.get("method", "")
        if "scraping" in collection_method.lower():
            violations.append(EthicsViolation(
                violation_type="privacy_violation",
                severity="low",
                description="Web scraping performed - ensure robots.txt compliance",
                context=context,
                recommended_action="Verify robots.txt compliance and rate limiting"
            ))
            
        return violations
        
    def generate_ethics_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate ethics compliance report."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_violations = [v for v in self.violations_log if v.timestamp > cutoff_date]
        
        # Categorize violations
        violations_by_type = {}
        violations_by_severity = {"low": 0, "medium": 0, "high": 0}
        
        for violation in recent_violations:
            vtype = violation.violation_type
            violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
            violations_by_severity[violation.severity] += 1
            
        report = {
            "reporting_period": f"Last {days_back} days",
            "total_violations": len(recent_violations),
            "violations_by_type": violations_by_type,
            "violations_by_severity": violations_by_severity,
            "ethics_principles_status": self._assess_principles_compliance(recent_violations),
            "recommendations": self._generate_ethics_recommendations(recent_violations),
            "compliance_score": self._calculate_compliance_score(recent_violations, days_back)
        }
        
        return report
        
    def _assess_principles_compliance(self, violations: List[EthicsViolation]) -> Dict[str, str]:
        """Assess compliance with ethics principles."""
        principle_status = {}
        
        for principle in self.ethics_principles:
            violation_count = 0
            for violation in violations:
                # Simple keyword matching for principle assessment
                if any(word in violation.description.lower() 
                      for word in principle.lower().split()):
                    violation_count += 1
                    
            if violation_count == 0:
                principle_status[principle] = "compliant"
            elif violation_count <= 2:
                principle_status[principle] = "minor_issues"
            else:
                principle_status[principle] = "needs_attention"
                
        return principle_status
        
    def _generate_ethics_recommendations(self, violations: List[EthicsViolation]) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        violation_types = [v.violation_type for v in violations]
        type_counts = {vtype: violation_types.count(vtype) for vtype in set(violation_types)}
        
        for vtype, count in type_counts.items():
            if count >= 3:
                if vtype == "misrepresentation":
                    recommendations.append("Implement stronger uncertainty communication protocols")
                elif vtype == "privacy_violation":
                    recommendations.append("Review and strengthen privacy protection measures")
                elif vtype == "bias_amplification":
                    recommendations.append("Enhance source quality filtering and bias detection")
                    
        if not recommendations:
            recommendations.append("Continue monitoring ethics compliance")
            
        return recommendations
        
    def _calculate_compliance_score(self, violations: List[EthicsViolation], days: int) -> float:
        """Calculate overall ethics compliance score."""
        if not violations:
            return 1.0
            
        # Weight violations by severity
        severity_weights = {"low": 1, "medium": 3, "high": 9}
        weighted_violations = sum(severity_weights.get(v.severity, 1) for v in violations)
        
        # Normalize by time period (target: <1 weighted violation per week)
        weeks = max(1, days / 7)
        target_violations = weeks * 1
        
        score = max(0.0, 1.0 - (weighted_violations / (target_violations * 2)))
        return round(score, 2)


class SafetyEthicsFramework:
    """Main framework for safety and ethics guidelines."""
    
    def __init__(self):
        self.safety_filter = SafetyFilter()
        self.privacy_protector = PrivacyProtector()
        self.information_validator = InformationValidator()
        self.ethics_monitor = EthicsMonitor()
        self.logger = logging.getLogger(f"{__name__}.SafetyEthicsFramework")
        
    def comprehensive_assessment(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive safety and ethics assessment.
        
        Args:
            query: Query or operation to assess
            context: Additional context
            
        Returns:
            Comprehensive assessment results
        """
        context = context or {}
        
        # Safety assessment
        safety = self.safety_filter.assess_query_safety(query, context)
        
        # Privacy assessment
        privacy = self.privacy_protector.assess_privacy_risk(query, context)
        
        # Overall risk level
        risk_levels = {
            SafetyLevel.SAFE: 1,
            SafetyLevel.CAUTION: 2,
            SafetyLevel.WARNING: 3,
            SafetyLevel.DANGEROUS: 4,
            SafetyLevel.BLOCKED: 5
        }
        
        privacy_risk_levels = {
            PrivacyLevel.PUBLIC: 1,
            PrivacyLevel.PERSONAL: 2,
            PrivacyLevel.SENSITIVE: 3,
            PrivacyLevel.CONFIDENTIAL: 4,
            PrivacyLevel.RESTRICTED: 5
        }
        
        max_risk = max(risk_levels[safety.level], privacy_risk_levels[privacy.level])
        
        assessment = {
            "overall_risk_level": max_risk,
            "safe_to_proceed": max_risk <= 2,
            "requires_supervision": max_risk == 3,
            "blocked": max_risk >= 4,
            "safety_assessment": {
                "level": safety.level.value,
                "reasons": safety.reasons,
                "recommendations": safety.recommendations,
                "alternatives": safety.alternative_approaches
            },
            "privacy_assessment": {
                "level": privacy.level.value,
                "data_types": privacy.identified_data_types,
                "risks": privacy.risks,
                "protection_measures": privacy.protection_measures
            },
            "guidelines": self._generate_usage_guidelines(safety, privacy),
            "timestamp": datetime.now().isoformat()
        }
        
        return assessment
        
    def validate_research_output(self, content: str, sources: List[Dict[str, Any]],
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate research output for accuracy and ethics.
        
        Args:
            content: Research content to validate
            sources: Sources used
            context: Research context
            
        Returns:
            Validation results
        """
        context = context or {}
        
        # Information validation
        info_validation = self.information_validator.validate_information(content, sources)
        
        # Ethics monitoring
        ethics_violations = self.ethics_monitor.monitor_research_ethics(
            "information_synthesis", context, content
        )
        
        # Content safety filtering
        filtered_content, removed_sections = self.safety_filter.filter_content(content)
        
        validation = {
            "information_reliability": {
                "level": info_validation.reliability.value,
                "confidence": info_validation.confidence_level,
                "verification_methods": info_validation.verification_methods,
                "notes": info_validation.verification_notes
            },
            "ethics_compliance": {
                "violations_detected": len(ethics_violations),
                "violation_details": [
                    {
                        "type": v.violation_type,
                        "severity": v.severity,
                        "description": v.description,
                        "action": v.recommended_action
                    }
                    for v in ethics_violations
                ],
                "compliance_status": "compliant" if not ethics_violations else "violations_detected"
            },
            "content_safety": {
                "content_modified": len(removed_sections) > 0,
                "removed_sections_count": len(removed_sections),
                "safety_issues": removed_sections
            },
            "overall_validation": {
                "approved": len(ethics_violations) == 0 and len(removed_sections) == 0,
                "requires_review": len(ethics_violations) > 0 or len(removed_sections) > 0,
                "confidence_in_output": info_validation.confidence_level
            }
        }
        
        return validation
        
    def _generate_usage_guidelines(self, safety: SafetyAssessment, 
                                 privacy: PrivacyAssessment) -> List[str]:
        """Generate usage guidelines based on assessments."""
        guidelines = []
        
        # Safety guidelines
        if safety.level == SafetyLevel.BLOCKED:
            guidelines.append("BLOCKED: Do not proceed with this query")
        elif safety.level == SafetyLevel.DANGEROUS:
            guidelines.append("DANGEROUS: Seek appropriate authorization before proceeding")
        elif safety.level == SafetyLevel.WARNING:
            guidelines.append("WARNING: Proceed with extreme caution and supervision")
        elif safety.level == SafetyLevel.CAUTION:
            guidelines.append("CAUTION: Verify all information from authoritative sources")
            
        # Privacy guidelines
        if privacy.level in [PrivacyLevel.RESTRICTED, PrivacyLevel.CONFIDENTIAL]:
            guidelines.append("PRIVACY: Implement maximum data protection measures")
        elif privacy.level == PrivacyLevel.SENSITIVE:
            guidelines.append("PRIVACY: Use privacy-preserving techniques")
            
        # General guidelines
        guidelines.extend([
            "Always cite sources and indicate confidence levels",
            "Distinguish between verified facts and preliminary information",
            "Respect rate limits and terms of service",
            "Report any security issues through proper channels"
        ])
        
        return guidelines
        
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate overall compliance report."""
        ethics_report = self.ethics_monitor.generate_ethics_report()
        
        report = {
            "framework_version": "1.0",
            "report_timestamp": datetime.now().isoformat(),
            "ethics_compliance": ethics_report,
            "safety_guidelines_active": True,
            "privacy_protection_active": True,
            "information_validation_active": True,
            "overall_status": "operational",
            "recommendations": [
                "Continue regular compliance monitoring",
                "Update safety patterns based on new threats",
                "Review ethics guidelines quarterly"
            ]
        }
        
        return report


# Example usage
def main():
    """Example usage of the Safety and Ethics Framework."""
    framework = SafetyEthicsFramework()
    
    # Test queries
    test_queries = [
        "How to improve cybersecurity in healthcare systems?",
        "What are the latest treatments for diabetes?",
        "How to create malware for educational purposes?",
        "John Smith's personal email address and phone number"
    ]
    
    print("Safety and Ethics Assessment Results:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        assessment = framework.comprehensive_assessment(query)
        
        print(f"Overall Risk Level: {assessment['overall_risk_level']}/5")
        print(f"Safe to Proceed: {assessment['safe_to_proceed']}")
        print(f"Safety Level: {assessment['safety_assessment']['level']}")
        print(f"Privacy Level: {assessment['privacy_assessment']['level']}")
        
        if assessment['safety_assessment']['recommendations']:
            print(f"Recommendations: {assessment['safety_assessment']['recommendations'][0]}")
            
    # Generate compliance report
    print("\n" + "=" * 60)
    print("Compliance Report:")
    compliance = framework.generate_compliance_report()
    print(f"Ethics Compliance Score: {compliance['ethics_compliance']['compliance_score']}")
    print(f"Overall Status: {compliance['overall_status']}")


if __name__ == "__main__":
    main()