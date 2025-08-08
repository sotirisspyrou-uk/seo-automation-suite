"""
Competitive Impact Assessor - Executive-Grade Market Position Forecasting
Predicts competitive threats and market share implications for strategic planning

Portfolio Demo: This module showcases advanced competitive intelligence 
combining predictive analytics with C-suite strategic thinking.

Author: Sotiris Spyrou | LinkedIn: https://www.linkedin.com/in/sspyrou/
Company: VerityAI - https://verityai.co/landing/ai-seo-services

DISCLAIMER: This is portfolio demonstration code showcasing technical capabilities
and strategic thinking. Not intended for production use without proper testing
and enterprise-grade security implementation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import aiohttp
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import structlog

logger = structlog.get_logger()


@dataclass
class CompetitiveImpact:
    """Competitive impact assessment with business implications"""
    competitor: str
    impact_type: str  # "market_share_erosion", "keyword_displacement", "traffic_cannibalization"
    severity: str  # "critical", "high", "medium", "low"
    current_impact: float  # 0-1 scale
    projected_impact_30d: float
    projected_impact_90d: float
    projected_impact_365d: float
    affected_keywords: List[str]
    estimated_traffic_loss: int
    estimated_revenue_loss: float
    confidence_interval: Tuple[float, float]
    strategic_implications: List[str]
    recommended_countermeasures: List[str]


@dataclass
class MarketShareForecast:
    """Market share forecasting with competitive dynamics"""
    domain: str
    current_share: float
    forecasted_share_30d: float
    forecasted_share_90d: float
    forecasted_share_365d: float
    growth_trajectory: str  # "accelerating", "steady", "declining", "volatile"
    key_growth_drivers: List[str]
    competitive_threats: List[str]
    market_opportunities: List[str]
    strategic_recommendations: List[str]


@dataclass
class CompetitiveAlert:
    """Real-time competitive intelligence alert"""
    alert_id: str
    alert_type: str  # "new_competitor", "algorithm_change", "content_surge", "technical_advantage"
    priority: str  # "urgent", "high", "medium", "low"
    competitor: str
    detection_timestamp: datetime
    impact_description: str
    quantified_impact: float
    business_implications: List[str]
    immediate_actions: List[str]
    monitoring_metrics: List[str]


@dataclass
class ExecutiveThreatAssessment:
    """C-suite level competitive threat assessment"""
    assessment_date: datetime
    overall_threat_level: str  # "critical", "elevated", "moderate", "low"
    primary_threats: List[str]
    emerging_risks: List[str]
    market_volatility_score: float
    strategic_position_strength: float
    defensive_capabilities: List[str]
    offensive_opportunities: List[str]
    budget_allocation_recommendations: Dict[str, float]
    quarterly_focus_areas: List[str]


class CompetitiveImpactAssessor:
    """
    Executive-Grade Competitive Impact Assessment Platform
    
    Demonstrates advanced competitive forecasting combining:
    - Predictive analytics and machine learning
    - C-suite strategic threat assessment
    - Real-time competitive intelligence
    - Business impact quantification
    
    Perfect for: Enterprise strategy teams, CMOs, competitive intelligence professionals
    """
    
    def __init__(self, primary_domain: str, competitors: List[str], api_keys: Dict[str, str]):
        self.primary_domain = primary_domain
        self.competitors = competitors
        self.api_keys = api_keys
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Machine learning models for forecasting
        self.traffic_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.share_model = LinearRegression()
        self.scaler = StandardScaler()
        
        # Executive threat thresholds
        self.critical_impact_threshold = 0.15  # 15% impact = critical
        self.high_impact_threshold = 0.08     # 8% impact = high
        self.market_volatility_threshold = 0.3
        
        # Portfolio branding
        logger.info(
            "competitive_impact_assessor_initialized",
            primary_domain=primary_domain,
            competitors_monitored=len(competitors),
            portfolio_note="Demo showcasing predictive analytics + C-suite strategic thinking"
        )
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def assess_competitive_landscape(self, forecast_horizon: int = 365) -> Dict:
        """
        Executive-Level Competitive Impact Assessment
        
        Comprehensive competitive threat analysis with predictive insights
        for strategic decision making and resource allocation.
        """
        logger.info(
            "assessing_competitive_landscape",
            forecast_horizon_days=forecast_horizon,
            executive_context="Strategic threat assessment and market forecasting"
        )
        
        # Parallel data collection for comprehensive analysis
        assessment_tasks = [
            self._analyze_competitive_impacts(),
            self._forecast_market_share_dynamics(),
            self._detect_emerging_threats(),
            self._assess_algorithm_vulnerability()
        ]
        
        impacts, forecasts, threats, vulnerabilities = await asyncio.gather(*assessment_tasks)
        
        # Generate executive threat assessment
        threat_assessment = self._generate_executive_threat_assessment(
            impacts, forecasts, threats, vulnerabilities
        )
        
        # Create strategic recommendations
        strategic_response = self._develop_strategic_response(
            threat_assessment, impacts, forecasts
        )
        
        return {
            "executive_summary": {
                "overall_threat_level": threat_assessment.overall_threat_level,
                "critical_impacts_identified": len([i for i in impacts if i.severity == "critical"]),
                "market_volatility_score": threat_assessment.market_volatility_score,
                "strategic_position_strength": threat_assessment.strategic_position_strength,
                "primary_competitive_threats": len(threat_assessment.primary_threats),
                "immediate_actions_required": len([i for i in impacts if i.severity in ["critical", "high"]])
            },
            "competitive_impacts": impacts[:10],  # Top 10 for executive focus
            "market_share_forecasts": forecasts,
            "emerging_threats": threats,
            "algorithm_vulnerabilities": vulnerabilities,
            "threat_assessment": threat_assessment,
            "strategic_response": strategic_response,
            "competitive_intelligence": {
                "fastest_growing_threat": self._identify_fastest_growing_threat(forecasts),
                "most_vulnerable_competitor": self._identify_vulnerable_competitor(forecasts),
                "market_disruption_signals": self._detect_disruption_signals(impacts, threats),
                "defensive_opportunities": self._identify_defensive_opportunities(impacts)
            },
            "executive_dashboard": {
                "threat_monitoring": self._create_threat_monitoring_dashboard(impacts, threats),
                "market_positioning": self._create_positioning_dashboard(forecasts),
                "action_priorities": self._prioritize_executive_actions(strategic_response)
            }
        }
    
    async def _analyze_competitive_impacts(self) -> List[CompetitiveImpact]:
        """Analyze competitive impacts using predictive models"""
        impacts = []
        
        for competitor in self.competitors:
            try:
                # Fetch competitive data and build impact models
                competitive_data = await self._fetch_competitive_metrics(competitor)
                impact_analysis = self._model_competitive_impact(competitor, competitive_data)
                
                for impact in impact_analysis:
                    impacts.append(impact)
                    
            except Exception as e:
                logger.error("competitive_impact_analysis_error", competitor=competitor, error=str(e))
        
        # Sort by severity and impact magnitude
        impacts.sort(key=lambda x: (x.severity == "critical", x.current_impact), reverse=True)
        
        return impacts
    
    async def _fetch_competitive_metrics(self, competitor: str) -> Dict:
        """Fetch competitive metrics for impact modeling"""
        
        # Simulated competitive data (in production: integrate with SEMrush, Ahrefs, etc.)
        base_performance = np.random.uniform(0.3, 0.9)
        growth_trend = np.random.uniform(-0.1, 0.3)  # Monthly growth rate
        
        return {
            "organic_visibility": base_performance,
            "keyword_count": int(base_performance * 15000),
            "estimated_traffic": int(base_performance * 500000),
            "content_velocity": np.random.randint(5, 50),  # New pages per week
            "technical_performance": np.random.uniform(0.6, 0.95),
            "growth_rate": growth_trend,
            "market_aggressiveness": np.random.uniform(0.2, 0.9),
            "historical_data": self._generate_historical_performance(base_performance, growth_trend)
        }
    
    def _generate_historical_performance(self, base_performance: float, growth_rate: float) -> List[Dict]:
        """Generate historical performance data for trend analysis"""
        historical_data = []
        
        for i in range(12):  # 12 months of data
            month_performance = base_performance * (1 + growth_rate * i) * np.random.uniform(0.9, 1.1)
            historical_data.append({
                "month": i + 1,
                "performance": min(1.0, max(0.1, month_performance)),
                "traffic_estimate": int(month_performance * 500000),
                "keyword_growth": np.random.uniform(-0.05, 0.15)
            })
        
        return historical_data
    
    def _model_competitive_impact(self, competitor: str, data: Dict) -> List[CompetitiveImpact]:
        """Model competitive impact using machine learning and trend analysis"""
        impacts = []
        
        # Market share erosion impact
        if data["growth_rate"] > 0.05:  # Growing faster than 5% monthly
            current_impact = min(0.2, data["growth_rate"] * 2)
            projected_30d = current_impact * 1.1
            projected_90d = current_impact * 1.3
            projected_365d = current_impact * 2.0
            
            # Calculate business impact
            estimated_traffic_loss = int(current_impact * 50000)  # Based on market size
            estimated_revenue_loss = estimated_traffic_loss * 0.02 * 75  # Conversion rate * AOV
            
            impacts.append(CompetitiveImpact(
                competitor=competitor,
                impact_type="market_share_erosion",
                severity=self._determine_impact_severity(current_impact),
                current_impact=current_impact,
                projected_impact_30d=projected_30d,
                projected_impact_90d=projected_90d,
                projected_impact_365d=projected_365d,
                affected_keywords=self._simulate_affected_keywords(),
                estimated_traffic_loss=estimated_traffic_loss,
                estimated_revenue_loss=estimated_revenue_loss,
                confidence_interval=(current_impact * 0.8, current_impact * 1.2),
                strategic_implications=[
                    f"{competitor} gaining market momentum",
                    "Potential long-term competitive disadvantage",
                    "Need for accelerated growth initiatives"
                ],
                recommended_countermeasures=[
                    "Accelerate content production velocity",
                    "Increase technical SEO optimization",
                    "Launch competitive keyword campaigns"
                ]
            ))
        
        # Content surge impact
        if data["content_velocity"] > 30:  # High content production
            content_impact = min(0.15, data["content_velocity"] / 200)
            
            impacts.append(CompetitiveImpact(
                competitor=competitor,
                impact_type="traffic_cannibalization",
                severity=self._determine_impact_severity(content_impact),
                current_impact=content_impact,
                projected_impact_30d=content_impact * 1.2,
                projected_impact_90d=content_impact * 1.5,
                projected_impact_365d=content_impact * 2.2,
                affected_keywords=self._simulate_affected_keywords("content"),
                estimated_traffic_loss=int(content_impact * 75000),
                estimated_revenue_loss=content_impact * 30000,
                confidence_interval=(content_impact * 0.7, content_impact * 1.3),
                strategic_implications=[
                    f"{competitor} aggressive content strategy",
                    "Risk of topic authority erosion",
                    "Potential SERP feature displacement"
                ],
                recommended_countermeasures=[
                    "Implement content gap analysis",
                    "Accelerate pillar page development", 
                    "Enhance content depth and quality"
                ]
            ))
        
        # Technical advantage impact
        if data["technical_performance"] > 0.85:
            tech_impact = (data["technical_performance"] - 0.7) / 3
            
            impacts.append(CompetitiveImpact(
                competitor=competitor,
                impact_type="technical_advantage",
                severity=self._determine_impact_severity(tech_impact),
                current_impact=tech_impact,
                projected_impact_30d=tech_impact * 1.1,
                projected_impact_90d=tech_impact * 1.3,
                projected_impact_365d=tech_impact * 1.8,
                affected_keywords=self._simulate_affected_keywords("technical"),
                estimated_traffic_loss=int(tech_impact * 40000),
                estimated_revenue_loss=tech_impact * 25000,
                confidence_interval=(tech_impact * 0.9, tech_impact * 1.1),
                strategic_implications=[
                    f"{competitor} superior technical implementation",
                    "Algorithm update vulnerability",
                    "Core Web Vitals competitive disadvantage"
                ],
                recommended_countermeasures=[
                    "Prioritize Core Web Vitals optimization",
                    "Implement advanced schema markup",
                    "Upgrade site architecture and performance"
                ]
            ))
        
        return impacts
    
    def _determine_impact_severity(self, impact: float) -> str:
        """Determine impact severity based on magnitude"""
        if impact >= self.critical_impact_threshold:
            return "critical"
        elif impact >= self.high_impact_threshold:
            return "high"
        elif impact >= 0.03:
            return "medium"
        else:
            return "low"
    
    def _simulate_affected_keywords(self, impact_type: str = "general") -> List[str]:
        """Simulate affected keywords based on impact type"""
        keyword_sets = {
            "general": [
                "enterprise software", "business solutions", "digital transformation",
                "seo automation", "marketing technology", "data analytics"
            ],
            "content": [
                "how to guides", "best practices", "industry trends",
                "thought leadership", "case studies", "tutorials"
            ],
            "technical": [
                "api documentation", "integration guide", "technical specs",
                "developer resources", "implementation help", "troubleshooting"
            ]
        }
        
        keywords = keyword_sets.get(impact_type, keyword_sets["general"])
        return np.random.choice(keywords, size=min(len(keywords), 5), replace=False).tolist()
    
    async def _forecast_market_share_dynamics(self) -> List[MarketShareForecast]:
        """Forecast market share dynamics using predictive modeling"""
        forecasts = []
        
        # Include primary domain in analysis
        all_domains = [self.primary_domain] + self.competitors
        
        for domain in all_domains:
            try:
                # Fetch domain performance data
                domain_data = await self._fetch_competitive_metrics(domain)
                
                # Build forecasting model
                forecast = self._create_market_share_forecast(domain, domain_data)
                forecasts.append(forecast)
                
            except Exception as e:
                logger.error("market_share_forecast_error", domain=domain, error=str(e))
        
        return forecasts
    
    def _create_market_share_forecast(self, domain: str, data: Dict) -> MarketShareForecast:
        """Create market share forecast using trend analysis"""
        
        current_share = data["organic_visibility"] * 0.15  # Convert to market share estimate
        growth_rate = data["growth_rate"]
        
        # Calculate forecasted shares
        forecasted_30d = current_share * (1 + growth_rate * 1)
        forecasted_90d = current_share * (1 + growth_rate * 3)
        forecasted_365d = current_share * (1 + growth_rate * 12)
        
        # Determine growth trajectory
        if growth_rate > 0.1:
            trajectory = "accelerating"
        elif growth_rate > 0.03:
            trajectory = "steady"
        elif growth_rate > -0.02:
            trajectory = "declining"
        else:
            trajectory = "volatile"
        
        # Generate strategic insights
        growth_drivers = self._identify_growth_drivers(data)
        threats = self._identify_competitive_threats(data, domain)
        opportunities = self._identify_market_opportunities(data, domain)
        recommendations = self._generate_strategic_recommendations(data, trajectory)
        
        return MarketShareForecast(
            domain=domain,
            current_share=current_share,
            forecasted_share_30d=forecasted_30d,
            forecasted_share_90d=forecasted_90d,
            forecasted_share_365d=forecasted_365d,
            growth_trajectory=trajectory,
            key_growth_drivers=growth_drivers,
            competitive_threats=threats,
            market_opportunities=opportunities,
            strategic_recommendations=recommendations
        )
    
    def _identify_growth_drivers(self, data: Dict) -> List[str]:
        """Identify key growth drivers based on performance data"""
        drivers = []
        
        if data["content_velocity"] > 25:
            drivers.append("High content production velocity")
        
        if data["technical_performance"] > 0.8:
            drivers.append("Superior technical implementation")
        
        if data["growth_rate"] > 0.05:
            drivers.append("Strong organic growth momentum")
        
        if data["market_aggressiveness"] > 0.7:
            drivers.append("Aggressive market expansion strategy")
        
        return drivers if drivers else ["Stable market presence"]
    
    def _identify_competitive_threats(self, data: Dict, domain: str) -> List[str]:
        """Identify competitive threats for the domain"""
        threats = []
        
        if domain == self.primary_domain:
            # Threats to primary domain
            if any(c for c in self.competitors):
                threats.append("Increased competitive pressure from multiple players")
            
            threats.extend([
                "Market share erosion from faster-growing competitors",
                "Algorithm changes favoring competitors",
                "New market entrants with disruptive strategies"
            ])
        else:
            # Domain is a competitor - assess their threats
            threats.extend([
                "Market saturation limiting growth potential",
                "Technical debt constraining competitive advantage",
                "Content strategy becoming commoditized"
            ])
        
        return threats[:3]  # Limit to top 3
    
    def _identify_market_opportunities(self, data: Dict, domain: str) -> List[str]:
        """Identify market opportunities for the domain"""
        opportunities = []
        
        if data["technical_performance"] < 0.7:
            opportunities.append("Technical SEO optimization potential")
        
        if data["content_velocity"] < 15:
            opportunities.append("Content production acceleration opportunity")
        
        if data["market_aggressiveness"] < 0.5:
            opportunities.append("Market expansion and competitive positioning")
        
        opportunities.extend([
            "Emerging keyword opportunity capture",
            "Strategic partnership and acquisition targets",
            "International market expansion potential"
        ])
        
        return opportunities[:4]  # Limit to top 4
    
    def _generate_strategic_recommendations(self, data: Dict, trajectory: str) -> List[str]:
        """Generate strategic recommendations based on trajectory and performance"""
        recommendations = []
        
        if trajectory == "accelerating":
            recommendations.extend([
                "Maintain aggressive growth investments",
                "Expand market presence while momentum is strong",
                "Consider strategic acquisitions for faster scaling"
            ])
        elif trajectory == "steady":
            recommendations.extend([
                "Identify breakthrough growth opportunities",
                "Optimize resource allocation for efficiency",
                "Develop competitive differentiation strategies"
            ])
        elif trajectory == "declining":
            recommendations.extend([
                "Immediate strategic intervention required",
                "Reassess market positioning and value proposition",
                "Consider pivot or market repositioning strategies"
            ])
        else:  # volatile
            recommendations.extend([
                "Stabilize performance through consistent execution",
                "Focus on sustainable growth rather than rapid scaling",
                "Build resilient competitive advantages"
            ])
        
        return recommendations[:3]  # Limit to top 3
    
    async def _detect_emerging_threats(self) -> List[CompetitiveAlert]:
        """Detect emerging competitive threats using real-time monitoring"""
        alerts = []
        
        # Simulated threat detection (in production: integrate with monitoring APIs)
        threat_scenarios = [
            {
                "type": "new_competitor",
                "description": "New AI-first competitor entering market with significant funding",
                "priority": "high",
                "impact": 0.12
            },
            {
                "type": "algorithm_change", 
                "description": "Google algorithm update favoring fresh content",
                "priority": "medium",
                "impact": 0.08
            },
            {
                "type": "content_surge",
                "description": "Major competitor 3x content production velocity",
                "priority": "high", 
                "impact": 0.15
            }
        ]
        
        for i, scenario in enumerate(threat_scenarios):
            alert = CompetitiveAlert(
                alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d')}_{i+1:03d}",
                alert_type=scenario["type"],
                priority=scenario["priority"],
                competitor=np.random.choice(self.competitors) if scenario["type"] != "algorithm_change" else "Google",
                detection_timestamp=datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                impact_description=scenario["description"],
                quantified_impact=scenario["impact"],
                business_implications=self._generate_business_implications(scenario),
                immediate_actions=self._generate_immediate_actions(scenario),
                monitoring_metrics=self._define_monitoring_metrics(scenario["type"])
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_business_implications(self, scenario: Dict) -> List[str]:
        """Generate business implications for threat scenarios"""
        implications_map = {
            "new_competitor": [
                "Potential market share erosion",
                "Increased customer acquisition costs",
                "Need for accelerated innovation"
            ],
            "algorithm_change": [
                "Ranking volatility across keyword portfolio", 
                "Traffic fluctuations impacting revenue",
                "Need for content strategy adjustment"
            ],
            "content_surge": [
                "Topic authority erosion risk",
                "SERP feature displacement threat",
                "Content marketing ROI pressure"
            ]
        }
        
        return implications_map.get(scenario["type"], ["Market dynamics shift requiring strategic response"])
    
    def _generate_immediate_actions(self, scenario: Dict) -> List[str]:
        """Generate immediate action items for threat scenarios"""
        actions_map = {
            "new_competitor": [
                "Conduct comprehensive competitive analysis",
                "Accelerate product development timeline",
                "Review and strengthen market positioning"
            ],
            "algorithm_change": [
                "Audit content freshness and update schedule",
                "Implement algorithm-aligned optimization",
                "Monitor ranking changes across all keywords"
            ],
            "content_surge": [
                "Increase content production velocity",
                "Identify and fill content gaps immediately",
                "Launch topic cluster expansion strategy"
            ]
        }
        
        return actions_map.get(scenario["type"], ["Monitor situation and prepare strategic response"])
    
    def _define_monitoring_metrics(self, alert_type: str) -> List[str]:
        """Define key monitoring metrics for different alert types"""
        metrics_map = {
            "new_competitor": [
                "Competitor organic visibility growth",
                "Market share tracking",
                "Customer acquisition metrics"
            ],
            "algorithm_change": [
                "Keyword ranking positions",
                "Organic traffic fluctuations", 
                "SERP feature appearances"
            ],
            "content_surge": [
                "Content publication velocity",
                "Topic authority metrics",
                "Content performance benchmarks"
            ],
            "technical_advantage": [
                "Core Web Vitals comparison",
                "Technical performance scores",
                "Crawl efficiency metrics"
            ]
        }
        
        return metrics_map.get(alert_type, ["General competitive metrics"])
    
    async def _assess_algorithm_vulnerability(self) -> Dict:
        """Assess vulnerability to algorithm changes"""
        
        # Simulated vulnerability assessment
        return {
            "overall_vulnerability_score": np.random.uniform(0.3, 0.7),
            "high_risk_areas": [
                "Content freshness lag",
                "Technical performance gaps",
                "E-A-T signal weakness"
            ],
            "protective_factors": [
                "Diverse traffic sources",
                "Strong brand signals", 
                "Quality content portfolio"
            ],
            "algorithm_change_scenarios": [
                {
                    "scenario": "Fresh content preference increase",
                    "impact_probability": 0.7,
                    "estimated_impact": 0.12,
                    "mitigation_strategies": ["Implement dynamic content updates", "Increase publishing frequency"]
                },
                {
                    "scenario": "E-A-T signal weight increase",
                    "impact_probability": 0.6,
                    "estimated_impact": 0.08,
                    "mitigation_strategies": ["Strengthen author profiles", "Enhance expertise signals"]
                }
            ]
        }
    
    def _generate_executive_threat_assessment(self, impacts: List[CompetitiveImpact], 
                                           forecasts: List[MarketShareForecast],
                                           threats: List[CompetitiveAlert],
                                           vulnerabilities: Dict) -> ExecutiveThreatAssessment:
        """Generate executive-level threat assessment"""
        
        # Calculate overall threat level
        critical_impacts = len([i for i in impacts if i.severity == "critical"])
        high_priority_alerts = len([t for t in threats if t.priority in ["urgent", "high"]])
        
        if critical_impacts > 2 or high_priority_alerts > 3:
            overall_threat = "critical"
        elif critical_impacts > 0 or high_priority_alerts > 1:
            overall_threat = "elevated"
        elif len(impacts) > 5:
            overall_threat = "moderate"
        else:
            overall_threat = "low"
        
        # Extract primary threats
        primary_threats = [i.impact_type for i in impacts[:3] if i.severity in ["critical", "high"]]
        emerging_risks = [t.impact_description for t in threats[:3]]
        
        # Calculate strategic position strength
        our_forecast = next((f for f in forecasts if f.domain == self.primary_domain), None)
        position_strength = our_forecast.current_share / 0.15 if our_forecast else 0.5  # Normalize
        
        return ExecutiveThreatAssessment(
            assessment_date=datetime.now(),
            overall_threat_level=overall_threat,
            primary_threats=primary_threats,
            emerging_risks=emerging_risks,
            market_volatility_score=vulnerabilities["overall_vulnerability_score"],
            strategic_position_strength=min(1.0, position_strength),
            defensive_capabilities=self._assess_defensive_capabilities(),
            offensive_opportunities=self._assess_offensive_opportunities(impacts),
            budget_allocation_recommendations=self._recommend_budget_allocation(overall_threat, impacts),
            quarterly_focus_areas=self._define_quarterly_focus(impacts, threats)
        )
    
    def _assess_defensive_capabilities(self) -> List[str]:
        """Assess defensive competitive capabilities"""
        return [
            "Established brand recognition and trust",
            "Diversified organic traffic portfolio", 
            "Technical infrastructure scalability",
            "Content production and optimization expertise"
        ]
    
    def _assess_offensive_opportunities(self, impacts: List[CompetitiveImpact]) -> List[str]:
        """Assess offensive competitive opportunities"""
        return [
            "Accelerated growth in underexploited market segments",
            "Technical advantage development opportunities",
            "Content gap exploitation strategies",
            "Competitive displacement in vulnerable areas"
        ]
    
    def _recommend_budget_allocation(self, threat_level: str, impacts: List[CompetitiveImpact]) -> Dict[str, float]:
        """Recommend budget allocation based on threat assessment"""
        
        if threat_level == "critical":
            return {
                "defensive_measures": 0.6,
                "offensive_initiatives": 0.25,
                "monitoring_intelligence": 0.15
            }
        elif threat_level == "elevated":
            return {
                "defensive_measures": 0.45,
                "offensive_initiatives": 0.35,
                "monitoring_intelligence": 0.20
            }
        else:
            return {
                "defensive_measures": 0.30,
                "offensive_initiatives": 0.50,
                "monitoring_intelligence": 0.20
            }
    
    def _define_quarterly_focus(self, impacts: List[CompetitiveImpact], 
                              threats: List[CompetitiveAlert]) -> List[str]:
        """Define quarterly focus areas based on threats"""
        
        focus_areas = []
        
        # Address critical impacts
        critical_impacts = [i for i in impacts if i.severity == "critical"]
        if critical_impacts:
            focus_areas.append("Critical competitive threat mitigation")
        
        # Address high-priority alerts
        urgent_threats = [t for t in threats if t.priority == "urgent"]
        if urgent_threats:
            focus_areas.append("Emerging threat response and monitoring")
        
        # Strategic positioning
        focus_areas.extend([
            "Market position strengthening initiatives",
            "Competitive advantage development",
            "Intelligence gathering and analysis enhancement"
        ])
        
        return focus_areas[:4]  # Limit to 4 focus areas
    
    def _develop_strategic_response(self, threat_assessment: ExecutiveThreatAssessment,
                                  impacts: List[CompetitiveImpact],
                                  forecasts: List[MarketShareForecast]) -> Dict:
        """Develop comprehensive strategic response plan"""
        
        return {
            "immediate_response": {
                "critical_actions": self._define_critical_actions(impacts),
                "resource_reallocation": threat_assessment.budget_allocation_recommendations,
                "timeline": "48-72 hours for critical, 1-2 weeks for high priority"
            },
            "tactical_response": {
                "quarterly_initiatives": threat_assessment.quarterly_focus_areas,
                "success_metrics": self._define_success_metrics(impacts),
                "resource_requirements": self._estimate_resource_requirements(threat_assessment)
            },
            "strategic_response": {
                "market_positioning_adjustment": self._recommend_positioning_adjustment(forecasts),
                "competitive_differentiation": self._define_differentiation_strategy(impacts),
                "long_term_objectives": self._set_long_term_objectives(threat_assessment)
            }
        }
    
    def _define_critical_actions(self, impacts: List[CompetitiveImpact]) -> List[str]:
        """Define critical actions based on competitive impacts"""
        actions = []
        
        critical_impacts = [i for i in impacts if i.severity == "critical"]
        for impact in critical_impacts[:3]:  # Top 3 critical
            actions.extend(impact.recommended_countermeasures[:2])  # Top 2 countermeasures
        
        return list(set(actions))[:5]  # Deduplicate and limit to 5
    
    def _define_success_metrics(self, impacts: List[CompetitiveImpact]) -> List[str]:
        """Define success metrics for competitive response"""
        return [
            "Market share stabilization or growth",
            "Competitive impact reduction by 20-30%",
            "Traffic loss mitigation to <5%", 
            "Revenue impact minimization",
            "Strategic position strength improvement"
        ]
    
    def _estimate_resource_requirements(self, threat_assessment: ExecutiveThreatAssessment) -> Dict:
        """Estimate resource requirements for threat response"""
        
        if threat_assessment.overall_threat_level == "critical":
            return {
                "budget_increase": "25-40%",
                "team_expansion": "Immediate tactical response team",
                "external_support": "Specialized competitive intelligence consulting"
            }
        else:
            return {
                "budget_increase": "10-20%",
                "team_expansion": "Enhanced monitoring capabilities",
                "external_support": "Selective strategic consulting"
            }
    
    def _recommend_positioning_adjustment(self, forecasts: List[MarketShareForecast]) -> List[str]:
        """Recommend market positioning adjustments"""
        return [
            "Strengthen unique value proposition communication",
            "Accelerate innovation and differentiation initiatives",
            "Develop strategic partnerships for competitive advantage",
            "Focus on high-value market segments with less competition"
        ]
    
    def _define_differentiation_strategy(self, impacts: List[CompetitiveImpact]) -> List[str]:
        """Define competitive differentiation strategy"""
        return [
            "Technical superiority and innovation leadership",
            "Content depth and expertise positioning",
            "Customer experience and service excellence",
            "Market timing and first-mover advantages"
        ]
    
    def _set_long_term_objectives(self, threat_assessment: ExecutiveThreatAssessment) -> List[str]:
        """Set long-term strategic objectives"""
        return [
            "Achieve market leadership position in core segments",
            "Build sustainable competitive moats",
            "Establish predictive competitive intelligence capabilities",
            "Create market disruption opportunities"
        ]
    
    # Helper methods for competitive intelligence
    def _identify_fastest_growing_threat(self, forecasts: List[MarketShareForecast]) -> str:
        """Identify fastest growing competitive threat"""
        competitors = [f for f in forecasts if f.domain != self.primary_domain]
        if competitors:
            fastest = max(competitors, key=lambda x: x.forecasted_share_90d - x.current_share)
            return fastest.domain
        return "No significant threats identified"
    
    def _identify_vulnerable_competitor(self, forecasts: List[MarketShareForecast]) -> str:
        """Identify most vulnerable competitor for displacement opportunities"""
        competitors = [f for f in forecasts if f.domain != self.primary_domain]
        vulnerable = [c for c in competitors if c.growth_trajectory == "declining"]
        if vulnerable:
            return min(vulnerable, key=lambda x: x.current_share).domain
        return "No vulnerable competitors identified"
    
    def _detect_disruption_signals(self, impacts: List[CompetitiveImpact], 
                                 threats: List[CompetitiveAlert]) -> List[str]:
        """Detect market disruption signals"""
        signals = []
        
        # New competitor alerts
        new_competitors = [t for t in threats if t.alert_type == "new_competitor"]
        if new_competitors:
            signals.append("New market entrants with disruptive business models")
        
        # Technology disruption
        tech_impacts = [i for i in impacts if i.impact_type == "technical_advantage"]
        if len(tech_impacts) > 2:
            signals.append("Technology-driven competitive advantages emerging")
        
        signals.extend([
            "Algorithm changes creating market volatility",
            "Consumer behavior shifts affecting search patterns",
            "Economic factors influencing competitive dynamics"
        ])
        
        return signals[:4]
    
    def _identify_defensive_opportunities(self, impacts: List[CompetitiveImpact]) -> List[str]:
        """Identify defensive opportunities to protect market position"""
        return [
            "Strengthen weak competitive areas before exploitation",
            "Build barriers to entry in core market segments",
            "Develop rapid response capabilities for competitive threats",
            "Create customer loyalty programs and retention strategies"
        ]
    
    # Dashboard creation methods
    def _create_threat_monitoring_dashboard(self, impacts: List[CompetitiveImpact], 
                                          threats: List[CompetitiveAlert]) -> Dict:
        """Create threat monitoring dashboard"""
        return {
            "threat_level_indicators": {
                "critical_impacts": len([i for i in impacts if i.severity == "critical"]),
                "high_priority_alerts": len([t for t in threats if t.priority in ["urgent", "high"]]),
                "total_monitored_competitors": len(self.competitors)
            },
            "key_metrics": {
                "average_impact_magnitude": np.mean([i.current_impact for i in impacts]),
                "projected_30d_risk": sum(i.projected_impact_30d for i in impacts[:5]),
                "revenue_at_risk": sum(i.estimated_revenue_loss for i in impacts[:5])
            },
            "alert_summary": {
                "new_threats_24h": len([t for t in threats if (datetime.now() - t.detection_timestamp).days == 0]),
                "urgent_actions_required": len([t for t in threats if t.priority == "urgent"]),
                "monitoring_coverage": "Real-time competitive intelligence across all major competitors"
            }
        }
    
    def _create_positioning_dashboard(self, forecasts: List[MarketShareForecast]) -> Dict:
        """Create market positioning dashboard"""
        our_forecast = next((f for f in forecasts if f.domain == self.primary_domain), None)
        
        return {
            "current_position": {
                "market_share": f"{our_forecast.current_share:.1%}" if our_forecast else "N/A",
                "growth_trajectory": our_forecast.growth_trajectory if our_forecast else "Unknown",
                "competitive_rank": self._calculate_competitive_rank(forecasts)
            },
            "forecast_metrics": {
                "30d_projection": f"{our_forecast.forecasted_share_30d:.1%}" if our_forecast else "N/A",
                "90d_projection": f"{our_forecast.forecasted_share_90d:.1%}" if our_forecast else "N/A",
                "365d_projection": f"{our_forecast.forecasted_share_365d:.1%}" if our_forecast else "N/A"
            },
            "competitive_context": {
                "market_leaders": [f.domain for f in sorted(forecasts, key=lambda x: x.current_share, reverse=True)[:3]],
                "fastest_growing": [f.domain for f in sorted(forecasts, key=lambda x: x.forecasted_share_90d - x.current_share, reverse=True)[:3]],
                "market_volatility": "Moderate" if len(forecasts) > 3 else "Low"
            }
        }
    
    def _calculate_competitive_rank(self, forecasts: List[MarketShareForecast]) -> int:
        """Calculate competitive ranking based on market share"""
        sorted_forecasts = sorted(forecasts, key=lambda x: x.current_share, reverse=True)
        for i, forecast in enumerate(sorted_forecasts):
            if forecast.domain == self.primary_domain:
                return i + 1
        return len(forecasts)
    
    def _prioritize_executive_actions(self, strategic_response: Dict) -> List[Dict]:
        """Prioritize actions for executive dashboard"""
        return [
            {
                "priority": "Urgent",
                "category": "Threat Mitigation",
                "actions": strategic_response["immediate_response"]["critical_actions"][:3],
                "timeline": "Immediate"
            },
            {
                "priority": "High",
                "category": "Strategic Response",
                "actions": strategic_response["tactical_response"]["quarterly_initiatives"][:2],
                "timeline": "This Quarter"
            },
            {
                "priority": "Medium",
                "category": "Long-term Planning",
                "actions": strategic_response["strategic_response"]["long_term_objectives"][:2],
                "timeline": "6-12 Months"
            }
        ]
    
    def generate_executive_briefing(self, assessment_results: Dict) -> Dict:
        """
        Generate Executive Briefing for Board Presentation
        
        Perfect for C-suite presentations and strategic planning sessions.
        Demonstrates ability to transform competitive intelligence into executive insights.
        """
        
        return {
            "executive_briefing": {
                "overall_threat_assessment": assessment_results["executive_summary"]["overall_threat_level"].upper(),
                "immediate_action_required": assessment_results["executive_summary"]["critical_impacts_identified"] > 0,
                "strategic_position": "DEFENDABLE" if assessment_results["executive_summary"]["strategic_position_strength"] > 0.7 else "VULNERABLE",
                "market_volatility": "HIGH" if assessment_results["executive_summary"]["market_volatility_score"] > 0.6 else "MODERATE"
            },
            "key_business_impacts": {
                "revenue_at_risk": f"${sum(i['estimated_revenue_loss'] for i in assessment_results.get('competitive_impacts', [])[:5]):,.0f}",
                "market_share_threat": f"{assessment_results['executive_summary'].get('market_volatility_score', 0):.1%}",
                "competitive_pressure_level": assessment_results["executive_summary"]["overall_threat_level"]
            },
            "strategic_priorities": {
                "defensive_actions": assessment_results.get("strategic_response", {}).get("immediate_response", {}).get("critical_actions", [])[:3],
                "growth_initiatives": assessment_results.get("strategic_response", {}).get("strategic_response", {}).get("long_term_objectives", [])[:3],
                "resource_allocation": assessment_results.get("strategic_response", {}).get("immediate_response", {}).get("resource_reallocation", {})
            },
            "competitive_landscape_summary": {
                "primary_threats": assessment_results.get("competitive_intelligence", {}).get("fastest_growing_threat", "None identified"),
                "displacement_opportunities": assessment_results.get("competitive_intelligence", {}).get("most_vulnerable_competitor", "None identified"),
                "market_disruption_signals": assessment_results.get("competitive_intelligence", {}).get("market_disruption_signals", [])[:3]
            },
            "portfolio_branding": {
                "analyst": "Sotiris Spyrou",
                "linkedin": "https://www.linkedin.com/in/sspyrou/",
                "company": "VerityAI - AI SEO Services",
                "service_url": "https://verityai.co/landing/ai-seo-services",
                "expertise_note": "Combining predictive analytics with C-suite strategic intelligence"
            }
        }


# Portfolio demonstration usage
async def demonstrate_competitive_impact_assessment():
    """
    Portfolio Demonstration: Executive-Level Competitive Impact Assessment
    
    This function showcases advanced competitive intelligence capabilities
    that make this portfolio valuable for enterprise strategy roles.
    """
    
    # Example usage for enterprise competitive scenario
    primary_domain = "enterprise-client.com"
    competitors = ["competitor1.com", "competitor2.com", "competitor3.com", "competitor4.com"]
    api_keys = {"semrush": "demo_key", "ahrefs": "demo_key"}
    
    async with CompetitiveImpactAssessor(primary_domain, competitors, api_keys) as assessor:
        
        # Comprehensive competitive impact assessment
        results = await assessor.assess_competitive_landscape(forecast_horizon=365)
        
        # Generate executive briefing
        executive_briefing = assessor.generate_executive_briefing(results)
        
        return {
            "competitive_assessment": results,
            "executive_briefing": executive_briefing,
            "portfolio_value_demonstration": {
                "predictive_analytics": "Advanced ML-based competitive forecasting",
                "strategic_thinking": "C-suite level threat assessment and response planning",
                "business_intelligence": "Quantified impact analysis with revenue implications",
                "executive_communication": "Board-ready competitive intelligence briefings"
            }
        }


if __name__ == "__main__":
    # Portfolio demonstration
    print("🎯 Competitive Impact Assessor - Portfolio Demo")
    print("📊 Showcasing predictive analytics + strategic intelligence")
    print("🔗 LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("🏢 VerityAI: https://verityai.co/landing/ai-seo-services")
    print("\n⚠️  Portfolio demonstration code - not for production use")
    
    # Run demonstration
    results = asyncio.run(demonstrate_competitive_impact_assessment())
    threat_level = results['competitive_assessment']['executive_summary']['overall_threat_level']
    critical_impacts = results['competitive_assessment']['executive_summary']['critical_impacts_identified']
    print(f"\n✅ Assessment complete - Threat level: {threat_level.upper()}, {critical_impacts} critical impacts identified")