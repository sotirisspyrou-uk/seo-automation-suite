"""
Competitor Intelligence Analyzer - Executive-Grade Competitive SEO Analysis
Real-time competitive monitoring for strategic decision making

Portfolio Demo: This module showcases executive-level competitive intelligence 
combining C-suite strategy with hands-on technical implementation.

Author: Sotiris Spyrou | LinkedIn: https://www.linkedin.com/in/sspyrou/
Company: VerityAI - https://verityai.co/landing/ai-seo-services

DISCLAIMER: This is portfolio demonstration code showcasing technical capabilities
and strategic thinking. Not intended for production use without proper testing
and enterprise-grade security implementation.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import structlog

logger = structlog.get_logger()


@dataclass
class CompetitorInsight:
    """Executive-level competitive intelligence insight"""
    competitor: str
    opportunity_type: str  # "keyword_gap", "content_gap", "technical_advantage"
    market_impact: float  # 0-1 scale
    revenue_potential: float  # estimated monthly revenue opportunity
    implementation_effort: str  # "low", "medium", "high"
    strategic_priority: str  # "critical", "high", "medium", "low"
    executive_summary: str
    detailed_analysis: Dict
    action_items: List[str]
    timeline_to_impact: str  # "immediate", "short_term", "long_term"


@dataclass
class MarketPosition:
    """Strategic market positioning analysis"""
    domain: str
    market_share_estimate: float
    visibility_score: float
    content_authority: float
    technical_performance: float
    competitive_moat_strength: float
    vulnerability_areas: List[str]
    strategic_advantages: List[str]
    growth_trajectory: str  # "accelerating", "steady", "declining"


@dataclass
class ExecutiveAlert:
    """High-priority competitive intelligence alert for C-suite"""
    alert_type: str  # "market_shift", "new_competitor", "opportunity", "threat"
    severity: str  # "critical", "high", "medium", "low"
    competitor: str
    impact_description: str
    business_implications: List[str]
    recommended_actions: List[str]
    estimated_revenue_impact: float
    confidence_level: float


class CompetitorIntelligenceAnalyzer:
    """
    Executive-Grade Competitive Intelligence Platform
    
    Demonstrates advanced SEO competitive analysis combining:
    - C-suite strategic thinking
    - Technical implementation expertise  
    - Real-time market intelligence
    - Revenue-focused insights
    
    Perfect for: Enterprise SEO teams, Marketing Directors, Fortune 500 companies
    """
    
    def __init__(self, primary_domain: str, competitors: List[str], api_keys: Dict[str, str]):
        self.primary_domain = primary_domain
        self.competitors = competitors
        self.api_keys = api_keys
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Executive KPI thresholds
        self.critical_visibility_threshold = 0.8
        self.high_opportunity_threshold = 0.7
        self.market_share_significance = 0.05
        
        # Portfolio branding
        logger.info(
            "competitor_intelligence_initialized",
            domain=primary_domain,
            competitors_count=len(competitors),
            portfolio_note="Demo showcasing C-suite strategy + technical expertise"
        )
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_competitive_landscape(self, focus_areas: Optional[List[str]] = None) -> Dict:
        """
        Executive-Level Competitive Landscape Analysis
        
        Returns strategic insights for C-suite decision making with
        actionable intelligence for market positioning.
        """
        logger.info(
            "analyzing_competitive_landscape",
            focus_areas=focus_areas or "comprehensive",
            executive_context="Strategic market positioning analysis"
        )
        
        # Parallel data collection for executive efficiency
        market_data_tasks = [
            self._analyze_competitor_positions(),
            self._identify_market_gaps(),
            self._assess_competitive_threats(),
            self._discover_strategic_opportunities()
        ]
        
        positions, gaps, threats, opportunities = await asyncio.gather(*market_data_tasks)
        
        # Executive insights synthesis
        strategic_insights = self._synthesize_executive_insights(
            positions, gaps, threats, opportunities
        )
        
        # Generate C-suite recommendations
        executive_recommendations = self._generate_executive_recommendations(
            strategic_insights
        )
        
        return {
            "executive_summary": {
                "market_position_strength": self._calculate_market_position_strength(),
                "competitive_advantage_score": self._calculate_competitive_advantage(),
                "immediate_opportunities": len([o for o in opportunities if o.timeline_to_impact == "immediate"]),
                "strategic_threats": len([t for t in threats if t.severity in ["critical", "high"]]),
                "estimated_market_opportunity": sum(o.revenue_potential for o in opportunities)
            },
            "strategic_insights": strategic_insights,
            "competitive_positions": positions,
            "market_opportunities": opportunities[:10],  # Top 10 for executive focus
            "competitive_threats": threats,
            "executive_recommendations": executive_recommendations,
            "market_intelligence": {
                "fastest_growing_competitor": self._identify_fastest_growing_competitor(positions),
                "most_vulnerable_competitor": self._identify_most_vulnerable_competitor(positions),
                "market_consolidation_trends": self._assess_market_consolidation(),
                "emerging_competition_signals": self._detect_emerging_competition()
            },
            "action_dashboard": {
                "immediate_actions": [r for r in executive_recommendations if r.get("urgency") == "immediate"],
                "quarterly_initiatives": [r for r in executive_recommendations if r.get("timeline") == "quarterly"],
                "annual_strategy": [r for r in executive_recommendations if r.get("timeline") == "annual"]
            }
        }
    
    async def _analyze_competitor_positions(self) -> List[MarketPosition]:
        """Analyze strategic market positions of all competitors"""
        positions = []
        
        for competitor in [self.primary_domain] + self.competitors:
            try:
                position = await self._assess_competitor_position(competitor)
                positions.append(position)
            except Exception as e:
                logger.error("competitor_position_error", competitor=competitor, error=str(e))
        
        return positions
    
    async def _assess_competitor_position(self, domain: str) -> MarketPosition:
        """Deep competitive position assessment"""
        
        # Multi-dimensional competitive analysis
        visibility_metrics = await self._calculate_visibility_metrics(domain)
        content_authority = await self._assess_content_authority(domain)
        technical_performance = await self._evaluate_technical_performance(domain)
        
        # Strategic advantage assessment
        competitive_moat = self._assess_competitive_moat(domain, visibility_metrics)
        vulnerability_analysis = self._identify_vulnerabilities(domain, visibility_metrics)
        
        return MarketPosition(
            domain=domain,
            market_share_estimate=visibility_metrics.get("estimated_market_share", 0),
            visibility_score=visibility_metrics.get("overall_visibility", 0),
            content_authority=content_authority,
            technical_performance=technical_performance,
            competitive_moat_strength=competitive_moat,
            vulnerability_areas=vulnerability_analysis.get("areas", []),
            strategic_advantages=vulnerability_analysis.get("advantages", []),
            growth_trajectory=self._assess_growth_trajectory(domain, visibility_metrics)
        )
    
    async def _calculate_visibility_metrics(self, domain: str) -> Dict:
        """Calculate comprehensive visibility metrics"""
        
        # Simulated advanced metrics (in production: integrate with SEMrush, Ahrefs APIs)
        base_score = np.random.uniform(0.3, 0.9)  # Demo data
        
        return {
            "overall_visibility": base_score,
            "estimated_market_share": base_score * 0.15,  # Convert to market share estimate
            "keyword_rankings": {
                "top_3_positions": int(base_score * 1000),
                "top_10_positions": int(base_score * 2500),
                "total_ranking_keywords": int(base_score * 15000)
            },
            "traffic_estimation": {
                "organic_monthly_visits": int(base_score * 500000),
                "branded_traffic_ratio": 0.3 + (base_score * 0.4)
            },
            "serp_features": {
                "featured_snippets": int(base_score * 150),
                "knowledge_panels": int(base_score * 25),
                "local_pack_appearances": int(base_score * 80)
            }
        }
    
    async def _assess_content_authority(self, domain: str) -> float:
        """Assess content authority and thought leadership"""
        
        # Content quality indicators (demo implementation)
        content_factors = {
            "content_depth_score": np.random.uniform(0.4, 0.95),
            "expertise_signals": np.random.uniform(0.3, 0.9),
            "authoritativeness_score": np.random.uniform(0.5, 0.85),
            "trustworthiness_indicators": np.random.uniform(0.6, 0.9),
            "content_freshness": np.random.uniform(0.4, 0.8)
        }
        
        # Weighted authority score
        authority_score = (
            content_factors["content_depth_score"] * 0.25 +
            content_factors["expertise_signals"] * 0.25 +
            content_factors["authoritativeness_score"] * 0.25 +
            content_factors["trustworthiness_indicators"] * 0.15 +
            content_factors["content_freshness"] * 0.1
        )
        
        return authority_score
    
    async def _evaluate_technical_performance(self, domain: str) -> float:
        """Evaluate technical SEO performance"""
        
        # Technical performance simulation
        technical_factors = {
            "core_web_vitals_score": np.random.uniform(0.5, 0.95),
            "mobile_optimization": np.random.uniform(0.7, 0.98),
            "site_architecture": np.random.uniform(0.6, 0.9),
            "schema_implementation": np.random.uniform(0.4, 0.8),
            "crawl_efficiency": np.random.uniform(0.5, 0.85)
        }
        
        return np.mean(list(technical_factors.values()))
    
    def _assess_competitive_moat(self, domain: str, visibility_metrics: Dict) -> float:
        """Assess strength of competitive moat"""
        
        # Moat strength indicators
        brand_strength = visibility_metrics.get("traffic_estimation", {}).get("branded_traffic_ratio", 0.3)
        market_position = visibility_metrics.get("estimated_market_share", 0)
        content_advantage = visibility_metrics.get("keyword_rankings", {}).get("top_3_positions", 0) / 1000
        
        moat_strength = (brand_strength * 0.4 + market_position * 10 * 0.35 + content_advantage * 0.25)
        
        return min(1.0, moat_strength)
    
    def _identify_vulnerabilities(self, domain: str, visibility_metrics: Dict) -> Dict:
        """Identify competitive vulnerabilities and advantages"""
        
        vulnerabilities = []
        advantages = []
        
        # Analyze based on performance metrics
        if visibility_metrics.get("overall_visibility", 0) < 0.6:
            vulnerabilities.append("Low overall search visibility")
        else:
            advantages.append("Strong search visibility position")
        
        if visibility_metrics.get("estimated_market_share", 0) < 0.05:
            vulnerabilities.append("Limited market share")
        else:
            advantages.append("Significant market presence")
        
        # Technical vulnerabilities
        serp_features = visibility_metrics.get("serp_features", {})
        if serp_features.get("featured_snippets", 0) < 50:
            vulnerabilities.append("Limited SERP feature optimization")
        
        return {
            "areas": vulnerabilities,
            "advantages": advantages
        }
    
    def _assess_growth_trajectory(self, domain: str, visibility_metrics: Dict) -> str:
        """Assess competitive growth trajectory"""
        
        # Growth trajectory simulation (in production: use historical data)
        growth_score = np.random.uniform(0, 1)
        
        if growth_score > 0.7:
            return "accelerating"
        elif growth_score > 0.4:
            return "steady"
        else:
            return "declining"
    
    async def _identify_market_gaps(self) -> List[CompetitorInsight]:
        """Identify strategic market gaps and opportunities"""
        
        gaps = []
        
        # Content gap analysis
        content_gaps = await self._analyze_content_gaps()
        for gap in content_gaps:
            gaps.append(CompetitorInsight(
                competitor="Market Gap",
                opportunity_type="content_gap",
                market_impact=gap.get("impact_score", 0),
                revenue_potential=gap.get("revenue_estimate", 0),
                implementation_effort="medium",
                strategic_priority="high" if gap.get("impact_score", 0) > 0.7 else "medium",
                executive_summary=f"Content opportunity: {gap.get('topic', 'Unknown')}",
                detailed_analysis=gap,
                action_items=gap.get("recommendations", []),
                timeline_to_impact="short_term"
            ))
        
        return gaps
    
    async def _analyze_content_gaps(self) -> List[Dict]:
        """Analyze content gaps in the market"""
        
        # Simulated content gap analysis
        content_opportunities = [
            {
                "topic": "AI-Enhanced SEO Automation",
                "impact_score": 0.85,
                "revenue_estimate": 25000,
                "market_coverage": 0.3,
                "recommendations": [
                    "Create comprehensive AI SEO automation guide",
                    "Develop comparison content for AI SEO tools",
                    "Build thought leadership content series"
                ]
            },
            {
                "topic": "Enterprise Technical SEO",
                "impact_score": 0.78,
                "revenue_estimate": 35000,
                "market_coverage": 0.45,
                "recommendations": [
                    "Develop enterprise SEO implementation methodology",
                    "Create technical SEO audit frameworks",
                    "Build enterprise case study content"
                ]
            },
            {
                "topic": "SEO Performance Measurement",
                "impact_score": 0.72,
                "revenue_estimate": 18000,
                "market_coverage": 0.55,
                "recommendations": [
                    "Build advanced SEO analytics guides",
                    "Create ROI measurement frameworks",
                    "Develop executive reporting templates"
                ]
            }
        ]
        
        return content_opportunities
    
    async def _assess_competitive_threats(self) -> List[ExecutiveAlert]:
        """Assess competitive threats requiring executive attention"""
        
        threats = []
        
        # Simulate threat detection
        for competitor in self.competitors:
            threat_level = np.random.uniform(0, 1)
            
            if threat_level > 0.7:
                threats.append(ExecutiveAlert(
                    alert_type="competitive_advancement",
                    severity="high",
                    competitor=competitor,
                    impact_description=f"{competitor} showing significant SEO improvements",
                    business_implications=[
                        "Potential market share erosion",
                        "Increased competition for key terms",
                        "Need for defensive strategy adjustment"
                    ],
                    recommended_actions=[
                        "Accelerate technical SEO improvements",
                        "Increase content production velocity",
                        "Review and optimize conversion funnels"
                    ],
                    estimated_revenue_impact=50000,
                    confidence_level=0.8
                ))
        
        return threats
    
    async def _discover_strategic_opportunities(self) -> List[CompetitorInsight]:
        """Discover strategic opportunities for competitive advantage"""
        
        opportunities = []
        
        # Keyword opportunity analysis
        keyword_opportunities = await self._analyze_keyword_opportunities()
        
        # Technical advantage opportunities
        technical_opportunities = await self._identify_technical_advantages()
        
        # Content strategy opportunities
        content_opportunities = await self._discover_content_opportunities()
        
        return opportunities
    
    async def _analyze_keyword_opportunities(self) -> List[Dict]:
        """Analyze keyword opportunities vs competitors"""
        
        # Simulated keyword gap analysis
        opportunities = []
        
        keyword_clusters = [
            "enterprise seo automation",
            "technical seo audit tools", 
            "seo performance analytics",
            "ai seo optimization",
            "international seo strategy"
        ]
        
        for cluster in keyword_clusters:
            opportunity = {
                "keyword_cluster": cluster,
                "search_volume": np.random.randint(5000, 50000),
                "difficulty": np.random.uniform(0.3, 0.8),
                "competitor_strength": np.random.uniform(0.2, 0.9),
                "opportunity_score": np.random.uniform(0.4, 0.9)
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_technical_advantages(self) -> List[Dict]:
        """Identify technical SEO advantages to exploit"""
        
        return [
            {
                "advantage": "Core Web Vitals Optimization",
                "competitor_weakness": "competitors_slow_loading",
                "implementation_effort": "medium",
                "expected_impact": "15-25% traffic increase"
            },
            {
                "advantage": "Advanced Schema Implementation", 
                "competitor_weakness": "limited_structured_data",
                "implementation_effort": "low",
                "expected_impact": "10-20% SERP feature wins"
            }
        ]
    
    async def _discover_content_opportunities(self) -> List[Dict]:
        """Discover content strategy opportunities"""
        
        return [
            {
                "content_type": "Executive SEO Guides",
                "market_gap": "lack_of_c_suite_content",
                "target_audience": "enterprise_decision_makers",
                "revenue_potential": 75000
            },
            {
                "content_type": "Technical Implementation Tutorials",
                "market_gap": "limited_hands_on_guidance", 
                "target_audience": "technical_implementers",
                "revenue_potential": 45000
            }
        ]
    
    def _synthesize_executive_insights(self, positions: List[MarketPosition], 
                                     gaps: List[CompetitorInsight], threats: List[ExecutiveAlert],
                                     opportunities: List[CompetitorInsight]) -> Dict:
        """Synthesize insights for executive decision making"""
        
        return {
            "market_dynamics": {
                "competitive_intensity": self._calculate_competitive_intensity(positions),
                "market_consolidation_level": self._assess_market_consolidation_level(positions),
                "innovation_pressure": self._assess_innovation_pressure(threats, opportunities)
            },
            "strategic_position": {
                "relative_strength": self._calculate_relative_strength(positions),
                "differentiation_opportunities": len(gaps),
                "defensive_priorities": len([t for t in threats if t.severity == "critical"])
            },
            "growth_vectors": {
                "organic_opportunity_value": sum(o.revenue_potential for o in opportunities),
                "market_expansion_potential": self._calculate_market_expansion_potential(gaps),
                "competitive_displacement_opportunity": self._assess_displacement_opportunity(positions)
            }
        }
    
    def _calculate_market_position_strength(self) -> float:
        """Calculate overall market position strength"""
        # Simulated calculation combining multiple factors
        return np.random.uniform(0.6, 0.9)
    
    def _calculate_competitive_advantage(self) -> float:
        """Calculate competitive advantage score"""
        # Simulated competitive advantage assessment
        return np.random.uniform(0.5, 0.85)
    
    def _calculate_competitive_intensity(self, positions: List[MarketPosition]) -> str:
        """Calculate competitive intensity level"""
        avg_visibility = np.mean([p.visibility_score for p in positions])
        if avg_visibility > 0.8:
            return "very_high"
        elif avg_visibility > 0.6:
            return "high"
        else:
            return "moderate"
    
    def _assess_market_consolidation_level(self, positions: List[MarketPosition]) -> str:
        """Assess market consolidation level"""
        # Analyze market share distribution
        market_shares = [p.market_share_estimate for p in positions]
        top_3_share = sum(sorted(market_shares, reverse=True)[:3])
        
        if top_3_share > 0.7:
            return "highly_consolidated"
        elif top_3_share > 0.5:
            return "moderately_consolidated"
        else:
            return "fragmented"
    
    def _assess_innovation_pressure(self, threats: List[ExecutiveAlert], 
                                  opportunities: List[CompetitorInsight]) -> str:
        """Assess innovation pressure in market"""
        threat_count = len([t for t in threats if t.alert_type == "new_competitor"])
        opportunity_count = len(opportunities)
        
        if threat_count > 3 or opportunity_count > 10:
            return "high"
        elif threat_count > 1 or opportunity_count > 5:
            return "moderate"
        else:
            return "low"
    
    def _calculate_relative_strength(self, positions: List[MarketPosition]) -> float:
        """Calculate relative competitive strength"""
        if not positions:
            return 0.5
        
        primary_position = next((p for p in positions if p.domain == self.primary_domain), None)
        if not primary_position:
            return 0.5
        
        competitor_avg = np.mean([p.visibility_score for p in positions if p.domain != self.primary_domain])
        relative_strength = primary_position.visibility_score / max(competitor_avg, 0.1)
        
        return min(1.0, relative_strength)
    
    def _calculate_market_expansion_potential(self, gaps: List[CompetitorInsight]) -> float:
        """Calculate market expansion potential"""
        return sum(g.market_impact * g.revenue_potential for g in gaps) / 10000
    
    def _assess_displacement_opportunity(self, positions: List[MarketPosition]) -> float:
        """Assess opportunity to displace competitors"""
        vulnerable_competitors = [p for p in positions if p.growth_trajectory == "declining"]
        return len(vulnerable_competitors) / max(len(positions), 1)
    
    def _identify_fastest_growing_competitor(self, positions: List[MarketPosition]) -> str:
        """Identify fastest growing competitor"""
        growing_competitors = [p for p in positions if p.growth_trajectory == "accelerating"]
        if growing_competitors:
            return max(growing_competitors, key=lambda x: x.visibility_score).domain
        return "None identified"
    
    def _identify_most_vulnerable_competitor(self, positions: List[MarketPosition]) -> str:
        """Identify most vulnerable competitor"""
        vulnerable = [p for p in positions if p.growth_trajectory == "declining"]
        if vulnerable:
            return min(vulnerable, key=lambda x: x.competitive_moat_strength).domain
        return "None identified"
    
    def _assess_market_consolidation(self) -> str:
        """Assess market consolidation trends"""
        # Simulated consolidation assessment
        return "Moderate consolidation with opportunities for market share gains"
    
    def _detect_emerging_competition(self) -> List[str]:
        """Detect signals of emerging competition"""
        # Simulated emerging competition detection
        return [
            "New AI-first SEO platforms entering market",
            "Increased VC investment in marketing technology",
            "Enterprise buyers evaluating new solutions"
        ]
    
    def _generate_executive_recommendations(self, insights: Dict) -> List[Dict]:
        """Generate executive-level strategic recommendations"""
        
        recommendations = []
        
        # Strategic recommendations based on insights
        if insights["strategic_position"]["relative_strength"] < 0.7:
            recommendations.append({
                "priority": "critical",
                "category": "competitive_positioning",
                "recommendation": "Accelerate market position strengthening initiatives",
                "rationale": "Below-threshold competitive position requires immediate attention",
                "success_metrics": ["Market share increase", "Visibility score improvement"],
                "timeline": "quarterly",
                "urgency": "immediate",
                "investment_required": "high"
            })
        
        if insights["growth_vectors"]["organic_opportunity_value"] > 100000:
            recommendations.append({
                "priority": "high", 
                "category": "growth_acceleration",
                "recommendation": "Execute aggressive organic growth strategy",
                "rationale": f"${insights['growth_vectors']['organic_opportunity_value']:,.0f} opportunity identified",
                "success_metrics": ["Organic traffic growth", "Revenue attribution"],
                "timeline": "annual",
                "urgency": "short_term",
                "investment_required": "medium"
            })
        
        return recommendations

    def generate_executive_dashboard(self, analysis_results: Dict) -> Dict:
        """
        Generate C-Suite Executive Dashboard
        
        Perfect for board presentations and strategic planning sessions.
        Demonstrates ability to transform technical SEO data into business intelligence.
        """
        
        return {
            "executive_kpis": {
                "market_position_score": analysis_results["executive_summary"]["competitive_advantage_score"],
                "organic_growth_rate": "+23.4%",  # Demo metric
                "competitor_threat_level": "Medium",
                "revenue_opportunity": f"${analysis_results['executive_summary']['estimated_market_opportunity']:,.0f}",
                "strategic_priority_items": analysis_results["executive_summary"]["immediate_opportunities"]
            },
            "competitive_intelligence_summary": {
                "market_leader": self._identify_market_leader(analysis_results),
                "fastest_growing_threat": analysis_results["market_intelligence"]["fastest_growing_competitor"],
                "biggest_opportunity": self._identify_biggest_opportunity(analysis_results),
                "defensive_priorities": len(analysis_results.get("competitive_threats", []))
            },
            "strategic_recommendations": {
                "immediate_actions": analysis_results["action_dashboard"]["immediate_actions"],
                "quarterly_focus": analysis_results["action_dashboard"]["quarterly_initiatives"],
                "annual_strategy": analysis_results["action_dashboard"]["annual_strategy"]
            },
            "business_impact_projections": {
                "6_month_trajectory": "15-25% organic traffic increase",
                "12_month_potential": "30-50% market share improvement", 
                "18_month_vision": "Market leadership position in key segments"
            },
            "portfolio_branding": {
                "analyst": "Sotiris Spyrou",
                "linkedin": "https://www.linkedin.com/in/sspyrou/",
                "company": "VerityAI - AI SEO Services",
                "service_url": "https://verityai.co/landing/ai-seo-services",
                "expertise_note": "Combining C-suite strategy with hands-on technical implementation"
            }
        }
    
    def _identify_market_leader(self, analysis_results: Dict) -> str:
        """Identify current market leader"""
        positions = analysis_results.get("competitive_positions", [])
        if positions:
            leader = max(positions, key=lambda x: x.visibility_score)
            return leader.domain
        return "Analysis pending"
    
    def _identify_biggest_opportunity(self, analysis_results: Dict) -> str:
        """Identify biggest market opportunity"""
        opportunities = analysis_results.get("market_opportunities", [])
        if opportunities:
            biggest = max(opportunities, key=lambda x: x.revenue_potential)
            return biggest.executive_summary
        return "Market gap analysis pending"


# Portfolio demonstration usage
async def demonstrate_executive_analysis():
    """
    Portfolio Demonstration: Executive-Level Competitive Intelligence
    
    This function showcases the strategic thinking and technical implementation 
    capabilities that make this portfolio valuable for enterprise roles.
    """
    
    # Example usage for enterprise SEO scenario
    primary_domain = "enterprise-client.com"
    competitors = ["competitor1.com", "competitor2.com", "competitor3.com"]
    api_keys = {"semrush": "demo_key", "ahrefs": "demo_key"}
    
    async with CompetitorIntelligenceAnalyzer(primary_domain, competitors, api_keys) as analyzer:
        
        # Executive-level competitive analysis
        results = await analyzer.analyze_competitive_landscape()
        
        # Generate C-suite dashboard  
        executive_dashboard = analyzer.generate_executive_dashboard(results)
        
        return {
            "competitive_analysis": results,
            "executive_dashboard": executive_dashboard,
            "portfolio_value_demonstration": {
                "strategic_thinking": "C-suite level market analysis",
                "technical_execution": "Advanced async Python implementation",
                "business_impact": "Revenue-focused competitive intelligence",
                "enterprise_ready": "Scalable architecture and executive reporting"
            }
        }


if __name__ == "__main__":
    # Portfolio demonstration
    print("🚀 Competitive Intelligence Analyzer - Portfolio Demo")
    print("👨‍💼 Showcasing C-suite strategy + technical implementation")
    print("🔗 LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("🏢 VerityAI: https://verityai.co/landing/ai-seo-services")
    print("\n⚠️  Portfolio demonstration code - not for production use")
    
    # Run demonstration
    results = asyncio.run(demonstrate_executive_analysis())
    print(f"\n✅ Analysis complete - {len(results['competitive_analysis']['market_opportunities'])} opportunities identified")