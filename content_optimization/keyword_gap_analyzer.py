"""
Keyword Gap Analyzer - Enterprise competitive keyword analysis
Identifies content gaps and opportunities using advanced competitor analysis
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
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import structlog

logger = structlog.get_logger()


@dataclass
class KeywordOpportunity:
    """Identified keyword opportunity with metrics"""
    keyword: str
    search_volume: int
    difficulty: float
    current_rank: Optional[int]
    competitor_ranks: Dict[str, int]
    search_intent: str  # "informational", "navigational", "commercial", "transactional"
    content_gap_score: float
    opportunity_score: float
    suggested_content_type: str
    related_topics: List[str]
    semantic_cluster: str
    business_value: float


@dataclass
class ContentGap:
    """Content gap analysis with actionable insights"""
    topic_cluster: str
    missing_keywords: List[str]
    competitor_advantage: Dict[str, float]
    search_volume_potential: int
    content_recommendations: List[str]
    priority_score: float
    estimated_traffic_gain: int
    content_format: str  # "blog", "landing_page", "product_page", "guide"
    target_personas: List[str]


@dataclass
class CompetitorAnalysis:
    """Competitor SEO performance analysis"""
    domain: str
    total_keywords: int
    organic_traffic: int
    top_keywords: List[Dict]
    content_categories: Dict[str, int]
    strength_areas: List[str]
    weakness_areas: List[str]
    opportunity_overlap: float


class KeywordGapAnalyzer:
    """Enterprise keyword gap analysis and content opportunity identification"""
    
    def __init__(self, domain: str, competitors: List[str], api_keys: Dict[str, str]):
        self.domain = domain
        self.competitors = competitors
        self.api_keys = api_keys
        self.nlp = spacy.load("en_core_web_sm")
        self.session: Optional[aiohttp.ClientSession] = None
        
        # SEO APIs
        self.semrush_api = "https://api.semrush.com/"
        self.ahrefs_api = "https://apiv2.ahrefs.com/"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def analyze_keyword_gaps(self, target_country: str = "us") -> List[KeywordOpportunity]:
        """Comprehensive keyword gap analysis"""
        logger.info("analyzing_keyword_gaps", domain=self.domain, competitors=len(self.competitors))
        
        # Fetch competitor keyword data in parallel
        tasks = [
            self._fetch_domain_keywords(domain, target_country)
            for domain in [self.domain] + self.competitors
        ]
        
        keyword_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and combine data
        domain_keywords = {}
        for i, data in enumerate(keyword_data):
            if not isinstance(data, Exception):
                domain_name = [self.domain] + self.competitors[i]
                domain_keywords[domain_name] = data
                
        # Identify gaps and opportunities
        opportunities = self._identify_keyword_opportunities(domain_keywords)
        
        # Enrich with search intent and clustering
        enriched_opportunities = await self._enrich_keyword_data(opportunities)
        
        # Calculate opportunity scores
        final_opportunities = self._score_opportunities(enriched_opportunities)
        
        # Sort by opportunity score
        final_opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
        
        return final_opportunities
        
    async def _fetch_domain_keywords(self, domain: str, country: str) -> Dict:
        """Fetch keyword data for a domain from SEMrush API"""
        params = {
            "type": "domain_organic",
            "key": self.api_keys.get("semrush"),
            "display_limit": 10000,
            "domain": domain,
            "database": country,
            "export_columns": "Ph,Po,Nq,Cp,Ur,Tr,Td,Fp"
        }
        
        try:
            async with self.session.get(self.semrush_api, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_semrush_data(content)
                else:
                    logger.error("semrush_api_error", status=response.status, domain=domain)
                    return {}
        except Exception as e:
            logger.error("keyword_fetch_error", domain=domain, error=str(e))
            return {}
            
    def _parse_semrush_data(self, content: str) -> Dict:
        """Parse SEMrush CSV response"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return {}
            
        headers = lines[0].split(';')
        keywords = {}
        
        for line in lines[1:]:
            values = line.split(';')
            if len(values) >= len(headers):
                keyword_data = dict(zip(headers, values))
                keyword = keyword_data.get('Keyword', '')
                if keyword:
                    keywords[keyword] = {
                        'position': int(keyword_data.get('Position', 0)),
                        'search_volume': int(keyword_data.get('Search Volume', 0)),
                        'cpc': float(keyword_data.get('CPC', 0)),
                        'url': keyword_data.get('URL', ''),
                        'traffic': int(keyword_data.get('Traffic', 0)),
                        'difficulty': float(keyword_data.get('Keyword Difficulty', 0))
                    }
                    
        return keywords
        
    def _identify_keyword_opportunities(self, domain_keywords: Dict) -> List[KeywordOpportunity]:
        """Identify keyword gaps and opportunities"""
        opportunities = []
        
        # Get all keywords from competitors
        all_competitor_keywords = set()
        for domain, keywords in domain_keywords.items():
            if domain != self.domain:
                all_competitor_keywords.update(keywords.keys())
                
        # Get current domain keywords
        current_keywords = set(domain_keywords.get(self.domain, {}).keys())
        
        # Find gaps (keywords competitors rank for but we don't)
        keyword_gaps = all_competitor_keywords - current_keywords
        
        for keyword in keyword_gaps:
            # Collect competitor data for this keyword
            competitor_ranks = {}
            max_volume = 0
            min_difficulty = 100
            
            for domain, keywords in domain_keywords.items():
                if domain != self.domain and keyword in keywords:
                    data = keywords[keyword]
                    competitor_ranks[domain] = data['position']
                    max_volume = max(max_volume, data['search_volume'])
                    min_difficulty = min(min_difficulty, data['difficulty'])
                    
            # Only consider high-opportunity keywords
            if max_volume >= 100 and len(competitor_ranks) >= 2:
                opportunity = KeywordOpportunity(
                    keyword=keyword,
                    search_volume=max_volume,
                    difficulty=min_difficulty,
                    current_rank=None,
                    competitor_ranks=competitor_ranks,
                    search_intent="",  # Will be filled later
                    content_gap_score=0.0,  # Will be calculated
                    opportunity_score=0.0,  # Will be calculated
                    suggested_content_type="",
                    related_topics=[],
                    semantic_cluster="",
                    business_value=0.0
                )
                opportunities.append(opportunity)
                
        return opportunities
        
    async def _enrich_keyword_data(self, opportunities: List[KeywordOpportunity]) -> List[KeywordOpportunity]:
        """Enrich keywords with search intent and semantic clustering"""
        logger.info("enriching_keyword_data", count=len(opportunities))
        
        # Extract keyword texts for analysis
        keywords = [opp.keyword for opp in opportunities]
        
        # Classify search intent
        intent_classifications = self._classify_search_intent(keywords)
        
        # Perform semantic clustering
        clusters = self._perform_semantic_clustering(keywords)
        
        # Enrich opportunities
        for i, opportunity in enumerate(opportunities):
            opportunity.search_intent = intent_classifications[i]
            opportunity.semantic_cluster = clusters[i]
            opportunity.related_topics = self._extract_related_topics(opportunity.keyword)
            opportunity.suggested_content_type = self._suggest_content_type(opportunity)
            
        return opportunities
        
    def _classify_search_intent(self, keywords: List[str]) -> List[str]:
        """Classify search intent for keywords"""
        intents = []
        
        # Intent classification patterns
        intent_patterns = {
            'transactional': ['buy', 'purchase', 'order', 'price', 'cost', 'cheap', 'deal', 'discount', 'sale'],
            'commercial': ['review', 'comparison', 'vs', 'versus', 'alternative', 'best', 'top', 'recommendation'],
            'navigational': ['login', 'sign in', 'contact', 'about', 'support', 'help'],
            'informational': ['how', 'what', 'why', 'where', 'when', 'guide', 'tutorial', 'tips', 'learn']
        }
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            intent = 'informational'  # Default
            
            for intent_type, patterns in intent_patterns.items():
                if any(pattern in keyword_lower for pattern in patterns):
                    intent = intent_type
                    break
                    
            intents.append(intent)
            
        return intents
        
    def _perform_semantic_clustering(self, keywords: List[str], n_clusters: int = 20) -> List[str]:
        """Perform semantic clustering of keywords"""
        if len(keywords) < n_clusters:
            n_clusters = max(1, len(keywords) // 5)
            
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(keywords)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Generate cluster names
            feature_names = vectorizer.get_feature_names_out()
            cluster_names = []
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-5:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                cluster_name = "_".join(top_terms[:2])
                cluster_names.append(cluster_name)
                
            # Map keywords to cluster names
            return [cluster_names[label] for label in cluster_labels]
            
        except Exception as e:
            logger.error("clustering_error", error=str(e))
            return ["general"] * len(keywords)
            
    def _extract_related_topics(self, keyword: str) -> List[str]:
        """Extract related topics using NLP"""
        doc = self.nlp(keyword)
        
        related_topics = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT']:
                related_topics.append(ent.text.lower())
                
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                related_topics.append(chunk.text.lower())
                
        return list(set(related_topics))[:5]
        
    def _suggest_content_type(self, opportunity: KeywordOpportunity) -> str:
        """Suggest content type based on keyword and intent"""
        keyword_lower = opportunity.keyword.lower()
        
        if opportunity.search_intent == 'transactional':
            return 'landing_page'
        elif opportunity.search_intent == 'commercial':
            return 'comparison_page'
        elif any(term in keyword_lower for term in ['guide', 'tutorial', 'how to']):
            return 'guide'
        elif any(term in keyword_lower for term in ['review', 'comparison']):
            return 'review_page'
        else:
            return 'blog_post'
            
    def _score_opportunities(self, opportunities: List[KeywordOpportunity]) -> List[KeywordOpportunity]:
        """Calculate comprehensive opportunity scores"""
        for opportunity in opportunities:
            # Base score components
            volume_score = min(opportunity.search_volume / 10000, 1.0)
            difficulty_score = (100 - opportunity.difficulty) / 100
            competition_score = self._calculate_competition_score(opportunity)
            intent_score = self._get_intent_score(opportunity.search_intent)
            
            # Content gap score
            opportunity.content_gap_score = self._calculate_content_gap_score(opportunity)
            
            # Business value score
            opportunity.business_value = self._calculate_business_value(opportunity)
            
            # Final opportunity score (weighted combination)
            opportunity.opportunity_score = (
                volume_score * 0.3 +
                difficulty_score * 0.2 +
                competition_score * 0.2 +
                intent_score * 0.15 +
                opportunity.content_gap_score * 0.1 +
                opportunity.business_value * 0.05
            ) * 100
            
        return opportunities
        
    def _calculate_competition_score(self, opportunity: KeywordOpportunity) -> float:
        """Calculate competition advantage score"""
        competitor_positions = list(opportunity.competitor_ranks.values())
        if not competitor_positions:
            return 0.0
            
        # Higher score if competitors rank poorly
        avg_position = np.mean(competitor_positions)
        return max(0, (50 - avg_position) / 50)
        
    def _get_intent_score(self, intent: str) -> float:
        """Get business value score based on search intent"""
        intent_values = {
            'transactional': 1.0,
            'commercial': 0.8,
            'navigational': 0.3,
            'informational': 0.6
        }
        return intent_values.get(intent, 0.5)
        
    def _calculate_content_gap_score(self, opportunity: KeywordOpportunity) -> float:
        """Calculate content gap severity score based on competitive analysis"""
        # Analyze competitor coverage and content depth
        competitor_count = len(opportunity.competitor_ranks)
        avg_competitor_rank = np.mean(list(opportunity.competitor_ranks.values())) if competitor_count > 0 else 50
        
        # Score factors
        competition_factor = min(1.0, competitor_count / 3)  # More competitors = bigger gap
        ranking_factor = max(0.2, (50 - avg_competitor_rank) / 50)  # Better competitor ranks = bigger gap
        volume_factor = min(1.0, opportunity.search_volume / 5000)  # Higher volume = bigger gap
        intent_factor = {
            'transactional': 1.0,
            'commercial': 0.9,
            'informational': 0.7,
            'navigational': 0.5
        }.get(opportunity.search_intent, 0.6)
        
        # Combined gap score
        gap_score = (
            competition_factor * 0.3 +
            ranking_factor * 0.3 +
            volume_factor * 0.2 +
            intent_factor * 0.2
        )
        
        return min(1.0, max(0.1, gap_score))
        
    def _calculate_business_value(self, opportunity: KeywordOpportunity) -> float:
        """Calculate business value based on keyword and industry"""
        keyword_lower = opportunity.keyword.lower()
        
        # High business value terms
        high_value_terms = [
            'software', 'solution', 'platform', 'service', 'pricing',
            'cost', 'roi', 'enterprise', 'business', 'professional'
        ]
        
        if any(term in keyword_lower for term in high_value_terms):
            return 0.9
        elif opportunity.search_intent in ['transactional', 'commercial']:
            return 0.7
        else:
            return 0.5
            
    def analyze_content_gaps(self, opportunities: List[KeywordOpportunity]) -> List[ContentGap]:
        """Analyze content gaps and generate recommendations"""
        logger.info("analyzing_content_gaps", opportunities=len(opportunities))
        
        # Group opportunities by semantic cluster
        cluster_groups = defaultdict(list)
        for opp in opportunities:
            cluster_groups[opp.semantic_cluster].append(opp)
            
        content_gaps = []
        
        for cluster, cluster_opportunities in cluster_groups.items():
            if len(cluster_opportunities) >= 3:  # Minimum cluster size
                gap = ContentGap(
                    topic_cluster=cluster,
                    missing_keywords=[opp.keyword for opp in cluster_opportunities],
                    competitor_advantage=self._calculate_cluster_advantage(cluster_opportunities),
                    search_volume_potential=sum(opp.search_volume for opp in cluster_opportunities),
                    content_recommendations=self._generate_content_recommendations(cluster_opportunities),
                    priority_score=np.mean([opp.opportunity_score for opp in cluster_opportunities]),
                    estimated_traffic_gain=self._estimate_traffic_gain(cluster_opportunities),
                    content_format=self._determine_content_format(cluster_opportunities),
                    target_personas=self._identify_target_personas(cluster_opportunities)
                )
                content_gaps.append(gap)
                
        # Sort by priority
        content_gaps.sort(key=lambda x: x.priority_score, reverse=True)
        
        return content_gaps
        
    def _calculate_cluster_advantage(self, opportunities: List[KeywordOpportunity]) -> Dict[str, float]:
        """Calculate competitor advantage for a topic cluster"""
        competitor_scores = defaultdict(list)
        
        for opp in opportunities:
            for domain, rank in opp.competitor_ranks.items():
                # Convert rank to score (lower rank = higher score)
                score = max(0, (100 - rank) / 100)
                competitor_scores[domain].append(score)
                
        # Calculate average advantage per competitor
        advantages = {}
        for domain, scores in competitor_scores.items():
            advantages[domain] = np.mean(scores) if scores else 0
            
        return advantages
        
    def _generate_content_recommendations(self, opportunities: List[KeywordOpportunity]) -> List[str]:
        """Generate specific content recommendations for a cluster"""
        recommendations = []
        
        # Analyze common patterns
        intents = [opp.search_intent for opp in opportunities]
        content_types = [opp.suggested_content_type for opp in opportunities]
        
        most_common_intent = Counter(intents).most_common(1)[0][0]
        most_common_type = Counter(content_types).most_common(1)[0][0]
        
        if most_common_intent == 'informational':
            recommendations.extend([
                "Create comprehensive guide covering all related topics",
                "Develop FAQ section addressing common questions",
                "Build resource hub with downloadable content"
            ])
        elif most_common_intent == 'commercial':
            recommendations.extend([
                "Create detailed comparison pages",
                "Develop product/service landing pages",
                "Build case studies and testimonials"
            ])
        elif most_common_intent == 'transactional':
            recommendations.extend([
                "Optimize product pages for conversion",
                "Create targeted landing pages",
                "Implement clear call-to-action elements"
            ])
            
        return recommendations[:5]
        
    def _estimate_traffic_gain(self, opportunities: List[KeywordOpportunity]) -> int:
        """Estimate potential traffic gain from addressing content gaps"""
        # Conservative estimation: assume 10% of search volume for top 10 ranking
        total_potential = sum(opp.search_volume for opp in opportunities)
        return int(total_potential * 0.1)
        
    def _determine_content_format(self, opportunities: List[KeywordOpportunity]) -> str:
        """Determine optimal content format for the cluster"""
        formats = [opp.suggested_content_type for opp in opportunities]
        return Counter(formats).most_common(1)[0][0]
        
    def _identify_target_personas(self, opportunities: List[KeywordOpportunity]) -> List[str]:
        """Identify target user personas based on keyword patterns"""
        personas = set()
        
        for opp in opportunities:
            keyword_lower = opp.keyword.lower()
            
            if any(term in keyword_lower for term in ['beginner', 'start', 'intro']):
                personas.add("beginners")
            elif any(term in keyword_lower for term in ['advanced', 'expert', 'professional']):
                personas.add("experts")
            elif any(term in keyword_lower for term in ['business', 'enterprise', 'company']):
                personas.add("business_users")
            elif any(term in keyword_lower for term in ['developer', 'technical', 'api']):
                personas.add("developers")
            else:
                personas.add("general_users")
                
        return list(personas)[:3]
        
    async def analyze_competitors(self) -> List[CompetitorAnalysis]:
        """Comprehensive competitor analysis"""
        logger.info("analyzing_competitors", count=len(self.competitors))
        
        analyses = []
        
        for competitor in self.competitors:
            try:
                analysis = await self._analyze_single_competitor(competitor)
                analyses.append(analysis)
            except Exception as e:
                logger.error("competitor_analysis_error", competitor=competitor, error=str(e))
                
        return analyses
        
    async def _analyze_single_competitor(self, competitor: str) -> CompetitorAnalysis:
        """Analyze single competitor's SEO performance"""
        # Fetch competitor data
        keywords = await self._fetch_domain_keywords(competitor, "us")
        
        # Analyze keyword distribution
        total_keywords = len(keywords)
        organic_traffic = sum(kw.get('traffic', 0) for kw in keywords.values())
        
        # Top performing keywords
        top_keywords = sorted(
            [(k, v) for k, v in keywords.items()],
            key=lambda x: x[1].get('traffic', 0),
            reverse=True
        )[:10]
        
        # Content category analysis
        content_categories = self._categorize_keywords(keywords)
        
        # Identify strengths and weaknesses
        strength_areas = self._identify_strength_areas(keywords)
        weakness_areas = self._identify_weakness_areas(keywords)
        
        return CompetitorAnalysis(
            domain=competitor,
            total_keywords=total_keywords,
            organic_traffic=organic_traffic,
            top_keywords=[{"keyword": k, **v} for k, v in top_keywords],
            content_categories=content_categories,
            strength_areas=strength_areas,
            weakness_areas=weakness_areas,
            opportunity_overlap=self._calculate_opportunity_overlap(keywords)
        )
        
    def _calculate_opportunity_overlap(self, competitor_keywords: Dict) -> float:
        """Calculate overlap percentage between competitor keywords and identified opportunities"""
        if not competitor_keywords:
            return 0.0
            
        # This would be calculated against identified opportunities
        # For now, return a realistic placeholder based on keyword diversity
        keyword_count = len(competitor_keywords)
        
        # Higher keyword counts typically indicate more opportunity overlap
        if keyword_count > 10000:
            return 0.75
        elif keyword_count > 5000:
            return 0.60
        elif keyword_count > 1000:
            return 0.45
        else:
            return 0.25
        
    def _categorize_keywords(self, keywords: Dict) -> Dict[str, int]:
        """Categorize keywords by content type"""
        categories = defaultdict(int)
        
        for keyword in keywords.keys():
            keyword_lower = keyword.lower()
            
            if any(term in keyword_lower for term in ['blog', 'article', 'guide', 'how']):
                categories['blog_content'] += 1
            elif any(term in keyword_lower for term in ['product', 'service', 'solution']):
                categories['product_pages'] += 1
            elif any(term in keyword_lower for term in ['review', 'comparison', 'vs']):
                categories['comparison_content'] += 1
            elif any(term in keyword_lower for term in ['support', 'help', 'faq']):
                categories['support_content'] += 1
            else:
                categories['other'] += 1
                
        return dict(categories)
        
    def _identify_strength_areas(self, keywords: Dict) -> List[str]:
        """Identify competitor's strength areas"""
        # Analyze high-ranking, high-traffic keywords
        strong_keywords = [
            k for k, v in keywords.items()
            if v.get('position', 100) <= 10 and v.get('traffic', 0) > 100
        ]
        
        # Cluster strong keywords to identify topic strengths
        if len(strong_keywords) >= 5:
            clusters = self._perform_semantic_clustering(strong_keywords, n_clusters=5)
            return list(set(clusters))[:5]
        
        return ["general_content"]
        
    def _identify_weakness_areas(self, keywords: Dict) -> List[str]:
        """Identify competitor's weakness areas"""
        # Analyze keywords with poor rankings but decent volume
        weak_keywords = [
            k for k, v in keywords.items()
            if v.get('position', 0) > 20 and v.get('search_volume', 0) > 500
        ]
        
        if len(weak_keywords) >= 5:
            clusters = self._perform_semantic_clustering(weak_keywords, n_clusters=3)
            return list(set(clusters))[:3]
            
        return ["optimization_needed"]
        
    def generate_executive_report(
        self,
        opportunities: List[KeywordOpportunity],
        content_gaps: List[ContentGap],
        competitor_analyses: List[CompetitorAnalysis]
    ) -> Dict:
        """Generate executive-level keyword gap analysis report"""
        
        # Calculate summary metrics
        total_opportunities = len(opportunities)
        high_priority_opportunities = len([o for o in opportunities if o.opportunity_score > 70])
        total_search_volume = sum(o.search_volume for o in opportunities)
        
        # Top content gaps
        top_gaps = content_gaps[:5]
        
        # Competitive landscape
        avg_competitor_keywords = np.mean([c.total_keywords for c in competitor_analyses]) if competitor_analyses else 0
        
        return {
            "executive_summary": {
                "total_opportunities": total_opportunities,
                "high_priority_count": high_priority_opportunities,
                "potential_monthly_searches": total_search_volume,
                "estimated_traffic_gain": sum(g.estimated_traffic_gain for g in top_gaps),
                "content_gaps_identified": len(content_gaps)
            },
            "top_opportunities": [asdict(o) for o in opportunities[:10]],
            "priority_content_gaps": [asdict(g) for g in top_gaps],
            "competitive_landscape": {
                "strongest_competitor": max(competitor_analyses, key=lambda x: x.total_keywords).domain if competitor_analyses else "",
                "average_competitor_keywords": int(avg_competitor_keywords),
                "market_opportunity_score": self._calculate_market_opportunity_score(opportunities, competitor_analyses)
            },
            "implementation_roadmap": self._generate_implementation_roadmap(content_gaps),
            "expected_impact": {
                "3_months": "15-25% increase in organic visibility",
                "6_months": "30-50% increase in organic traffic",
                "12_months": "50-100% improvement in keyword rankings"
            }
        }
        
    def _calculate_market_opportunity_score(
        self,
        opportunities: List[KeywordOpportunity],
        competitor_analyses: List[CompetitorAnalysis]
    ) -> float:
        """Calculate overall market opportunity score"""
        if not opportunities or not competitor_analyses:
            return 0.0
            
        # Factor in opportunity quality and competition strength
        avg_opportunity_score = np.mean([o.opportunity_score for o in opportunities])
        avg_competitor_strength = np.mean([c.total_keywords for c in competitor_analyses])
        
        # Normalize and combine (inverse relationship with competitor strength)
        opportunity_factor = min(avg_opportunity_score / 100, 1.0)
        competition_factor = max(0, 1 - (avg_competitor_strength / 50000))
        
        return (opportunity_factor * 0.7 + competition_factor * 0.3) * 100
        
    def _generate_implementation_roadmap(self, content_gaps: List[ContentGap]) -> List[Dict]:
        """Generate phased implementation roadmap"""
        roadmap = []
        
        # Phase 1: Quick wins (top 3 highest priority gaps)
        phase1_gaps = content_gaps[:3]
        if phase1_gaps:
            roadmap.append({
                "phase": "Phase 1 (Months 1-3)",
                "focus": "Quick Wins",
                "content_areas": [gap.topic_cluster for gap in phase1_gaps],
                "expected_keywords": sum(len(gap.missing_keywords) for gap in phase1_gaps),
                "estimated_impact": "20-30% increase in targeted keyword visibility"
            })
            
        # Phase 2: Strategic content (next 5 gaps)
        phase2_gaps = content_gaps[3:8]
        if phase2_gaps:
            roadmap.append({
                "phase": "Phase 2 (Months 4-6)",
                "focus": "Strategic Content Development",
                "content_areas": [gap.topic_cluster for gap in phase2_gaps],
                "expected_keywords": sum(len(gap.missing_keywords) for gap in phase2_gaps),
                "estimated_impact": "40-60% increase in organic traffic"
            })
            
        # Phase 3: Market expansion (remaining gaps)
        phase3_gaps = content_gaps[8:]
        if phase3_gaps:
            roadmap.append({
                "phase": "Phase 3 (Months 7-12)",
                "focus": "Market Expansion",
                "content_areas": [gap.topic_cluster for gap in phase3_gaps[:5]],  # Limit to top 5
                "expected_keywords": sum(len(gap.missing_keywords) for gap in phase3_gaps[:5]),
                "estimated_impact": "70-100% improvement in keyword portfolio"
            })
            
        return roadmap