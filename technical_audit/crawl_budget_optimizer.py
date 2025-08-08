"""
Crawl Budget Optimizer - Enterprise crawl efficiency optimization
Analyzes log files, sitemaps, robots.txt for maximum crawl budget utilization
"""

import re
import gzip
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from urllib.parse import urlparse, urljoin
import pandas as pd
import numpy as np
from lxml import etree
import requests
import structlog

logger = structlog.get_logger()


@dataclass
class CrawlStats:
    """Crawl statistics from log analysis"""
    total_crawls: int
    unique_urls: int
    googlebot_crawls: int
    bingbot_crawls: int
    other_bot_crawls: int
    avg_response_time: float
    error_rate: float
    redirect_chains: int
    orphan_pages: int
    duplicate_crawls: int
    wasted_budget_percentage: float


@dataclass
class URLPriority:
    """URL priority scoring for crawl optimization"""
    url: str
    priority_score: float
    crawl_frequency: str  # "daily", "weekly", "monthly"
    changefreq: str
    lastmod: Optional[datetime]
    traffic_value: float
    conversion_value: float
    content_freshness: float
    internal_links: int
    external_links: int


@dataclass 
class CrawlBudgetRecommendation:
    """Actionable crawl budget optimization recommendation"""
    issue: str
    impact: str  # "critical", "high", "medium", "low"
    affected_urls: int
    wasted_budget: float
    solution: str
    implementation_steps: List[str]
    expected_improvement: str


class CrawlBudgetOptimizer:
    """Enterprise-grade crawl budget optimization"""
    
    def __init__(self, domain: str, log_file_path: Optional[str] = None):
        self.domain = domain
        self.log_file_path = log_file_path
        self.base_url = f"https://{domain}"
        self.sitemap_urls = []
        self.robots_rules = {}
        self.crawl_data = defaultdict(list)
        
    def analyze_log_files(self, log_path: str, date_range: Optional[Tuple[datetime, datetime]] = None) -> CrawlStats:
        """Analyze server log files for crawl patterns"""
        logger.info("analyzing_log_files", path=log_path)
        
        crawl_records = []
        bot_patterns = {
            'googlebot': r'Googlebot',
            'bingbot': r'bingbot', 
            'yandexbot': r'YandexBot',
            'baidubot': r'Baiduspider',
            'facebookbot': r'facebookexternalhit',
            'twitterbot': r'Twitterbot',
            'linkedinbot': r'LinkedInBot',
            'slackbot': r'Slackbot'
        }
        
        # Parse log file (supports Apache/Nginx common log format)
        with self._open_log_file(log_path) as f:
            for line in f:
                record = self._parse_log_line(line)
                if record and self._is_bot_crawl(record, bot_patterns):
                    if not date_range or self._in_date_range(record['timestamp'], date_range):
                        crawl_records.append(record)
                        
        # Analyze crawl patterns
        df = pd.DataFrame(crawl_records)
        
        if df.empty:
            return CrawlStats(
                total_crawls=0,
                unique_urls=0,
                googlebot_crawls=0,
                bingbot_crawls=0,
                other_bot_crawls=0,
                avg_response_time=0,
                error_rate=0,
                redirect_chains=0,
                orphan_pages=0,
                duplicate_crawls=0,
                wasted_budget_percentage=0
            )
            
        # Calculate statistics
        total_crawls = len(df)
        unique_urls = df['url'].nunique()
        googlebot_crawls = len(df[df['bot'] == 'googlebot'])
        bingbot_crawls = len(df[df['bot'] == 'bingbot'])
        other_bot_crawls = total_crawls - googlebot_crawls - bingbot_crawls
        
        # Response times and errors
        avg_response_time = df['response_time'].mean() if 'response_time' in df else 0
        error_rate = len(df[df['status_code'] >= 400]) / total_crawls if total_crawls > 0 else 0
        
        # Identify inefficiencies
        redirect_chains = self._identify_redirect_chains(df)
        orphan_pages = self._identify_orphan_pages(df)
        duplicate_crawls = self._calculate_duplicate_crawls(df)
        
        # Calculate wasted budget
        wasted_crawls = len(df[df['status_code'] >= 400]) + duplicate_crawls + redirect_chains
        wasted_budget_percentage = (wasted_crawls / total_crawls * 100) if total_crawls > 0 else 0
        
        return CrawlStats(
            total_crawls=total_crawls,
            unique_urls=unique_urls,
            googlebot_crawls=googlebot_crawls,
            bingbot_crawls=bingbot_crawls,
            other_bot_crawls=other_bot_crawls,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            redirect_chains=redirect_chains,
            orphan_pages=orphan_pages,
            duplicate_crawls=duplicate_crawls,
            wasted_budget_percentage=wasted_budget_percentage
        )
        
    def _open_log_file(self, path: str):
        """Open log file, handling gzip compression"""
        if path.endswith('.gz'):
            return gzip.open(path, 'rt')
        return open(path, 'r')
        
    def _parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse Apache/Nginx common log format"""
        # Common log format regex
        pattern = r'(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+) "([^"]*)" "([^"]*)"'
        match = re.match(pattern, line)
        
        if match:
            return {
                'ip': match.group(1),
                'timestamp': datetime.strptime(match.group(2), '%d/%b/%Y:%H:%M:%S %z'),
                'method': match.group(3),
                'url': match.group(4),
                'protocol': match.group(5),
                'status_code': int(match.group(6)),
                'bytes': int(match.group(7)),
                'referer': match.group(8),
                'user_agent': match.group(9),
                'bot': self._identify_bot(match.group(9))
            }
        return None
        
    def _identify_bot(self, user_agent: str) -> str:
        """Identify bot from user agent string"""
        user_agent_lower = user_agent.lower()
        if 'googlebot' in user_agent_lower:
            return 'googlebot'
        elif 'bingbot' in user_agent_lower:
            return 'bingbot'
        elif 'yandex' in user_agent_lower:
            return 'yandexbot'
        elif 'baidu' in user_agent_lower:
            return 'baidubot'
        return 'other'
        
    def _is_bot_crawl(self, record: Dict, patterns: Dict) -> bool:
        """Check if request is from a bot"""
        user_agent = record.get('user_agent', '')
        for bot, pattern in patterns.items():
            if re.search(pattern, user_agent, re.IGNORECASE):
                record['bot'] = bot
                return True
        return False
        
    def _in_date_range(self, timestamp: datetime, date_range: Tuple[datetime, datetime]) -> bool:
        """Check if timestamp is within date range"""
        return date_range[0] <= timestamp <= date_range[1]
        
    def _identify_redirect_chains(self, df: pd.DataFrame) -> int:
        """Identify redirect chains in crawl data"""
        redirect_count = 0
        redirect_urls = df[df['status_code'].between(300, 399)]['url'].unique()
        
        for url in redirect_urls:
            # Check if redirected URL is also crawled
            if url in df['referer'].values:
                redirect_count += 1
                
        return redirect_count
        
    def _identify_orphan_pages(self, df: pd.DataFrame) -> int:
        """Identify orphan pages with no internal links"""
        all_urls = set(df['url'].unique())
        linked_urls = set(df['referer'].unique())
        orphans = all_urls - linked_urls
        return len(orphans)
        
    def _calculate_duplicate_crawls(self, df: pd.DataFrame) -> int:
        """Calculate duplicate crawls of the same URL"""
        url_counts = df['url'].value_counts()
        duplicates = url_counts[url_counts > 1].sum() - len(url_counts[url_counts > 1])
        return int(duplicates)
        
    def analyze_xml_sitemap(self, sitemap_url: Optional[str] = None) -> Dict:
        """Analyze XML sitemap for optimization opportunities"""
        if not sitemap_url:
            sitemap_url = f"{self.base_url}/sitemap.xml"
            
        logger.info("analyzing_sitemap", url=sitemap_url)
        
        try:
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            # Parse sitemap
            root = etree.fromstring(response.content)
            
            # Handle sitemap index
            if root.tag.endswith('sitemapindex'):
                return self._analyze_sitemap_index(root)
            else:
                return self._analyze_urlset(root)
                
        except Exception as e:
            logger.error("sitemap_analysis_error", error=str(e))
            return {}
            
    def _analyze_sitemap_index(self, root) -> Dict:
        """Analyze sitemap index file"""
        namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sitemaps = []
        
        for sitemap in root.xpath('//sm:sitemap', namespaces=namespaces):
            loc = sitemap.find('sm:loc', namespaces=namespaces)
            lastmod = sitemap.find('sm:lastmod', namespaces=namespaces)
            
            if loc is not None:
                sitemap_data = self.analyze_xml_sitemap(loc.text)
                sitemaps.append(sitemap_data)
                
        return {
            'type': 'index',
            'sitemap_count': len(sitemaps),
            'total_urls': sum(s.get('url_count', 0) for s in sitemaps),
            'sitemaps': sitemaps
        }
        
    def _analyze_urlset(self, root) -> Dict:
        """Analyze URL set sitemap"""
        namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []
        
        for url in root.xpath('//sm:url', namespaces=namespaces):
            loc = url.find('sm:loc', namespaces=namespaces)
            lastmod = url.find('sm:lastmod', namespaces=namespaces)
            changefreq = url.find('sm:changefreq', namespaces=namespaces)
            priority = url.find('sm:priority', namespaces=namespaces)
            
            if loc is not None:
                urls.append({
                    'url': loc.text,
                    'lastmod': lastmod.text if lastmod is not None else None,
                    'changefreq': changefreq.text if changefreq is not None else None,
                    'priority': float(priority.text) if priority is not None else 0.5
                })
                
        # Analyze sitemap quality
        issues = self._identify_sitemap_issues(urls)
        
        return {
            'type': 'urlset',
            'url_count': len(urls),
            'urls': urls,
            'issues': issues,
            'recommendations': self._generate_sitemap_recommendations(issues)
        }
        
    def _identify_sitemap_issues(self, urls: List[Dict]) -> List[str]:
        """Identify common sitemap issues"""
        issues = []
        
        # Check for missing lastmod
        missing_lastmod = sum(1 for u in urls if not u['lastmod'])
        if missing_lastmod > len(urls) * 0.2:
            issues.append(f"{missing_lastmod} URLs missing lastmod date")
            
        # Check for incorrect priorities
        default_priority = sum(1 for u in urls if u['priority'] == 0.5)
        if default_priority == len(urls):
            issues.append("All URLs have default priority (0.5)")
            
        # Check for too many URLs
        if len(urls) > 50000:
            issues.append(f"Sitemap contains {len(urls)} URLs (max recommended: 50,000)")
            
        # Check for non-canonical URLs
        non_https = sum(1 for u in urls if not u['url'].startswith('https://'))
        if non_https > 0:
            issues.append(f"{non_https} non-HTTPS URLs in sitemap")
            
        return issues
        
    def _generate_sitemap_recommendations(self, issues: List[str]) -> List[str]:
        """Generate sitemap optimization recommendations"""
        recommendations = []
        
        if any('lastmod' in issue for issue in issues):
            recommendations.append("Add lastmod dates to track content freshness")
            
        if any('priority' in issue for issue in issues):
            recommendations.append("Set strategic priority values based on business importance")
            
        if any('50,000' in issue for issue in issues):
            recommendations.append("Split sitemap into multiple files using sitemap index")
            
        if any('HTTPS' in issue for issue in issues):
            recommendations.append("Ensure all URLs use HTTPS protocol")
            
        return recommendations
        
    def analyze_robots_txt(self) -> Dict:
        """Analyze robots.txt for crawl optimization"""
        robots_url = f"{self.base_url}/robots.txt"
        logger.info("analyzing_robots_txt", url=robots_url)
        
        try:
            response = requests.get(robots_url, timeout=10)
            response.raise_for_status()
            
            content = response.text
            analysis = self._parse_robots_txt(content)
            issues = self._identify_robots_issues(analysis)
            
            return {
                'rules': analysis,
                'issues': issues,
                'recommendations': self._generate_robots_recommendations(issues),
                'crawl_delay': analysis.get('crawl_delay'),
                'sitemap_references': analysis.get('sitemaps', [])
            }
            
        except Exception as e:
            logger.error("robots_txt_error", error=str(e))
            return {}
            
    def _parse_robots_txt(self, content: str) -> Dict:
        """Parse robots.txt content"""
        rules = defaultdict(dict)
        current_agent = '*'
        sitemaps = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith('#') or not line:
                continue
                
            if ':' in line:
                directive, value = line.split(':', 1)
                directive = directive.strip().lower()
                value = value.strip()
                
                if directive == 'user-agent':
                    current_agent = value.lower()
                    if current_agent not in rules:
                        rules[current_agent] = {
                            'disallow': [],
                            'allow': [],
                            'crawl_delay': None
                        }
                elif directive == 'disallow':
                    rules[current_agent]['disallow'].append(value)
                elif directive == 'allow':
                    rules[current_agent]['allow'].append(value)
                elif directive == 'crawl-delay':
                    rules[current_agent]['crawl_delay'] = int(value)
                elif directive == 'sitemap':
                    sitemaps.append(value)
                    
        return {
            'rules': dict(rules),
            'sitemaps': sitemaps
        }
        
    def _identify_robots_issues(self, analysis: Dict) -> List[str]:
        """Identify robots.txt optimization issues"""
        issues = []
        rules = analysis.get('rules', {})
        
        # Check for overly restrictive rules
        if '*' in rules:
            disallow_all = any(d == '/' for d in rules['*'].get('disallow', []))
            if disallow_all:
                issues.append("Blocking all crawlers with Disallow: /")
                
        # Check for missing sitemap reference
        if not analysis.get('sitemaps'):
            issues.append("No sitemap reference in robots.txt")
            
        # Check for redundant rules
        for agent, agent_rules in rules.items():
            disallow = agent_rules.get('disallow', [])
            if len(disallow) > 50:
                issues.append(f"Excessive disallow rules for {agent} ({len(disallow)} rules)")
                
        # Check for wildcard usage
        for agent, agent_rules in rules.items():
            for rule in agent_rules.get('disallow', []):
                if '*' in rule or '$' in rule:
                    issues.append(f"Complex pattern in disallow rule: {rule}")
                    
        return issues
        
    def _generate_robots_recommendations(self, issues: List[str]) -> List[str]:
        """Generate robots.txt optimization recommendations"""
        recommendations = []
        
        if any('Blocking all' in issue for issue in issues):
            recommendations.append("Review and refine crawler access rules")
            
        if any('sitemap' in issue.lower() for issue in issues):
            recommendations.append("Add Sitemap directive to robots.txt")
            
        if any('Excessive' in issue for issue in issues):
            recommendations.append("Consolidate disallow rules using patterns")
            
        if any('Complex pattern' in issue for issue in issues):
            recommendations.append("Simplify wildcard patterns for better crawler understanding")
            
        return recommendations
        
    def optimize_internal_linking(self, crawl_data: pd.DataFrame) -> List[URLPriority]:
        """Optimize internal linking structure for crawl efficiency"""
        logger.info("optimizing_internal_linking")
        
        # Calculate PageRank-style importance scores
        link_graph = self._build_link_graph(crawl_data)
        importance_scores = self._calculate_page_importance(link_graph)
        
        # Combine with business metrics
        url_priorities = []
        
        for url, importance in importance_scores.items():
            priority = URLPriority(
                url=url,
                priority_score=importance,
                crawl_frequency=self._determine_crawl_frequency(importance),
                changefreq=self._determine_changefreq(url, crawl_data),
                lastmod=self._get_last_modified(url, crawl_data),
                traffic_value=self._estimate_traffic_value(url),
                conversion_value=self._estimate_conversion_value(url),
                content_freshness=self._calculate_freshness(url, crawl_data),
                internal_links=link_graph.get(url, {}).get('inbound', 0),
                external_links=link_graph.get(url, {}).get('outbound', 0)
            )
            url_priorities.append(priority)
            
        # Sort by priority
        url_priorities.sort(key=lambda x: x.priority_score, reverse=True)
        
        return url_priorities
        
    def _build_link_graph(self, crawl_data: pd.DataFrame) -> Dict:
        """Build internal link graph from crawl data"""
        link_graph = defaultdict(lambda: {'inbound': 0, 'outbound': 0, 'links': []})
        
        for _, row in crawl_data.iterrows():
            source = row.get('referer', '')
            target = row.get('url', '')
            
            if source and target and self.domain in source:
                link_graph[target]['inbound'] += 1
                link_graph[source]['outbound'] += 1
                link_graph[source]['links'].append(target)
                
        return dict(link_graph)
        
    def _calculate_page_importance(self, link_graph: Dict) -> Dict[str, float]:
        """Calculate page importance using simplified PageRank"""
        pages = list(link_graph.keys())
        n = len(pages)
        
        if n == 0:
            return {}
            
        # Initialize scores
        scores = {page: 1.0 / n for page in pages}
        damping = 0.85
        
        # Iterate to convergence
        for _ in range(10):
            new_scores = {}
            for page in pages:
                rank = (1 - damping) / n
                for other_page, data in link_graph.items():
                    if page in data.get('links', []):
                        rank += damping * scores[other_page] / max(data['outbound'], 1)
                new_scores[page] = rank
            scores = new_scores
            
        return scores
        
    def _determine_crawl_frequency(self, importance: float) -> str:
        """Determine optimal crawl frequency based on importance"""
        if importance > 0.8:
            return "daily"
        elif importance > 0.5:
            return "weekly"
        else:
            return "monthly"
            
    def _determine_changefreq(self, url: str, crawl_data: pd.DataFrame) -> str:
        """Determine change frequency for URL"""
        # Check URL patterns
        if '/blog/' in url or '/news/' in url:
            return "daily"
        elif '/products/' in url or '/services/' in url:
            return "weekly"
        elif '/about/' in url or '/contact/' in url:
            return "yearly"
        else:
            return "monthly"
            
    def _get_last_modified(self, url: str, crawl_data: pd.DataFrame) -> Optional[datetime]:
        """Get last modified date for URL"""
        url_data = crawl_data[crawl_data['url'] == url]
        if not url_data.empty:
            return url_data['timestamp'].max()
        return None
        
    def _estimate_traffic_value(self, url: str) -> float:
        """Estimate traffic value of URL based on path patterns and analytics integration"""
        # URL pattern-based scoring with analytics integration points
        url_lower = url.lower()
        
        # High-value commercial pages
        if any(pattern in url_lower for pattern in ['/products/', '/services/', '/solutions/']):
            base_value = 0.9
        # Content marketing pages
        elif any(pattern in url_lower for pattern in ['/blog/', '/resources/', '/guides/']):
            base_value = 0.7
        # Support and information pages  
        elif any(pattern in url_lower for pattern in ['/support/', '/help/', '/faq/']):
            base_value = 0.4
        # Category and listing pages
        elif any(pattern in url_lower for pattern in ['/category/', '/archive/', '/tag/']):
            base_value = 0.6
        # Corporate pages
        elif any(pattern in url_lower for pattern in ['/about/', '/contact/', '/careers/']):
            base_value = 0.3
        else:
            base_value = 0.5
            
        # Adjust based on URL depth (shallower URLs typically have higher traffic)
        url_depth = len([seg for seg in url.split('/') if seg])
        depth_multiplier = max(0.5, 1.0 - (url_depth - 2) * 0.1)
        
        # Adjust for URL length (shorter URLs typically perform better)
        length_multiplier = 1.0 if len(url) < 100 else 0.8
        
        final_value = base_value * depth_multiplier * length_multiplier
        return min(1.0, max(0.1, final_value))
            
    def _estimate_conversion_value(self, url: str) -> float:
        """Estimate conversion value of URL based on funnel position and conversion tracking"""
        url_lower = url.lower()
        
        # High-conversion transactional pages
        if any(pattern in url_lower for pattern in ['/checkout', '/cart', '/buy', '/purchase']):
            return 1.0
        # Product and service detail pages
        elif any(pattern in url_lower for pattern in ['/products/', '/services/', '/pricing']):
            return 0.8
        # Landing pages and offers
        elif any(pattern in url_lower for pattern in ['/landing', '/offer', '/demo', '/trial']):
            return 0.9
        # Lead generation pages
        elif any(pattern in url_lower for pattern in ['/contact', '/quote', '/consultation']):
            return 0.7
        # Content that drives conversions
        elif any(pattern in url_lower for pattern in ['/case-studies', '/testimonials', '/reviews']):
            return 0.6
        # Educational content (lower direct conversion but builds trust)
        elif any(pattern in url_lower for pattern in ['/blog/', '/guides/', '/resources/']):
            return 0.4
        # Support content (retention value)
        elif any(pattern in url_lower for pattern in ['/support/', '/help/', '/faq/']):
            return 0.5
        # About and corporate pages
        elif any(pattern in url_lower for pattern in ['/about/', '/team/', '/careers/']):
            return 0.2
        else:
            return 0.3
            
        # Note: In a production system, this would integrate with:
        # - Google Analytics goal completion data
        # - E-commerce tracking data
        # - CRM conversion attribution
        # - A/B testing platform results
            
    def _calculate_freshness(self, url: str, crawl_data: pd.DataFrame) -> float:
        """Calculate content freshness score"""
        last_modified = self._get_last_modified(url, crawl_data)
        if last_modified:
            days_old = (datetime.now() - last_modified).days
            if days_old < 7:
                return 1.0
            elif days_old < 30:
                return 0.7
            elif days_old < 90:
                return 0.4
            else:
                return 0.1
        return 0.5
        
    def generate_recommendations(self, stats: CrawlStats, sitemap_analysis: Dict, robots_analysis: Dict) -> List[CrawlBudgetRecommendation]:
        """Generate comprehensive crawl budget optimization recommendations"""
        recommendations = []
        
        # High error rate
        if stats.error_rate > 0.05:
            recommendations.append(CrawlBudgetRecommendation(
                issue="High error rate in crawls",
                impact="critical",
                affected_urls=int(stats.total_crawls * stats.error_rate),
                wasted_budget=stats.error_rate * 100,
                solution="Fix 4xx and 5xx errors",
                implementation_steps=[
                    "Identify and fix broken internal links",
                    "Implement proper 301 redirects for moved content",
                    "Set up custom 404 page with navigation",
                    "Monitor server errors and fix backend issues"
                ],
                expected_improvement=f"Recover {stats.error_rate * 100:.1f}% of crawl budget"
            ))
            
        # Duplicate crawls
        if stats.duplicate_crawls > stats.total_crawls * 0.1:
            recommendations.append(CrawlBudgetRecommendation(
                issue="Excessive duplicate crawling",
                impact="high",
                affected_urls=stats.duplicate_crawls,
                wasted_budget=(stats.duplicate_crawls / stats.total_crawls) * 100,
                solution="Implement URL canonicalization",
                implementation_steps=[
                    "Add canonical tags to all pages",
                    "Consolidate duplicate content",
                    "Fix URL parameter handling",
                    "Implement consistent URL structure"
                ],
                expected_improvement=f"Save {(stats.duplicate_crawls / stats.total_crawls) * 100:.1f}% crawl budget"
            ))
            
        # Redirect chains
        if stats.redirect_chains > 0:
            recommendations.append(CrawlBudgetRecommendation(
                issue="Redirect chains wasting crawl budget",
                impact="medium",
                affected_urls=stats.redirect_chains,
                wasted_budget=(stats.redirect_chains / stats.total_crawls) * 100,
                solution="Eliminate redirect chains",
                implementation_steps=[
                    "Map all redirect chains",
                    "Update links to final destination",
                    "Implement direct 301 redirects",
                    "Update internal links in content"
                ],
                expected_improvement="Reduce crawl depth and improve indexing speed"
            ))
            
        # Orphan pages
        if stats.orphan_pages > 0:
            recommendations.append(CrawlBudgetRecommendation(
                issue="Orphan pages with no internal links",
                impact="medium",
                affected_urls=stats.orphan_pages,
                wasted_budget=0,
                solution="Improve internal linking",
                implementation_steps=[
                    "Identify valuable orphan pages",
                    "Add contextual internal links",
                    "Update navigation structure",
                    "Create HTML sitemaps for deep pages"
                ],
                expected_improvement="Improve crawl discovery and page authority"
            ))
            
        return recommendations
        
    def generate_executive_report(self, stats: CrawlStats, recommendations: List[CrawlBudgetRecommendation]) -> Dict:
        """Generate executive-level crawl budget report"""
        # Calculate crawl efficiency score (0-100)
        efficiency_factors = [
            (100 - stats.error_rate * 100) * 0.3,  # Error rate impact
            (100 - stats.wasted_budget_percentage) * 0.4,  # Budget waste impact
            min(100, (stats.unique_urls / max(stats.total_crawls, 1)) * 100) * 0.3  # Crawl diversity
        ]
        crawl_efficiency_score = sum(efficiency_factors)
        
        # Estimate monthly crawl budget (based on current patterns)
        daily_crawls = stats.total_crawls / 30 if stats.total_crawls > 0 else 0
        monthly_crawl_budget = daily_crawls * 30
        
        # Extract top opportunities from recommendations
        top_opportunities = sorted(
            recommendations,
            key=lambda x: x.wasted_budget,
            reverse=True
        )[:5]
        
        # Generate implementation roadmap
        implementation_roadmap = []
        critical_recs = [r for r in recommendations if r.impact == "critical"]
        high_recs = [r for r in recommendations if r.impact == "high"]
        
        if critical_recs:
            implementation_roadmap.append({
                "phase": "Immediate (Week 1-2)",
                "focus": "Critical Issues",
                "recommendations": [r.solution for r in critical_recs[:3]],
                "expected_impact": f"Recover {sum(r.wasted_budget for r in critical_recs[:3]):.1f}% crawl budget"
            })
            
        if high_recs:
            implementation_roadmap.append({
                "phase": "Short-term (Month 1-2)", 
                "focus": "High Impact Optimizations",
                "recommendations": [r.solution for r in high_recs[:3]],
                "expected_impact": f"Improve crawl efficiency by {len(high_recs) * 5}%"
            })
            
        implementation_roadmap.append({
            "phase": "Long-term (Month 3-6)",
            "focus": "Strategic Improvements",
            "recommendations": [
                "Implement dynamic XML sitemap generation",
                "Deploy advanced internal linking automation",
                "Establish crawl budget monitoring dashboards"
            ],
            "expected_impact": "Achieve enterprise-scale crawl optimization"
        })
        
        # Calculate expected improvements
        potential_efficiency_gain = min(25, stats.wasted_budget_percentage * 0.7)
        potential_indexation_gain = min(30, len(recommendations) * 3)
        potential_traffic_gain = min(20, potential_indexation_gain * 0.6)
        
        return {
            "crawl_efficiency_score": round(crawl_efficiency_score, 1),
            "monthly_crawl_budget": int(monthly_crawl_budget),
            "wasted_budget_percentage": round(stats.wasted_budget_percentage, 2),
            "total_issues_identified": len(recommendations),
            "top_opportunities": [
                {
                    "issue": opp.issue,
                    "impact": opp.impact,
                    "affected_urls": opp.affected_urls,
                    "potential_savings": f"{opp.wasted_budget:.1f}% crawl budget"
                }
                for opp in top_opportunities
            ],
            "implementation_roadmap": implementation_roadmap,
            "expected_improvements": {
                "crawl_efficiency": f"+{potential_efficiency_gain:.0f}%",
                "indexation_rate": f"+{potential_indexation_gain:.0f}%",
                "organic_traffic": f"+{potential_traffic_gain:.0f}%"
            },
            "business_impact": {
                "technical_debt_reduction": f"{len([r for r in recommendations if r.impact in ['critical', 'high']])} critical issues addressed",
                "resource_optimization": f"Up to {stats.wasted_budget_percentage:.0f}% crawl budget recovered",
                "competitive_advantage": "Improved discovery and indexation of strategic content"
            },
            "next_steps": [
                "Implement redirect chain fixes within 48 hours",
                "Deploy updated XML sitemaps within 1 week", 
                "Establish ongoing crawl budget monitoring",
                "Schedule quarterly technical SEO audits"
            ]
        }