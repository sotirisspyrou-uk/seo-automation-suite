"""
Mobile-First Analyzer - Enterprise Mobile SEO Optimization Platform
Advanced mobile-first indexing readiness analysis and mobile SEO performance optimization

🎯 PORTFOLIO PROJECT: Demonstrates mobile SEO expertise and responsive design knowledge
Perfect for: Mobile developers, UX/UI specialists, technical SEO consultants

📄 DEMO/PORTFOLIO CODE: This is demonstration code showcasing mobile analysis capabilities.
   Real implementations require comprehensive device testing and performance measurement tools.

🔗 Connect with the developer: https://www.linkedin.com/in/sspyrou/
🚀 AI-Enhanced Mobile SEO Solutions: https://verityai.co

Built by a technical marketing leader with expertise in mobile-first strategies
that helped enterprises achieve optimal Google mobile-first indexing compliance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path
import re

import aiohttp
from bs4 import BeautifulSoup
import pandas as pd


@dataclass
class MobileIssue:
    """Individual mobile SEO issue"""
    url: str
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    recommendation: str
    technical_details: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class CoreWebVitals:
    """Core Web Vitals metrics"""
    lcp: Optional[float] = None  # Largest Contentful Paint (seconds)
    fid: Optional[float] = None  # First Input Delay (ms)
    cls: Optional[float] = None  # Cumulative Layout Shift
    fcp: Optional[float] = None  # First Contentful Paint (seconds)
    ttfb: Optional[float] = None  # Time to First Byte (ms)


@dataclass
class MobileAnalysisResult:
    """Mobile-first analysis results for a single URL"""
    url: str
    mobile_friendly: bool
    responsive_design: bool
    viewport_configured: bool
    mobile_content_parity: float  # 0-1 score
    mobile_speed_score: int  # 0-100
    core_web_vitals: CoreWebVitals
    issues: List[MobileIssue] = field(default_factory=list)
    mobile_usability_score: int = 0  # 0-100
    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class MobileSiteAnalysis:
    """Complete mobile-first analysis for entire site"""
    site_url: str
    overall_mobile_score: int  # 0-100
    mobile_ready_urls: int
    total_urls_analyzed: int
    critical_issues: int
    high_priority_issues: int
    url_results: List[MobileAnalysisResult] = field(default_factory=list)
    site_wide_recommendations: List[str] = field(default_factory=list)
    mobile_indexing_readiness: str = "unknown"  # ready, partial, not_ready
    analysis_summary: Dict[str, Any] = field(default_factory=dict)


class MobileFirstAnalyzer:
    """Comprehensive mobile-first indexing and mobile SEO analyzer"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "analysis": {
                "user_agents": {
                    "mobile": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
                    "desktop": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
                "viewport_sizes": {
                    "mobile": (375, 667),
                    "tablet": (768, 1024),
                    "desktop": (1920, 1080)
                },
                "timeout": 30,
                "max_concurrent": 10,
                "follow_redirects": True
            },
            "thresholds": {
                "mobile_speed_good": 90,
                "mobile_speed_needs_improvement": 50,
                "content_parity_threshold": 0.95,
                "lcp_good": 2.5,  # seconds
                "lcp_needs_improvement": 4.0,
                "fid_good": 100,  # milliseconds
                "fid_needs_improvement": 300,
                "cls_good": 0.1,
                "cls_needs_improvement": 0.25
            },
            "checks": {
                "viewport_meta": True,
                "responsive_images": True,
                "touch_targets": True,
                "mobile_redirects": True,
                "amp_validation": True,
                "structured_data": True,
                "mobile_sitemaps": True,
                "mobile_robots": True
            },
            "core_web_vitals": {
                "api_key": "",  # PageSpeed Insights API key
                "strategy": "mobile",
                "categories": ["performance", "accessibility", "best-practices", "seo"]
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.config["analysis"]["timeout"])
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def analyze_site_mobile_readiness(
        self, 
        site_url: str,
        urls_to_analyze: List[str] = None,
        sample_size: int = 50
    ) -> MobileSiteAnalysis:
        """Analyze entire site for mobile-first indexing readiness"""
        self.logger.info(f"Starting mobile-first analysis for {site_url}")
        
        # Get URLs to analyze
        if not urls_to_analyze:
            urls_to_analyze = await self._discover_key_urls(site_url, sample_size)
        
        # Analyze individual URLs
        semaphore = asyncio.Semaphore(self.config["analysis"]["max_concurrent"])
        tasks = [
            self._analyze_url_mobile_readiness(url, semaphore) 
            for url in urls_to_analyze
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        url_results = []
        for url, result in zip(urls_to_analyze, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error analyzing {url}: {result}")
                # Create error result
                error_result = MobileAnalysisResult(
                    url=url,
                    mobile_friendly=False,
                    responsive_design=False,
                    viewport_configured=False,
                    mobile_content_parity=0.0,
                    mobile_speed_score=0,
                    core_web_vitals=CoreWebVitals(),
                    mobile_usability_score=0
                )
                error_result.issues.append(MobileIssue(
                    url=url,
                    issue_type="analysis_error",
                    severity="high",
                    description=f"Failed to analyze URL: {str(result)}",
                    recommendation="Check URL accessibility and server response"
                ))
                url_results.append(error_result)
            else:
                url_results.append(result)
        
        # Compile site-wide analysis
        return self._compile_site_analysis(site_url, url_results)
    
    async def _discover_key_urls(self, site_url: str, sample_size: int) -> List[str]:
        """Discover key URLs for mobile analysis"""
        self.logger.info(f"Discovering key URLs for {site_url}")
        
        urls = set()
        
        # Always include homepage
        urls.add(site_url.rstrip('/'))
        
        # Try to get sitemap URLs
        sitemap_urls = await self._get_sitemap_urls(site_url)
        urls.update(sitemap_urls[:sample_size-1])
        
        # If still need more URLs, crawl from homepage
        if len(urls) < sample_size:
            crawled_urls = await self._crawl_key_pages(site_url, sample_size - len(urls))
            urls.update(crawled_urls)
        
        return list(urls)[:sample_size]
    
    async def _get_sitemap_urls(self, site_url: str) -> List[str]:
        """Extract URLs from sitemap.xml"""
        sitemap_url = urljoin(site_url, '/sitemap.xml')
        
        try:
            async with self.session.get(sitemap_url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'xml')
                    
                    urls = []
                    for loc in soup.find_all('loc'):
                        if loc.text:
                            urls.append(loc.text)
                    
                    self.logger.info(f"Found {len(urls)} URLs in sitemap")
                    return urls
        
        except Exception as e:
            self.logger.warning(f"Could not fetch sitemap: {e}")
        
        return []
    
    async def _crawl_key_pages(self, site_url: str, max_urls: int) -> List[str]:
        """Crawl key pages from homepage"""
        try:
            async with self.session.get(site_url) as response:
                if response.status != 200:
                    return []
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                urls = set()
                domain = urlparse(site_url).netloc
                
                # Find important page links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(site_url, href)
                    parsed = urlparse(full_url)
                    
                    # Only include same-domain URLs
                    if parsed.netloc == domain and len(urls) < max_urls:
                        urls.add(full_url)
                
                return list(urls)
        
        except Exception as e:
            self.logger.warning(f"Could not crawl homepage: {e}")
            return []
    
    async def _analyze_url_mobile_readiness(
        self, 
        url: str, 
        semaphore: asyncio.Semaphore
    ) -> MobileAnalysisResult:
        """Analyze single URL for mobile-first readiness"""
        async with semaphore:
            self.logger.debug(f"Analyzing mobile readiness for {url}")
            
            result = MobileAnalysisResult(
                url=url,
                mobile_friendly=True,
                responsive_design=True,
                viewport_configured=True,
                mobile_content_parity=1.0,
                mobile_speed_score=100,
                core_web_vitals=CoreWebVitals(),
                mobile_usability_score=100
            )
            
            try:
                # Fetch both mobile and desktop versions
                mobile_data = await self._fetch_with_user_agent(url, "mobile")
                desktop_data = await self._fetch_with_user_agent(url, "desktop")
                
                if not mobile_data or not desktop_data:
                    raise Exception("Failed to fetch mobile or desktop version")
                
                # Analyze viewport configuration
                await self._check_viewport_configuration(result, mobile_data)
                
                # Check responsive design
                await self._check_responsive_design(result, mobile_data)
                
                # Analyze mobile-desktop content parity
                await self._analyze_content_parity(result, mobile_data, desktop_data)
                
                # Check mobile usability
                await self._check_mobile_usability(result, mobile_data)
                
                # Analyze mobile performance
                await self._analyze_mobile_performance(result, url)
                
                # Check mobile-specific SEO elements
                await self._check_mobile_seo_elements(result, mobile_data)
                
                # Calculate overall mobile usability score
                result.mobile_usability_score = self._calculate_usability_score(result)
                
                return result
            
            except Exception as e:
                self.logger.error(f"Error analyzing {url}: {e}")
                result.mobile_friendly = False
                result.responsive_design = False
                result.mobile_content_parity = 0.0
                result.mobile_speed_score = 0
                result.mobile_usability_score = 0
                
                result.issues.append(MobileIssue(
                    url=url,
                    issue_type="analysis_error",
                    severity="critical",
                    description=f"Failed to analyze URL: {str(e)}",
                    recommendation="Verify URL accessibility and server configuration"
                ))
                
                return result
    
    async def _fetch_with_user_agent(
        self, 
        url: str, 
        device_type: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch URL with specific user agent"""
        user_agent = self.config["analysis"]["user_agents"][device_type]
        headers = {"User-Agent": user_agent}
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    return {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "content": content,
                        "soup": BeautifulSoup(content, 'html.parser'),
                        "device_type": device_type
                    }
        except Exception as e:
            self.logger.error(f"Failed to fetch {url} with {device_type} UA: {e}")
        
        return None
    
    async def _check_viewport_configuration(
        self, 
        result: MobileAnalysisResult, 
        mobile_data: Dict[str, Any]
    ):
        """Check viewport meta tag configuration"""
        soup = mobile_data["soup"]
        viewport_tag = soup.find('meta', attrs={'name': 'viewport'})
        
        if not viewport_tag:
            result.viewport_configured = False
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="viewport_missing",
                severity="critical",
                description="Viewport meta tag is missing",
                recommendation="Add <meta name='viewport' content='width=device-width, initial-scale=1'>"
            ))
            return
        
        viewport_content = viewport_tag.get('content', '')
        
        # Check for required viewport properties
        if 'width=device-width' not in viewport_content:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="viewport_width",
                severity="high",
                description="Viewport does not set width=device-width",
                recommendation="Include 'width=device-width' in viewport meta tag"
            ))
        
        if 'initial-scale=1' not in viewport_content:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="viewport_scale",
                severity="medium",
                description="Viewport does not set initial-scale=1",
                recommendation="Include 'initial-scale=1' in viewport meta tag"
            ))
        
        # Check for problematic viewport settings
        if 'user-scalable=no' in viewport_content:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="viewport_scalable",
                severity="medium",
                description="Viewport disables user scaling",
                recommendation="Allow user scaling for better accessibility"
            ))
    
    async def _check_responsive_design(
        self, 
        result: MobileAnalysisResult, 
        mobile_data: Dict[str, Any]
    ):
        """Check responsive design implementation"""
        soup = mobile_data["soup"]
        
        # Check for CSS media queries
        has_media_queries = False
        
        # Check inline styles
        for style_tag in soup.find_all('style'):
            if style_tag.string and '@media' in style_tag.string:
                has_media_queries = True
                break
        
        # Check linked stylesheets (simplified check)
        for link in soup.find_all('link', {'rel': 'stylesheet'}):
            media = link.get('media', '')
            if 'screen' in media or 'all' in media or not media:
                has_media_queries = True
                break
        
        if not has_media_queries:
            result.responsive_design = False
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="responsive_design",
                severity="high",
                description="No responsive design detected",
                recommendation="Implement responsive design with CSS media queries"
            ))
        
        # Check for fixed-width elements
        self._check_fixed_width_elements(result, soup)
        
        # Check for horizontal scrolling issues
        self._check_horizontal_scrolling(result, soup)
    
    def _check_fixed_width_elements(self, result: MobileAnalysisResult, soup: BeautifulSoup):
        """Check for problematic fixed-width elements"""
        # Check for tables without responsive styling
        tables = soup.find_all('table')
        for table in tables:
            if not table.get('class') or 'responsive' not in ' '.join(table.get('class', [])):
                result.issues.append(MobileIssue(
                    url=result.url,
                    issue_type="table_responsive",
                    severity="medium",
                    description="Table may not be mobile-friendly",
                    recommendation="Make tables responsive or use alternative layouts for mobile"
                ))
                break
        
        # Check for fixed-width inline styles
        elements_with_width = soup.find_all(style=True)
        for element in elements_with_width:
            style = element.get('style', '')
            if 'width:' in style and 'px' in style:
                # Simplified check - in reality would parse CSS properly
                result.issues.append(MobileIssue(
                    url=result.url,
                    issue_type="fixed_width",
                    severity="medium",
                    description="Fixed-width elements detected",
                    recommendation="Use relative units (%, em, rem) instead of fixed pixels"
                ))
                break
    
    def _check_horizontal_scrolling(self, result: MobileAnalysisResult, soup: BeautifulSoup):
        """Check for potential horizontal scrolling issues"""
        # Check for wide images without responsive attributes
        images = soup.find_all('img')
        for img in images:
            width = img.get('width')
            if width and width.isdigit() and int(width) > 400:
                if not img.get('class') or 'responsive' not in ' '.join(img.get('class', [])):
                    result.issues.append(MobileIssue(
                        url=result.url,
                        issue_type="image_responsive",
                        severity="medium",
                        description="Large images may cause horizontal scrolling",
                        recommendation="Make images responsive with max-width: 100%"
                    ))
                    break
    
    async def _analyze_content_parity(
        self, 
        result: MobileAnalysisResult, 
        mobile_data: Dict[str, Any], 
        desktop_data: Dict[str, Any]
    ):
        """Analyze content parity between mobile and desktop versions"""
        mobile_soup = mobile_data["soup"]
        desktop_soup = desktop_data["soup"]
        
        # Extract key content elements
        mobile_content = self._extract_content_elements(mobile_soup)
        desktop_content = self._extract_content_elements(desktop_soup)
        
        # Calculate content parity
        parity_score = self._calculate_content_parity(mobile_content, desktop_content)
        result.mobile_content_parity = parity_score
        
        threshold = self.config["thresholds"]["content_parity_threshold"]
        if parity_score < threshold:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="content_parity",
                severity="high",
                description=f"Mobile content parity is {parity_score:.1%} (below {threshold:.1%})",
                recommendation="Ensure mobile version includes all important content from desktop",
                technical_details={
                    "mobile_elements": len(mobile_content),
                    "desktop_elements": len(desktop_content),
                    "parity_score": parity_score
                }
            ))
        
        # Check specific content differences
        self._check_specific_content_differences(result, mobile_content, desktop_content)
    
    def _extract_content_elements(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract key content elements for comparison"""
        content = {
            "headings": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
            "paragraphs": [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()],
            "links": [a.get('href') for a in soup.find_all('a', href=True)],
            "images": [img.get('src') for img in soup.find_all('img', src=True)],
            "meta_title": soup.find('title').get_text() if soup.find('title') else "",
            "meta_description": soup.find('meta', {'name': 'description'}).get('content', '') if soup.find('meta', {'name': 'description'}) else ""
        }
        
        return content
    
    def _calculate_content_parity(
        self, 
        mobile_content: Dict[str, List[str]], 
        desktop_content: Dict[str, List[str]]
    ) -> float:
        """Calculate content parity score between mobile and desktop"""
        total_score = 0.0
        weights = {
            "headings": 0.3,
            "paragraphs": 0.25,
            "links": 0.2,
            "images": 0.15,
            "meta_title": 0.05,
            "meta_description": 0.05
        }
        
        for content_type, weight in weights.items():
            mobile_items = set(mobile_content.get(content_type, []))
            desktop_items = set(desktop_content.get(content_type, []))
            
            if not desktop_items:
                score = 1.0  # No desktop content to compare
            else:
                common_items = mobile_items.intersection(desktop_items)
                score = len(common_items) / len(desktop_items)
            
            total_score += score * weight
        
        return min(1.0, total_score)
    
    def _check_specific_content_differences(
        self, 
        result: MobileAnalysisResult,
        mobile_content: Dict[str, List[str]], 
        desktop_content: Dict[str, List[str]]
    ):
        """Check for specific content differences"""
        # Check meta title
        if mobile_content["meta_title"] != desktop_content["meta_title"]:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="meta_title_difference",
                severity="high",
                description="Meta title differs between mobile and desktop",
                recommendation="Ensure consistent meta titles across mobile and desktop"
            ))
        
        # Check critical headings (H1)
        mobile_h1s = [h for h in mobile_content["headings"] if h]
        desktop_h1s = [h for h in desktop_content["headings"] if h]
        
        if len(mobile_h1s) != len(desktop_h1s) or set(mobile_h1s) != set(desktop_h1s):
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="h1_difference",
                severity="high",
                description="H1 headings differ between mobile and desktop",
                recommendation="Maintain consistent H1 headings across all versions"
            ))
    
    async def _check_mobile_usability(
        self, 
        result: MobileAnalysisResult, 
        mobile_data: Dict[str, Any]
    ):
        """Check mobile usability factors"""
        soup = mobile_data["soup"]
        
        # Check touch target sizes
        self._check_touch_targets(result, soup)
        
        # Check font sizes
        self._check_font_sizes(result, soup)
        
        # Check mobile navigation
        self._check_mobile_navigation(result, soup)
        
        # Check form usability
        self._check_form_usability(result, soup)
    
    def _check_touch_targets(self, result: MobileAnalysisResult, soup: BeautifulSoup):
        """Check touch target accessibility"""
        # Check button and link spacing
        buttons = soup.find_all(['button', 'a', 'input'])
        
        small_targets_found = False
        for button in buttons[:10]:  # Check first 10
            style = button.get('style', '')
            # Simplified check - would need more sophisticated parsing
            if 'padding' not in style and 'height' not in style:
                small_targets_found = True
                break
        
        if small_targets_found:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="touch_targets",
                severity="medium",
                description="Some touch targets may be too small",
                recommendation="Ensure touch targets are at least 44x44 pixels with adequate spacing"
            ))
    
    def _check_font_sizes(self, result: MobileAnalysisResult, soup: BeautifulSoup):
        """Check font size accessibility"""
        # This is a simplified check - in practice would analyze computed styles
        meta_viewport = soup.find('meta', {'name': 'viewport'})
        if meta_viewport and 'user-scalable=no' in meta_viewport.get('content', ''):
            # If scaling is disabled, font sizes become more critical
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="font_size_scaling",
                severity="medium",
                description="User scaling disabled - ensure adequate font sizes",
                recommendation="Use minimum 16px font size for body text or enable user scaling"
            ))
    
    def _check_mobile_navigation(self, result: MobileAnalysisResult, soup: BeautifulSoup):
        """Check mobile navigation implementation"""
        # Look for mobile navigation patterns
        nav_elements = soup.find_all(['nav', 'ul', 'ol'])
        hamburger_found = False
        
        for nav in nav_elements:
            # Check for hamburger menu indicators
            classes = ' '.join(nav.get('class', [])).lower()
            if any(term in classes for term in ['hamburger', 'menu-toggle', 'mobile-menu']):
                hamburger_found = True
                break
        
        # Check for navigation in header
        header = soup.find('header')
        if header and not hamburger_found:
            nav_links = header.find_all('a')
            if len(nav_links) > 5:  # Many links might indicate poor mobile navigation
                result.issues.append(MobileIssue(
                    url=result.url,
                    issue_type="mobile_navigation",
                    severity="medium",
                    description="Navigation may not be optimized for mobile",
                    recommendation="Implement hamburger menu or collapsible navigation for mobile"
                ))
    
    def _check_form_usability(self, result: MobileAnalysisResult, soup: BeautifulSoup):
        """Check form mobile usability"""
        forms = soup.find_all('form')
        
        for form in forms:
            inputs = form.find_all(['input', 'select', 'textarea'])
            
            for input_elem in inputs:
                input_type = input_elem.get('type', 'text')
                
                # Check for appropriate input types
                if input_elem.name == 'input':
                    if 'email' in str(input_elem).lower() and input_type != 'email':
                        result.issues.append(MobileIssue(
                            url=result.url,
                            issue_type="input_type",
                            severity="low",
                            description="Email inputs not using type='email'",
                            recommendation="Use appropriate input types (email, tel, number) for better mobile keyboards"
                        ))
                        break
    
    async def _analyze_mobile_performance(
        self, 
        result: MobileAnalysisResult, 
        url: str
    ):
        """Analyze mobile performance metrics"""
        # In practice, would integrate with PageSpeed Insights API
        # For demo, we'll simulate performance analysis
        
        # Simulate Core Web Vitals (would come from real API)
        result.core_web_vitals = CoreWebVitals(
            lcp=2.3,  # Simulated values
            fid=85,
            cls=0.12,
            fcp=1.8,
            ttfb=200
        )
        
        # Simulate mobile speed score
        result.mobile_speed_score = 78  # Simulated score
        
        # Check performance against thresholds
        self._evaluate_core_web_vitals(result)
    
    def _evaluate_core_web_vitals(self, result: MobileAnalysisResult):
        """Evaluate Core Web Vitals against Google's thresholds"""
        cwv = result.core_web_vitals
        thresholds = self.config["thresholds"]
        
        # LCP evaluation
        if cwv.lcp and cwv.lcp > thresholds["lcp_needs_improvement"]:
            severity = "high" if cwv.lcp > 4.0 else "medium"
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="lcp_slow",
                severity=severity,
                description=f"Largest Contentful Paint is {cwv.lcp:.1f}s (should be <2.5s)",
                recommendation="Optimize images, server response times, and render-blocking resources"
            ))
        
        # FID evaluation
        if cwv.fid and cwv.fid > thresholds["fid_needs_improvement"]:
            severity = "high" if cwv.fid > 300 else "medium"
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="fid_slow",
                severity=severity,
                description=f"First Input Delay is {cwv.fid:.0f}ms (should be <100ms)",
                recommendation="Reduce JavaScript execution time and eliminate render-blocking scripts"
            ))
        
        # CLS evaluation
        if cwv.cls and cwv.cls > thresholds["cls_needs_improvement"]:
            severity = "high" if cwv.cls > 0.25 else "medium"
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="cls_poor",
                severity=severity,
                description=f"Cumulative Layout Shift is {cwv.cls:.2f} (should be <0.1)",
                recommendation="Set dimensions for images and ads, avoid inserting content above existing content"
            ))
    
    async def _check_mobile_seo_elements(
        self, 
        result: MobileAnalysisResult, 
        mobile_data: Dict[str, Any]
    ):
        """Check mobile-specific SEO elements"""
        soup = mobile_data["soup"]
        
        # Check for AMP version
        amp_link = soup.find('link', {'rel': 'amphtml'})
        if amp_link:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="amp_detected",
                severity="low",
                description="AMP version detected",
                recommendation="Ensure AMP implementation follows best practices and is properly validated"
            ))
        
        # Check structured data (simplified)
        json_ld = soup.find_all('script', {'type': 'application/ld+json'})
        if not json_ld:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="structured_data_missing",
                severity="medium",
                description="No structured data detected",
                recommendation="Implement JSON-LD structured data for better search visibility"
            ))
        
        # Check mobile-specific meta tags
        self._check_mobile_meta_tags(result, soup)
    
    def _check_mobile_meta_tags(self, result: MobileAnalysisResult, soup: BeautifulSoup):
        """Check mobile-specific meta tags"""
        # Check for mobile web app meta tags
        apple_mobile_capable = soup.find('meta', {'name': 'apple-mobile-web-app-capable'})
        if apple_mobile_capable and apple_mobile_capable.get('content') == 'yes':
            status_bar_style = soup.find('meta', {'name': 'apple-mobile-web-app-status-bar-style'})
            if not status_bar_style:
                result.issues.append(MobileIssue(
                    url=result.url,
                    issue_type="mobile_app_meta",
                    severity="low",
                    description="Mobile web app meta tags incomplete",
                    recommendation="Add apple-mobile-web-app-status-bar-style meta tag"
                ))
        
        # Check for theme color
        theme_color = soup.find('meta', {'name': 'theme-color'})
        if not theme_color:
            result.issues.append(MobileIssue(
                url=result.url,
                issue_type="theme_color_missing",
                severity="low",
                description="No theme-color meta tag found",
                recommendation="Add theme-color meta tag for better browser integration"
            ))
    
    def _calculate_usability_score(self, result: MobileAnalysisResult) -> int:
        """Calculate overall mobile usability score"""
        base_score = 100
        
        # Deduct points based on issues
        deductions = {
            "critical": 25,
            "high": 15,
            "medium": 8,
            "low": 3
        }
        
        for issue in result.issues:
            deduction = deductions.get(issue.severity, 5)
            base_score -= deduction
        
        # Bonus for good practices
        if result.viewport_configured:
            base_score += 5
        if result.responsive_design:
            base_score += 10
        if result.mobile_content_parity > 0.95:
            base_score += 5
        
        return max(0, min(100, base_score))
    
    def _compile_site_analysis(
        self, 
        site_url: str, 
        url_results: List[MobileAnalysisResult]
    ) -> MobileSiteAnalysis:
        """Compile site-wide mobile analysis results"""
        total_urls = len(url_results)
        mobile_ready_urls = len([r for r in url_results if r.mobile_usability_score >= 80])
        
        # Count issues by severity
        critical_issues = sum(len([i for i in r.issues if i.severity == "critical"]) for r in url_results)
        high_priority_issues = sum(len([i for i in r.issues if i.severity == "high"]) for r in url_results)
        
        # Calculate overall mobile score
        if url_results:
            overall_score = sum(r.mobile_usability_score for r in url_results) // len(url_results)
        else:
            overall_score = 0
        
        # Determine mobile indexing readiness
        if critical_issues == 0 and high_priority_issues < total_urls * 0.1:
            indexing_readiness = "ready"
        elif critical_issues < total_urls * 0.05:
            indexing_readiness = "partial"
        else:
            indexing_readiness = "not_ready"
        
        # Generate site-wide recommendations
        site_recommendations = self._generate_site_recommendations(url_results)
        
        # Create analysis summary
        analysis_summary = {
            "mobile_readiness_percentage": (mobile_ready_urls / total_urls * 100) if total_urls > 0 else 0,
            "average_mobile_score": overall_score,
            "average_content_parity": sum(r.mobile_content_parity for r in url_results) / len(url_results) if url_results else 0,
            "common_issues": self._identify_common_issues(url_results)
        }
        
        return MobileSiteAnalysis(
            site_url=site_url,
            overall_mobile_score=overall_score,
            mobile_ready_urls=mobile_ready_urls,
            total_urls_analyzed=total_urls,
            critical_issues=critical_issues,
            high_priority_issues=high_priority_issues,
            url_results=url_results,
            site_wide_recommendations=site_recommendations,
            mobile_indexing_readiness=indexing_readiness,
            analysis_summary=analysis_summary
        )
    
    def _generate_site_recommendations(
        self, 
        url_results: List[MobileAnalysisResult]
    ) -> List[str]:
        """Generate site-wide mobile optimization recommendations"""
        recommendations = []
        
        # Analyze common issues across URLs
        issue_counts = {}
        for result in url_results:
            for issue in result.issues:
                issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        total_urls = len(url_results)
        
        # Generate recommendations based on common issues
        if issue_counts.get("viewport_missing", 0) > total_urls * 0.2:
            recommendations.append("Implement proper viewport meta tags site-wide")
        
        if issue_counts.get("content_parity", 0) > total_urls * 0.1:
            recommendations.append("Ensure mobile content parity with desktop versions")
        
        if issue_counts.get("responsive_design", 0) > total_urls * 0.1:
            recommendations.append("Implement responsive design patterns across all pages")
        
        if issue_counts.get("lcp_slow", 0) > total_urls * 0.3:
            recommendations.append("Optimize site-wide Core Web Vitals, especially Largest Contentful Paint")
        
        if issue_counts.get("touch_targets", 0) > total_urls * 0.2:
            recommendations.append("Review and optimize touch target sizes for mobile usability")
        
        # Add general recommendations
        recommendations.extend([
            "Monitor mobile performance metrics continuously",
            "Test mobile experience across various devices and screen sizes",
            "Implement progressive web app features for enhanced mobile experience",
            "Regular mobile SEO audits to maintain Google mobile-first indexing compliance"
        ])
        
        return recommendations[:8]  # Return top 8 recommendations
    
    def _identify_common_issues(
        self, 
        url_results: List[MobileAnalysisResult]
    ) -> List[Dict[str, Any]]:
        """Identify most common issues across the site"""
        issue_counts = {}
        total_urls = len(url_results)
        
        for result in url_results:
            for issue in result.issues:
                key = f"{issue.issue_type}:{issue.severity}"
                if key not in issue_counts:
                    issue_counts[key] = {
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "count": 0,
                        "description": issue.description,
                        "recommendation": issue.recommendation
                    }
                issue_counts[key]["count"] += 1
        
        # Sort by count and return top issues
        common_issues = sorted(
            issue_counts.values(), 
            key=lambda x: x["count"], 
            reverse=True
        )[:5]
        
        # Add percentage
        for issue in common_issues:
            issue["percentage"] = (issue["count"] / total_urls * 100) if total_urls > 0 else 0
        
        return common_issues
    
    def export_mobile_analysis(
        self, 
        analysis: MobileSiteAnalysis, 
        output_path: str,
        format: str = "json"
    ):
        """Export mobile analysis results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            self._export_json(analysis, output_path)
        elif format == "csv":
            self._export_csv(analysis, output_path)
        elif format == "html":
            self._export_html(analysis, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Mobile analysis exported to {output_path}")
    
    def _export_json(self, analysis: MobileSiteAnalysis, path: Path):
        """Export analysis as JSON"""
        export_data = {
            "site_url": analysis.site_url,
            "overall_mobile_score": analysis.overall_mobile_score,
            "mobile_indexing_readiness": analysis.mobile_indexing_readiness,
            "summary": {
                "total_urls": analysis.total_urls_analyzed,
                "mobile_ready_urls": analysis.mobile_ready_urls,
                "critical_issues": analysis.critical_issues,
                "high_priority_issues": analysis.high_priority_issues
            },
            "analysis_summary": analysis.analysis_summary,
            "recommendations": analysis.site_wide_recommendations,
            "url_results": []
        }
        
        for result in analysis.url_results:
            url_data = {
                "url": result.url,
                "mobile_usability_score": result.mobile_usability_score,
                "mobile_friendly": result.mobile_friendly,
                "responsive_design": result.responsive_design,
                "viewport_configured": result.viewport_configured,
                "mobile_content_parity": result.mobile_content_parity,
                "mobile_speed_score": result.mobile_speed_score,
                "core_web_vitals": {
                    "lcp": result.core_web_vitals.lcp,
                    "fid": result.core_web_vitals.fid,
                    "cls": result.core_web_vitals.cls
                },
                "issues": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "recommendation": issue.recommendation
                    }
                    for issue in result.issues
                ]
            }
            export_data["url_results"].append(url_data)
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _export_csv(self, analysis: MobileSiteAnalysis, path: Path):
        """Export analysis as CSV"""
        rows = []
        for result in analysis.url_results:
            rows.append({
                "url": result.url,
                "mobile_usability_score": result.mobile_usability_score,
                "mobile_friendly": result.mobile_friendly,
                "responsive_design": result.responsive_design,
                "viewport_configured": result.viewport_configured,
                "content_parity": result.mobile_content_parity,
                "speed_score": result.mobile_speed_score,
                "lcp": result.core_web_vitals.lcp,
                "fid": result.core_web_vitals.fid,
                "cls": result.core_web_vitals.cls,
                "critical_issues": len([i for i in result.issues if i.severity == "critical"]),
                "high_issues": len([i for i in result.issues if i.severity == "high"]),
                "total_issues": len(result.issues)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    
    def _export_html(self, analysis: MobileSiteAnalysis, path: Path):
        """Export analysis as HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mobile-First SEO Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
                .score {{ font-size: 2em; font-weight: bold; color: #2e7d32; }}
                .critical {{ color: #d32f2f; }}
                .high {{ color: #f57c00; }}
                .medium {{ color: #fbc02d; }}
                .low {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendations {{ background: #e8f5e8; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Mobile-First SEO Analysis Report</h1>
            
            <div class="summary">
                <h2>Overall Mobile Score: <span class="score">{analysis.overall_mobile_score}</span></h2>
                <p><strong>Mobile Indexing Readiness:</strong> {analysis.mobile_indexing_readiness}</p>
                <p><strong>URLs Analyzed:</strong> {analysis.total_urls_analyzed}</p>
                <p><strong>Mobile-Ready URLs:</strong> {analysis.mobile_ready_urls} ({analysis.mobile_ready_urls/analysis.total_urls_analyzed*100:.1f}%)</p>
                <p><strong>Critical Issues:</strong> <span class="critical">{analysis.critical_issues}</span></p>
                <p><strong>High Priority Issues:</strong> <span class="high">{analysis.high_priority_issues}</span></p>
            </div>
            
            <div class="recommendations">
                <h2>Key Recommendations</h2>
                <ul>
        """
        
        for rec in analysis.site_wide_recommendations[:5]:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <h2>URL Analysis Results</h2>
            <table>
                <tr>
                    <th>URL</th>
                    <th>Mobile Score</th>
                    <th>Mobile Friendly</th>
                    <th>Responsive</th>
                    <th>Content Parity</th>
                    <th>Issues</th>
                </tr>
        """
        
        for result in analysis.url_results[:20]:  # Limit to first 20 for readability
            issues_count = len(result.issues)
            html_content += f"""
                <tr>
                    <td>{result.url}</td>
                    <td>{result.mobile_usability_score}</td>
                    <td>{'✓' if result.mobile_friendly else '✗'}</td>
                    <td>{'✓' if result.responsive_design else '✗'}</td>
                    <td>{result.mobile_content_parity:.1%}</td>
                    <td>{issues_count}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(path, 'w') as f:
            f.write(html_content)


async def main():
    """Demo usage of Mobile-First Analyzer"""
    
    async with MobileFirstAnalyzer() as analyzer:
        print("Mobile-First SEO Analyzer Demo")
        
        # Analyze site mobile readiness
        site_url = "https://example.com"
        sample_urls = [
            "https://example.com",
            "https://example.com/about",
            "https://example.com/products",
            "https://example.com/contact"
        ]
        
        print(f"\n📱 Analyzing mobile readiness for {site_url}...")
        
        analysis = await analyzer.analyze_site_mobile_readiness(
            site_url=site_url,
            urls_to_analyze=sample_urls
        )
        
        print(f"\nMobile-First Analysis Results:")
        print(f"Overall Mobile Score: {analysis.overall_mobile_score}/100")
        print(f"Mobile Indexing Readiness: {analysis.mobile_indexing_readiness}")
        print(f"URLs Analyzed: {analysis.total_urls_analyzed}")
        print(f"Mobile-Ready URLs: {analysis.mobile_ready_urls}")
        print(f"Mobile Readiness: {analysis.analysis_summary['mobile_readiness_percentage']:.1f}%")
        
        if analysis.critical_issues > 0:
            print(f"🚨 Critical Issues: {analysis.critical_issues}")
        
        if analysis.high_priority_issues > 0:
            print(f"⚠️  High Priority Issues: {analysis.high_priority_issues}")
        
        print(f"\n🎯 Top Recommendations:")
        for i, rec in enumerate(analysis.site_wide_recommendations[:3], 1):
            print(f"{i}. {rec}")
        
        # Show common issues
        if analysis.analysis_summary["common_issues"]:
            print(f"\n📊 Most Common Issues:")
            for issue in analysis.analysis_summary["common_issues"][:3]:
                print(f"• {issue['issue_type']}: {issue['count']} URLs ({issue['percentage']:.1f}%)")
        
        # Export results
        analyzer.export_mobile_analysis(analysis, "mobile_analysis.json", "json")
        print(f"\n✅ Analysis exported to mobile_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())