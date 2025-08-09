"""
Brand Consistency Monitor - Enterprise Brand Compliance and SEO Alignment Platform
Advanced brand monitoring with SEO impact analysis and automated compliance tracking

<¯ PORTFOLIO PROJECT: Demonstrates brand management expertise and marketing automation
=T Perfect for: Brand managers, marketing directors, enterprise marketing operations teams

   DEMO/PORTFOLIO CODE: This is demonstration code showcasing brand monitoring capabilities.
    Real implementations require brand asset integration and approval workflows.

= Connect with the developer: https://www.linkedin.com/in/sspyrou/
=€ AI-Enhanced Brand Solutions: https://verityai.co

Built by a technical marketing leader with expertise in brand governance,
marketing automation, and the critical intersection of brand consistency with SEO performance.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import aiohttp
from bs4 import BeautifulSoup
import structlog

logger = structlog.get_logger()


@dataclass
class BrandViolation:
    """Brand guideline violation with business impact assessment"""
    violation_type: str  # "logo_usage", "color_scheme", "typography", "messaging", "tone_of_voice"
    severity: str  # "critical", "major", "minor"
    element: str
    selector: str
    current_value: str
    expected_value: str
    page_url: str
    description: str
    brand_impact: str
    seo_impact: str
    user_impact: str
    remediation_steps: List[str]
    automated_fix_available: bool
    approval_required: bool


@dataclass
class BrandMetrics:
    """Comprehensive brand consistency metrics with performance correlation"""
    url: str
    overall_brand_score: float
    logo_compliance_score: float
    color_consistency_score: float
    typography_score: float
    messaging_consistency_score: float
    tone_alignment_score: float
    meta_brand_optimization: float
    brand_keyword_density: float
    brand_mention_sentiment: float
    social_proof_implementation: float
    trust_signals_score: float


@dataclass
class BrandInsight:
    """Strategic brand insights for marketing leadership"""
    insight_type: str
    title: str
    description: str
    brand_risk_level: str
    business_impact: str
    affected_touchpoints: int
    revenue_implications: str
    recommended_actions: List[str]
    success_metrics: List[str]
    implementation_timeline: str


class BrandConsistencyMonitor:
    """Enterprise-Grade Brand Consistency and SEO Alignment Platform
    
    Perfect for: Brand managers, marketing directors, enterprise marketing operations
    Demonstrates: Brand governance, marketing automation, SEO-brand alignment
    
    Business Value:
    " Automated brand guideline compliance monitoring at scale
    " SEO impact analysis of brand consistency violations
    " Real-time brand asset usage tracking and approval workflows
    " Executive dashboards for brand performance and risk assessment
    " Integration with marketing automation and content management systems
    
    <¯ Portfolio Highlight: Showcases sophisticated understanding of brand management
       at enterprise scale, demonstrating ability to build systems that protect and
       enhance brand value while optimizing for search performance.
    """
    
    def __init__(self, brand_guidelines: Dict = None):
        self.brand_guidelines = brand_guidelines or self._default_brand_guidelines()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Brand monitoring configuration
        self.monitoring_intervals = {
            "critical_pages": timedelta(hours=1),
            "product_pages": timedelta(hours=6),
            "content_pages": timedelta(days=1),
            "archive_pages": timedelta(days=7)
        }
        
    def _default_brand_guidelines(self) -> Dict:
        """Default brand guidelines for demonstration"""
        return {
            "primary_colors": ["#0066CC", "#FF6600", "#333333"],
            "secondary_colors": ["#F5F5F5", "#E6E6E6", "#999999"],
            "fonts": {
                "primary": ["Arial", "Helvetica", "sans-serif"],
                "heading": ["Georgia", "Times", "serif"],
                "monospace": ["Monaco", "Courier", "monospace"]
            },
            "logo_requirements": {
                "min_size": "120px",
                "clear_space": "20px",
                "approved_formats": [".svg", ".png"],
                "placement_rules": ["header", "footer"]
            },
            "messaging": {
                "taglines": ["Innovation Through Excellence", "Leading Tomorrow"],
                "prohibited_terms": ["cheap", "discount", "basic"],
                "tone_keywords": ["professional", "innovative", "reliable", "expert"]
            },
            "seo_brand_keywords": ["company name", "brand name", "trademark terms"],
            "social_proof": {
                "testimonials_required": True,
                "awards_display": True,
                "certifications": ["ISO 9001", "SOC 2", "GDPR Compliant"]
            }
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Brand-Consistency-Monitor/1.0'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def monitor_brand_consistency(self, urls: List[str], priority_level: str = "standard") -> Dict:
        """Comprehensive brand consistency monitoring across multiple URLs
        
        Executive Value: Complete brand compliance assessment with business risk analysis
        """
        logger.info("starting_brand_consistency_monitoring", urls=len(urls), priority=priority_level)
        
        monitoring_results = {
            "monitoring_summary": {},
            "page_results": [],
            "violations": [],
            "insights": [],
            "brand_score": 0.0,
            "compliance_status": "",
            "risk_assessment": {}
        }
        
        # Process URLs based on priority
        batch_size = 10 if priority_level == "high" else 5
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_tasks = [self._monitor_page_brand_consistency(url) for url in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error("page_monitoring_error", url=batch[j], error=str(result))
                    continue
                    
                if result:
                    monitoring_results["page_results"].append(result)
                    if result.get("violations"):
                        monitoring_results["violations"].extend(result["violations"])
        
        # Generate comprehensive analysis
        monitoring_results["monitoring_summary"] = self._generate_monitoring_summary(
            monitoring_results["page_results"]
        )
        monitoring_results["insights"] = self._generate_brand_insights(
            monitoring_results["page_results"], monitoring_results["violations"]
        )
        monitoring_results["brand_score"] = self._calculate_overall_brand_score(
            monitoring_results["page_results"]
        )
        monitoring_results["compliance_status"] = self._assess_compliance_status(
            monitoring_results["brand_score"], monitoring_results["violations"]
        )
        monitoring_results["risk_assessment"] = self._assess_brand_risks(
            monitoring_results["violations"]
        )
        
        logger.info("brand_consistency_monitoring_complete",
                   score=monitoring_results["brand_score"],
                   violations=len(monitoring_results["violations"]))
        
        return monitoring_results
    
    async def _monitor_page_brand_consistency(self, url: str) -> Optional[Dict]:
        """Monitor brand consistency for a single page"""
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Comprehensive brand analysis
                violations = []
                
                # Logo usage compliance
                violations.extend(self._check_logo_compliance(soup, url))
                
                # Color scheme compliance
                violations.extend(self._check_color_compliance(soup, url))
                
                # Typography compliance
                violations.extend(self._check_typography_compliance(soup, url))
                
                # Messaging compliance
                violations.extend(self._check_messaging_compliance(soup, url))
                
                # Meta tag brand optimization
                violations.extend(self._check_meta_brand_optimization(soup, url))
                
                # Social proof implementation
                violations.extend(self._check_social_proof_implementation(soup, url))
                
                # Brand keyword optimization
                violations.extend(self._check_brand_keyword_optimization(soup, url))
                
                # Trust signals
                violations.extend(self._check_trust_signals(soup, url))
                
                # Calculate brand metrics
                metrics = self._calculate_brand_metrics(url, violations, soup)
                
                return {
                    "url": url,
                    "metrics": asdict(metrics),
                    "violations": [asdict(v) for v in violations],
                    "monitoring_timestamp": datetime.now().isoformat(),
                    "page_classification": self._classify_page_type(url, soup)
                }
                
        except Exception as e:
            logger.error("page_brand_monitoring_error", url=url, error=str(e))
            return None
    
    def _check_logo_compliance(self, soup: BeautifulSoup, url: str) -> List[BrandViolation]:
        """Check logo usage compliance against brand guidelines"""
        violations = []
        
        # Find logo images
        logo_candidates = soup.find_all('img', src=re.compile(r'logo|brand', re.I))
        logo_candidates.extend(soup.find_all('img', alt=re.compile(r'logo|brand', re.I)))
        
        if not logo_candidates:
            violations.append(BrandViolation(
                violation_type="missing_logo",
                severity="major",
                element="img",
                selector="header, .header, .logo",
                current_value="No logo found",
                expected_value="Brand logo in header",
                page_url=url,
                description="Brand logo not found on page",
                brand_impact="High - Missing primary brand identifier",
                seo_impact="Moderate - Missing brand recognition signals",
                user_impact="High - Reduced brand recognition and trust",
                remediation_steps=[
                    "Add brand logo to page header",
                    "Ensure logo meets minimum size requirements",
                    "Use approved logo format (SVG or high-res PNG)",
                    "Include proper alt text for accessibility"
                ],
                automated_fix_available=False,
                approval_required=True
            ))
        
        for logo in logo_candidates:
            # Check logo size requirements
            width = logo.get('width')
            height = logo.get('height')
            
            if width and int(width.replace('px', '')) < 120:
                violations.append(BrandViolation(
                    violation_type="logo_too_small",
                    severity="minor",
                    element="img",
                    selector=self._generate_selector(logo),
                    current_value=f"Width: {width}",
                    expected_value="Minimum width: 120px",
                    page_url=url,
                    description="Logo below minimum size requirements",
                    brand_impact="Moderate - Reduced brand visibility",
                    seo_impact="Low - Minimal direct SEO impact",
                    user_impact="Moderate - Logo may not be clearly visible",
                    remediation_steps=[
                        "Increase logo size to meet brand guidelines",
                        "Ensure logo remains legible at minimum size",
                        "Test logo visibility across devices"
                    ],
                    automated_fix_available=True,
                    approval_required=False
                ))
            
            # Check logo alt text
            alt_text = logo.get('alt', '').lower()
            if not alt_text or not any(keyword in alt_text for keyword in ['logo', 'brand']):
                violations.append(BrandViolation(
                    violation_type="logo_missing_alt_text",
                    severity="minor",
                    element="img",
                    selector=self._generate_selector(logo),
                    current_value=f"Alt text: '{logo.get('alt', 'missing')}'",
                    expected_value="Descriptive alt text including brand name",
                    page_url=url,
                    description="Logo missing proper alt text",
                    brand_impact="Low - Accessibility and brand recognition",
                    seo_impact="Moderate - Missing brand keyword opportunity",
                    user_impact="Moderate - Accessibility issue for screen readers",
                    remediation_steps=[
                        "Add descriptive alt text including brand name",
                        "Ensure alt text is concise but informative",
                        "Include relevant brand keywords naturally"
                    ],
                    automated_fix_available=True,
                    approval_required=False
                ))
        
        return violations
    
    def _check_color_compliance(self, soup: BeautifulSoup, url: str) -> List[BrandViolation]:
        """Check color scheme compliance"""
        violations = []
        
        # Find elements with inline styles using colors
        elements_with_color = soup.find_all(attrs={"style": re.compile(r'color:', re.I)})
        
        for element in elements_with_color:
            style = element.get('style', '').lower()
            
            # Extract colors from style
            color_matches = re.findall(r'color:\s*([^;]+)', style)
            
            for color in color_matches:
                color = color.strip()
                
                # Check if color is in approved brand palette
                if not self._is_approved_color(color):
                    violations.append(BrandViolation(
                        violation_type="unauthorized_color_usage",
                        severity="minor",
                        element=element.name,
                        selector=self._generate_selector(element),
                        current_value=f"Color: {color}",
                        expected_value="Brand-approved colors only",
                        page_url=url,
                        description=f"Unauthorized color usage: {color}",
                        brand_impact="Moderate - Inconsistent brand presentation",
                        seo_impact="Low - Minimal direct SEO impact",
                        user_impact="Low - Visual consistency affected",
                        remediation_steps=[
                            "Replace with approved brand colors",
                            "Update style guide reference",
                            "Review design approval process"
                        ],
                        automated_fix_available=True,
                        approval_required=True
                    ))
        
        return violations
    
    def _check_typography_compliance(self, soup: BeautifulSoup, url: str) -> List[BrandViolation]:
        """Check typography compliance against brand guidelines"""
        violations = []
        
        # Check for font-family usage
        elements_with_fonts = soup.find_all(attrs={"style": re.compile(r'font-family:', re.I)})
        
        for element in elements_with_fonts:
            style = element.get('style', '').lower()
            font_matches = re.findall(r'font-family:\s*([^;]+)', style)
            
            for font_family in font_matches:
                font_family = font_family.strip().replace('"', '').replace("'", "")
                
                if not self._is_approved_font(font_family):
                    violations.append(BrandViolation(
                        violation_type="unauthorized_font_usage",
                        severity="minor",
                        element=element.name,
                        selector=self._generate_selector(element),
                        current_value=f"Font: {font_family}",
                        expected_value="Brand-approved fonts only",
                        page_url=url,
                        description=f"Unauthorized font usage: {font_family}",
                        brand_impact="Moderate - Typography inconsistency",
                        seo_impact="Low - Minimal direct impact",
                        user_impact="Low - Visual brand consistency affected",
                        remediation_steps=[
                            "Replace with approved brand fonts",
                            "Update CSS to use brand font stack",
                            "Ensure font licensing compliance"
                        ],
                        automated_fix_available=True,
                        approval_required=False
                    ))
        
        return violations
    
    def _check_messaging_compliance(self, soup: BeautifulSoup, url: str) -> List[BrandViolation]:
        """Check messaging and tone compliance"""
        violations = []
        
        # Get all text content
        text_content = soup.get_text().lower()
        
        # Check for prohibited terms
        prohibited_terms = self.brand_guidelines.get("messaging", {}).get("prohibited_terms", [])
        for term in prohibited_terms:
            if term.lower() in text_content:
                violations.append(BrandViolation(
                    violation_type="prohibited_term_usage",
                    severity="major",
                    element="text_content",
                    selector="page",
                    current_value=f"Contains: {term}",
                    expected_value="Avoid prohibited terms",
                    page_url=url,
                    description=f"Prohibited term '{term}' found in page content",
                    brand_impact="High - Contradicts brand positioning",
                    seo_impact="Moderate - May affect brand keyword rankings",
                    user_impact="Moderate - Inconsistent brand messaging",
                    remediation_steps=[
                        f"Remove or replace '{term}' with approved alternatives",
                        "Review content against brand messaging guidelines",
                        "Implement content approval workflow"
                    ],
                    automated_fix_available=False,
                    approval_required=True
                ))
        
        # Check for brand tagline presence on key pages
        taglines = self.brand_guidelines.get("messaging", {}).get("taglines", [])
        if taglines and self._is_key_page(url):
            tagline_found = any(tagline.lower() in text_content for tagline in taglines)
            
            if not tagline_found:
                violations.append(BrandViolation(
                    violation_type="missing_brand_tagline",
                    severity="minor",
                    element="page_content",
                    selector="page",
                    current_value="No tagline found",
                    expected_value="Include brand tagline on key pages",
                    page_url=url,
                    description="Brand tagline missing from key page",
                    brand_impact="Moderate - Missed brand reinforcement opportunity",
                    seo_impact="Low - Potential brand keyword opportunity",
                    user_impact="Low - Reduced brand messaging consistency",
                    remediation_steps=[
                        "Add approved brand tagline to page",
                        "Ensure tagline placement follows guidelines",
                        "Consider SEO benefits of consistent messaging"
                    ],
                    automated_fix_available=True,
                    approval_required=False
                ))
        
        return violations
    
    def _check_meta_brand_optimization(self, soup: BeautifulSoup, url: str) -> List[BrandViolation]:
        """Check meta tag brand optimization"""
        violations = []
        
        # Check title tag for brand inclusion
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text().lower()
            brand_keywords = self.brand_guidelines.get("seo_brand_keywords", [])
            
            brand_in_title = any(keyword.lower() in title_text for keyword in brand_keywords)
            
            if not brand_in_title:
                violations.append(BrandViolation(
                    violation_type="brand_missing_from_title",
                    severity="minor",
                    element="title",
                    selector="title",
                    current_value=title_tag.get_text(),
                    expected_value="Include brand name in title",
                    page_url=url,
                    description="Brand name missing from page title",
                    brand_impact="Moderate - Missed brand visibility opportunity",
                    seo_impact="High - Brand keywords missing from most important SEO element",
                    user_impact="Low - Search result may lack brand context",
                    remediation_steps=[
                        "Include brand name in page title",
                        "Balance brand name with page-specific keywords",
                        "Follow title tag SEO best practices"
                    ],
                    automated_fix_available=True,
                    approval_required=False
                ))
        
        # Check meta description for brand optimization
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            desc_content = meta_desc.get('content', '').lower()
            brand_in_description = any(keyword.lower() in desc_content for keyword in brand_keywords)
            
            if not brand_in_description:
                violations.append(BrandViolation(
                    violation_type="brand_missing_from_description",
                    severity="minor",
                    element="meta",
                    selector="meta[name='description']",
                    current_value=meta_desc.get('content', ''),
                    expected_value="Include brand context in meta description",
                    page_url=url,
                    description="Brand context missing from meta description",
                    brand_impact="Low - Missed brand messaging opportunity",
                    seo_impact="Moderate - Brand keywords missing from search snippet",
                    user_impact="Low - Search result may lack brand appeal",
                    remediation_steps=[
                        "Include brand context in meta description",
                        "Ensure description is compelling and brand-consistent",
                        "Optimize for search result click-through"
                    ],
                    automated_fix_available=True,
                    approval_required=False
                ))
        
        return violations
    
    def _check_social_proof_implementation(self, soup: BeautifulSoup, url: str) -> List[BrandViolation]:
        """Check social proof and credibility elements"""
        violations = []
        
        social_proof_config = self.brand_guidelines.get("social_proof", {})
        
        # Check for testimonials on key pages
        if social_proof_config.get("testimonials_required") and self._is_key_page(url):
            testimonial_indicators = soup.find_all(string=re.compile(r'testimonial|review|customer|client', re.I))
            
            if not testimonial_indicators:
                violations.append(BrandViolation(
                    violation_type="missing_testimonials",
                    severity="minor",
                    element="page_content",
                    selector="page",
                    current_value="No testimonials found",
                    expected_value="Include customer testimonials on key pages",
                    page_url=url,
                    description="Customer testimonials missing from key page",
                    brand_impact="Moderate - Missed credibility opportunity",
                    seo_impact="Low - Potential for rich snippets and user engagement",
                    user_impact="Moderate - Reduced social proof and trust signals",
                    remediation_steps=[
                        "Add customer testimonials or reviews",
                        "Include proper structured data for reviews",
                        "Ensure testimonials are authentic and compliant"
                    ],
                    automated_fix_available=False,
                    approval_required=True
                ))
        
        # Check for certifications display
        certifications = social_proof_config.get("certifications", [])
        if certifications:
            cert_found = any(cert.lower() in soup.get_text().lower() for cert in certifications)
            
            if not cert_found and self._is_key_page(url):
                violations.append(BrandViolation(
                    violation_type="missing_certifications",
                    severity="minor",
                    element="page_content",
                    selector="page",
                    current_value="No certifications displayed",
                    expected_value="Display relevant certifications",
                    page_url=url,
                    description="Brand certifications not displayed",
                    brand_impact="Moderate - Missed credibility enhancement",
                    seo_impact="Low - Potential trust signal for search engines",
                    user_impact="Moderate - Reduced trust and authority perception",
                    remediation_steps=[
                        "Add certification badges or mentions",
                        "Ensure certifications are current and valid",
                        "Link to verification pages where appropriate"
                    ],
                    automated_fix_available=False,
                    approval_required=True
                ))
        
        return violations
    
    def _check_brand_keyword_optimization(self, soup: BeautifulSoup, url: str) -> List[BrandViolation]:
        """Check brand keyword optimization and density"""
        violations = []
        
        text_content = soup.get_text().lower()
        brand_keywords = self.brand_guidelines.get("seo_brand_keywords", [])
        
        if brand_keywords:
            total_words = len(text_content.split())
            
            for keyword in brand_keywords:
                keyword_count = text_content.count(keyword.lower())
                keyword_density = (keyword_count / total_words) * 100 if total_words > 0 else 0
                
                # Check if brand keyword density is too low (less than 0.5%)
                if keyword_density < 0.5 and self._is_key_page(url):
                    violations.append(BrandViolation(
                        violation_type="low_brand_keyword_density",
                        severity="minor",
                        element="page_content",
                        selector="page",
                        current_value=f"{keyword}: {keyword_density:.2f}%",
                        expected_value=f"{keyword}: 0.5-2.0% density",
                        page_url=url,
                        description=f"Low brand keyword density for '{keyword}'",
                        brand_impact="Low - Missed brand association opportunity",
                        seo_impact="Moderate - Underoptimized for brand searches",
                        user_impact="Low - Reduced brand context",
                        remediation_steps=[
                            f"Increase natural usage of '{keyword}' in content",
                            "Ensure brand mentions are contextually relevant",
                            "Balance keyword optimization with readability"
                        ],
                        automated_fix_available=False,
                        approval_required=False
                    ))
        
        return violations
    
    def _check_trust_signals(self, soup: BeautifulSoup, url: str) -> List[BrandViolation]:
        """Check for trust signals and security indicators"""
        violations = []
        
        # Check for contact information
        contact_indicators = soup.find_all(string=re.compile(r'contact|phone|email|address', re.I))
        
        if not contact_indicators and self._is_key_page(url):
            violations.append(BrandViolation(
                violation_type="missing_contact_information",
                severity="minor",
                element="page_content",
                selector="page",
                current_value="No contact information found",
                expected_value="Clear contact information display",
                page_url=url,
                description="Contact information not clearly displayed",
                brand_impact="Moderate - Reduced trust and accessibility",
                seo_impact="Low - Local SEO and trust signals",
                user_impact="Moderate - Users cannot easily reach the company",
                remediation_steps=[
                    "Add clear contact information",
                    "Include multiple contact methods",
                    "Ensure contact info is easily accessible"
                ],
                automated_fix_available=False,
                approval_required=False
            ))
        
        return violations
    
    def _is_approved_color(self, color: str) -> bool:
        """Check if color is in approved brand palette"""
        approved_colors = (
            self.brand_guidelines.get("primary_colors", []) +
            self.brand_guidelines.get("secondary_colors", [])
        )
        
        # Convert color formats for comparison (simplified)
        color = color.lower().strip()
        return any(approved.lower() in color or color in approved.lower() for approved in approved_colors)
    
    def _is_approved_font(self, font_family: str) -> bool:
        """Check if font is in approved brand typography"""
        approved_fonts = []
        fonts_config = self.brand_guidelines.get("fonts", {})
        
        for font_category in fonts_config.values():
            if isinstance(font_category, list):
                approved_fonts.extend(font_category)
        
        return any(approved.lower() in font_family.lower() for approved in approved_fonts)
    
    def _is_key_page(self, url: str) -> bool:
        """Determine if URL represents a key brand page"""
        key_page_patterns = [
            'home', 'about', 'contact', 'services', 'products',
            'index', '/$', '/home', '/about-us'
        ]
        
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in key_page_patterns)
    
    def _generate_selector(self, element) -> str:
        """Generate CSS selector for element identification"""
        selectors = []
        
        if element.get('id'):
            return f"#{element.get('id')}"
        
        selectors.append(element.name)
        
        if element.get('class'):
            classes = element.get('class')
            if isinstance(classes, list):
                selectors.append('.' + '.'.join(classes))
            else:
                selectors.append(f".{classes}")
        
        return ''.join(selectors)
    
    def _classify_page_type(self, url: str, soup: BeautifulSoup) -> str:
        """Classify page type for targeted monitoring"""
        url_lower = url.lower()
        
        if any(pattern in url_lower for pattern in ['home', 'index', '/$']):
            return "homepage"
        elif 'product' in url_lower:
            return "product_page"
        elif any(pattern in url_lower for pattern in ['about', 'company', 'team']):
            return "about_page"
        elif 'contact' in url_lower:
            return "contact_page"
        elif any(pattern in url_lower for pattern in ['blog', 'news', 'article']):
            return "content_page"
        else:
            return "standard_page"
    
    def _calculate_brand_metrics(self, url: str, violations: List[BrandViolation], soup: BeautifulSoup) -> BrandMetrics:
        """Calculate comprehensive brand metrics"""
        
        # Count violations by type
        violation_counts = Counter(v.violation_type for v in violations)
        total_violations = len(violations)
        
        # Calculate component scores
        base_score = 100.0
        penalty_weights = {"critical": 25, "major": 15, "minor": 5}
        
        for violation in violations:
            base_score -= penalty_weights.get(violation.severity, 5)
        
        overall_score = max(0.0, base_score)
        
        # Calculate specific component scores
        logo_score = 100.0 - (violation_counts.get("missing_logo", 0) * 30 +
                             violation_counts.get("logo_too_small", 0) * 10)
        
        color_score = 100.0 - (violation_counts.get("unauthorized_color_usage", 0) * 15)
        
        typography_score = 100.0 - (violation_counts.get("unauthorized_font_usage", 0) * 10)
        
        messaging_score = 100.0 - (violation_counts.get("prohibited_term_usage", 0) * 20 +
                                  violation_counts.get("missing_brand_tagline", 0) * 10)
        
        # SEO brand optimization
        seo_brand_score = 100.0 - (violation_counts.get("brand_missing_from_title", 0) * 25 +
                                  violation_counts.get("brand_missing_from_description", 0) * 15)
        
        # Calculate brand keyword metrics
        text_content = soup.get_text().lower()
        brand_keywords = self.brand_guidelines.get("seo_brand_keywords", [])
        
        if brand_keywords and text_content:
            total_words = len(text_content.split())
            total_brand_mentions = sum(text_content.count(keyword.lower()) for keyword in brand_keywords)
            keyword_density = (total_brand_mentions / total_words) * 100 if total_words > 0 else 0
        else:
            keyword_density = 0.0
        
        return BrandMetrics(
            url=url,
            overall_brand_score=overall_score,
            logo_compliance_score=max(0.0, logo_score),
            color_consistency_score=max(0.0, color_score),
            typography_score=max(0.0, typography_score),
            messaging_consistency_score=max(0.0, messaging_score),
            tone_alignment_score=messaging_score,  # Simplified
            meta_brand_optimization=max(0.0, seo_brand_score),
            brand_keyword_density=keyword_density,
            brand_mention_sentiment=75.0,  # Placeholder - would use sentiment analysis
            social_proof_implementation=70.0,  # Placeholder
            trust_signals_score=80.0  # Placeholder
        )
    
    def _generate_monitoring_summary(self, page_results: List[Dict]) -> Dict:
        """Generate monitoring summary across all pages"""
        
        if not page_results:
            return {"error": "No pages successfully monitored"}
        
        total_pages = len(page_results)
        total_violations = sum(len(page.get("violations", [])) for page in page_results)
        avg_brand_score = np.mean([page["metrics"]["overall_brand_score"] for page in page_results])
        
        # Count pages by compliance level
        compliant_pages = len([page for page in page_results if page["metrics"]["overall_brand_score"] >= 80])
        
        # Page type analysis
        page_types = Counter(page.get("page_classification", "unknown") for page in page_results)
        
        return {
            "total_pages_monitored": total_pages,
            "average_brand_score": avg_brand_score,
            "compliant_pages": compliant_pages,
            "non_compliant_pages": total_pages - compliant_pages,
            "total_violations": total_violations,
            "average_violations_per_page": total_violations / total_pages if total_pages > 0 else 0,
            "page_type_breakdown": dict(page_types),
            "monitoring_timestamp": datetime.now().isoformat()
        }
    
    def _generate_brand_insights(self, page_results: List[Dict], violations: List[Dict]) -> List[BrandInsight]:
        """Generate strategic brand insights"""
        
        insights = []
        
        if not page_results:
            return insights
        
        # Most common violation insight
        violation_types = [v["violation_type"] for v in violations]
        if violation_types:
            most_common = Counter(violation_types).most_common(1)[0]
            insights.append(BrandInsight(
                insight_type="common_violation",
                title=f"Most Common Brand Issue: {most_common[0].replace('_', ' ').title()}",
                description=f"Found {most_common[1]} instances across {len(page_results)} pages",
                brand_risk_level="Medium",
                business_impact="Inconsistent brand presentation may reduce customer trust and recognition",
                affected_touchpoints=most_common[1],
                revenue_implications="Potential 5-10% reduction in brand-driven conversions",
                recommended_actions=[
                    f"Implement automated {most_common[0].replace('_', ' ')} compliance checking",
                    "Update brand guidelines and training materials",
                    "Establish approval workflow for brand asset usage"
                ],
                success_metrics=[
                    f"Reduce {most_common[0].replace('_', ' ')} violations by 90%",
                    "Achieve 95% brand compliance across all pages",
                    "Improve brand recognition metrics"
                ],
                implementation_timeline="30-60 days"
            ))
        
        # SEO-brand alignment opportunity
        avg_meta_score = np.mean([page["metrics"]["meta_brand_optimization"] for page in page_results])
        if avg_meta_score < 70:
            insights.append(BrandInsight(
                insight_type="seo_brand_opportunity",
                title="Brand-SEO Alignment Opportunity",
                description=f"Current meta brand optimization score is {avg_meta_score:.1f}% with room for improvement",
                brand_risk_level="Low",
                business_impact="Missing opportunities to reinforce brand in search results",
                affected_touchpoints=len([p for p in page_results if p["metrics"]["meta_brand_optimization"] < 70]),
                revenue_implications="Potential 10-15% increase in branded search click-through rates",
                recommended_actions=[
                    "Optimize title tags and meta descriptions for brand keywords",
                    "Ensure brand messaging consistency in search snippets",
                    "Implement structured data for brand entity recognition"
                ],
                success_metrics=[
                    "Achieve 85%+ meta brand optimization score",
                    "Increase branded search CTR by 15%",
                    "Improve brand search visibility"
                ],
                implementation_timeline="15-30 days"
            ))
        
        return insights
    
    def _calculate_overall_brand_score(self, page_results: List[Dict]) -> float:
        """Calculate overall brand compliance score"""
        
        if not page_results:
            return 0.0
        
        scores = [page["metrics"]["overall_brand_score"] for page in page_results]
        return np.mean(scores)
    
    def _assess_compliance_status(self, brand_score: float, violations: List[Dict]) -> str:
        """Assess overall brand compliance status"""
        
        critical_violations = len([v for v in violations if v.get("severity") == "critical"])
        major_violations = len([v for v in violations if v.get("severity") == "major"])
        
        if critical_violations > 0:
            return "Non-Compliant - Critical Issues"
        elif major_violations > 5:
            return "Non-Compliant - Major Issues"
        elif brand_score >= 85:
            return "Fully Compliant"
        elif brand_score >= 70:
            return "Mostly Compliant"
        else:
            return "Non-Compliant"
    
    def _assess_brand_risks(self, violations: List[Dict]) -> Dict:
        """Assess brand-related business risks"""
        
        risk_factors = []
        risk_level = "Low"
        
        critical_violations = len([v for v in violations if v.get("severity") == "critical"])
        major_violations = len([v for v in violations if v.get("severity") == "major"])
        
        if critical_violations > 0:
            risk_factors.append("Critical brand guideline violations detected")
            risk_level = "High"
        
        if major_violations > 5:
            risk_factors.append("Multiple major brand consistency issues")
            if risk_level != "High":
                risk_level = "Medium"
        
        # Check for specific high-risk violations
        high_risk_types = ["prohibited_term_usage", "missing_logo"]
        high_risk_violations = [v for v in violations if v.get("violation_type") in high_risk_types]
        
        if high_risk_violations:
            risk_factors.append("High-impact brand violations affecting trust and recognition")
            risk_level = "High"
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "total_violations": len(violations),
            "critical_violations": critical_violations,
            "major_violations": major_violations,
            "estimated_business_impact": self._estimate_business_impact(violations)
        }
    
    def _estimate_business_impact(self, violations: List[Dict]) -> str:
        """Estimate business impact of brand violations"""
        
        critical_violations = len([v for v in violations if v.get("severity") == "critical"])
        major_violations = len([v for v in violations if v.get("severity") == "major"])
        
        if critical_violations > 3:
            return "High - Significant brand damage risk, potential legal issues"
        elif major_violations > 10:
            return "Medium - Noticeable brand inconsistency, reduced customer trust"
        elif len(violations) > 20:
            return "Low-Medium - Minor brand presentation issues"
        else:
            return "Low - Minimal business impact expected"
    
    def generate_executive_brand_report(self, monitoring_results: Dict) -> Dict:
        """Generate executive brand monitoring report
        
        Perfect for: CMO reporting, brand manager briefings, marketing performance reviews
        """
        
        summary = monitoring_results.get("monitoring_summary", {})
        brand_score = monitoring_results.get("brand_score", 0)
        violations = monitoring_results.get("violations", [])
        risk_assessment = monitoring_results.get("risk_assessment", {})
        
        return {
            "executive_summary": {
                "brand_compliance_status": monitoring_results.get("compliance_status", "Unknown"),
                "pages_monitored": summary.get("total_pages_monitored", 0),
                "overall_brand_score": f"{brand_score:.1f}%",
                "risk_level": risk_assessment.get("risk_level", "Unknown"),
                "immediate_action_required": risk_assessment.get("risk_level") in ["High", "Critical"]
            },
            "brand_performance_metrics": {
                "compliance_score": f"{brand_score:.1f}%",
                "total_violations": len(violations),
                "critical_issues": risk_assessment.get("critical_violations", 0),
                "major_issues": risk_assessment.get("major_violations", 0),
                "compliant_pages": summary.get("compliant_pages", 0)
            },
            "business_impact_analysis": {
                "brand_risk_assessment": risk_assessment.get("estimated_business_impact", "Unknown"),
                "customer_trust_impact": "Brand inconsistencies may reduce customer confidence by 10-15%",
                "seo_performance_impact": "Poor brand optimization affects branded search performance",
                "competitive_positioning": "Brand compliance critical for market differentiation"
            },
            "strategic_recommendations": {
                "immediate_priorities": [
                    "Address all critical brand guideline violations",
                    "Implement automated brand compliance monitoring",
                    "Establish brand asset approval workflows"
                ],
                "30_day_goals": [
                    "Achieve 90%+ brand compliance across all pages",
                    "Optimize meta tags for brand keyword visibility",
                    "Train content teams on brand guidelines"
                ],
                "long_term_initiatives": [
                    "Integrate brand monitoring with CMS",
                    "Develop brand performance KPI dashboard",
                    "Implement AI-powered brand consistency checking"
                ]
            },
            "investment_requirements": {
                "compliance_remediation": f"¬{len(violations) * 200:,} - ¬{len(violations) * 400:,}",
                "monitoring_automation": "¬25,000 - ¬40,000 for enterprise platform",
                "ongoing_governance": "¬15,000 - ¬30,000 annually",
                "roi_projection": "15-25% improvement in brand-driven conversions within 6 months"
            },
            "portfolio_note": "<¯ Built by technical marketing leader with enterprise brand management expertise",
            "contact_info": "= https://www.linkedin.com/in/sspyrou/ | =€ https://verityai.co"
        }


# Example usage for portfolio demonstration
async def demonstrate_brand_monitoring():
    """Demonstration of brand consistency monitoring for portfolio showcase"""
    
    sample_urls = ['https://example.com', 'https://example.com/about', 'https://example.com/products']
    
    async with BrandConsistencyMonitor() as monitor:
        # Would perform actual brand monitoring in real implementation
        print("<¯ Brand Consistency Monitoring System Ready")
        print(f"=Ê Configured to monitor {len(sample_urls)} URLs")
        print("=€ Enterprise brand governance demonstrated")

if __name__ == "__main__":
    asyncio.run(demonstrate_brand_monitoring())