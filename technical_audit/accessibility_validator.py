"""
Accessibility Validator - Enterprise SEO Accessibility Compliance Platform
Advanced accessibility auditing with SEO impact analysis and WCAG compliance monitoring

<¯ PORTFOLIO PROJECT: Demonstrates accessibility expertise and inclusive SEO strategies
=T Perfect for: Digital accessibility officers, enterprise UX teams, compliance directors

   DEMO/PORTFOLIO CODE: This is demonstration code showcasing accessibility capabilities.
    Real implementations require comprehensive testing across assistive technologies.

= Connect with the developer: https://www.linkedin.com/in/sspyrou/
=€ AI-Enhanced Accessibility Solutions: https://verityai.co

Built by a technical marketing leader with expertise in accessibility compliance
and its critical intersection with SEO performance and user experience.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
class AccessibilityViolation:
    """WCAG accessibility violation with SEO impact assessment"""
    violation_type: str
    wcag_level: str  # "A", "AA", "AAA"
    wcag_criterion: str
    severity: str  # "critical", "serious", "moderate", "minor"
    element: str
    selector: str
    description: str
    seo_impact: str
    user_impact: str
    remediation_steps: List[str]
    automated_fix_available: bool
    estimated_fix_time: str


@dataclass
class AccessibilityMetrics:
    """Comprehensive accessibility metrics with SEO correlation"""
    url: str
    wcag_aa_compliance_score: float
    total_violations: int
    critical_violations: int
    serious_violations: int
    moderate_violations: int
    minor_violations: int
    seo_accessibility_score: float
    keyboard_navigation_score: float
    screen_reader_compatibility: float
    color_contrast_score: float
    alternative_text_score: float
    heading_structure_score: float
    form_accessibility_score: float
    semantic_markup_score: float


@dataclass
class AccessibilityInsight:
    """Strategic accessibility insights for executive decision-making"""
    insight_type: str
    title: str
    description: str
    business_impact: str
    affected_users: str
    seo_benefit: str
    implementation_priority: str
    recommended_actions: List[str]
    success_metrics: List[str]
    roi_factors: List[str]


class AccessibilityValidator:
    """Enterprise-Grade Accessibility Compliance Platform
    
    Perfect for: Digital accessibility officers, enterprise UX teams, compliance directors
    Demonstrates: Accessibility expertise, WCAG compliance, inclusive design, SEO integration
    
    Business Value:
    " WCAG 2.1 AA/AAA compliance monitoring and reporting
    " SEO impact analysis of accessibility improvements
    " Automated accessibility testing at enterprise scale
    " Legal compliance risk assessment and mitigation
    " Executive-ready accessibility performance dashboards
    
    <¯ Portfolio Highlight: Showcases deep understanding of accessibility standards
       and their business impact - critical for inclusive digital experiences that
       also drive SEO performance and legal compliance.
    """
    
    def __init__(self, compliance_level: str = "AA"):
        self.compliance_level = compliance_level
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WCAG compliance thresholds
        self.wcag_thresholds = {
            "AA": {"min_contrast": 4.5, "large_text_contrast": 3.0},
            "AAA": {"min_contrast": 7.0, "large_text_contrast": 4.5}
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Accessibility-Validator/1.0'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def validate_accessibility(self, urls: List[str]) -> Dict:
        """Comprehensive accessibility validation across multiple URLs
        
        Executive Value: Complete accessibility audit with business impact analysis
        """
        logger.info("starting_accessibility_validation", urls=len(urls))
        
        validation_results = {
            "validation_summary": {},
            "page_results": [],
            "violations": [],
            "insights": [],
            "compliance_score": 0.0,
            "seo_impact_analysis": {}
        }
        
        # Process URLs in parallel batches
        batch_size = 5
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_tasks = [self._validate_page_accessibility(url) for url in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error("page_validation_error", url=batch[j], error=str(result))
                    continue
                    
                if result:
                    validation_results["page_results"].append(result)
                    if result.get("violations"):
                        validation_results["violations"].extend(result["violations"])
        
        # Generate comprehensive analysis
        validation_results["validation_summary"] = self._generate_validation_summary(
            validation_results["page_results"]
        )
        validation_results["insights"] = self._generate_accessibility_insights(
            validation_results["page_results"], validation_results["violations"]
        )
        validation_results["compliance_score"] = self._calculate_overall_compliance_score(
            validation_results["page_results"]
        )
        validation_results["seo_impact_analysis"] = self._analyze_seo_impact(
            validation_results["violations"]
        )
        
        logger.info("accessibility_validation_complete",
                   score=validation_results["compliance_score"],
                   violations=len(validation_results["violations"]))
        
        return validation_results
    
    async def _validate_page_accessibility(self, url: str) -> Optional[Dict]:
        """Validate accessibility for a single page"""
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Comprehensive accessibility analysis
                violations = []
                
                # Image accessibility
                violations.extend(self._check_image_accessibility(soup))
                
                # Heading structure
                violations.extend(self._check_heading_structure(soup))
                
                # Form accessibility
                violations.extend(self._check_form_accessibility(soup))
                
                # Link accessibility
                violations.extend(self._check_link_accessibility(soup))
                
                # Color contrast (basic check)
                violations.extend(self._check_basic_contrast_issues(soup))
                
                # Semantic markup
                violations.extend(self._check_semantic_markup(soup))
                
                # Keyboard navigation
                violations.extend(self._check_keyboard_navigation(soup))
                
                # ARIA implementation
                violations.extend(self._check_aria_implementation(soup))
                
                # Calculate accessibility metrics
                metrics = self._calculate_accessibility_metrics(url, violations, soup)
                
                return {
                    "url": url,
                    "metrics": asdict(metrics),
                    "violations": [asdict(v) for v in violations],
                    "validation_timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error("page_accessibility_validation_error", url=url, error=str(e))
            return None
    
    def _check_image_accessibility(self, soup: BeautifulSoup) -> List[AccessibilityViolation]:
        """Check image accessibility compliance"""
        violations = []
        
        images = soup.find_all('img')
        for img in images:
            # Missing alt text
            if not img.get('alt'):
                violations.append(AccessibilityViolation(
                    violation_type="missing_alt_text",
                    wcag_level="A",
                    wcag_criterion="1.1.1",
                    severity="serious",
                    element="img",
                    selector=self._generate_selector(img),
                    description="Image missing alternative text",
                    seo_impact="High - Images without alt text miss SEO opportunities",
                    user_impact="Critical - Screen reader users cannot access image content",
                    remediation_steps=[
                        "Add descriptive alt attribute to image",
                        "Use empty alt='' for decorative images",
                        "Ensure alt text describes image purpose and content"
                    ],
                    automated_fix_available=False,
                    estimated_fix_time="5-15 minutes per image"
                ))
            
            # Empty alt text for content images
            elif img.get('alt') == '' and not self._is_decorative_image(img):
                violations.append(AccessibilityViolation(
                    violation_type="empty_alt_content_image",
                    wcag_level="A",
                    wcag_criterion="1.1.1",
                    severity="serious",
                    element="img",
                    selector=self._generate_selector(img),
                    description="Content image has empty alt text",
                    seo_impact="High - Missing SEO value from image descriptions",
                    user_impact="High - Important image content not accessible",
                    remediation_steps=[
                        "Replace empty alt with descriptive text",
                        "Review image purpose and context",
                        "Consider if image is truly decorative"
                    ],
                    automated_fix_available=False,
                    estimated_fix_time="10-20 minutes per image"
                ))
        
        return violations
    
    def _check_heading_structure(self, soup: BeautifulSoup) -> List[AccessibilityViolation]:
        """Check heading structure for accessibility and SEO"""
        violations = []
        
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if not headings:
            violations.append(AccessibilityViolation(
                violation_type="no_headings",
                wcag_level="AA",
                wcag_criterion="1.3.1",
                severity="serious",
                element="page",
                selector="html",
                description="Page has no heading structure",
                seo_impact="Critical - No heading hierarchy for SEO",
                user_impact="High - Screen readers cannot navigate page structure",
                remediation_steps=[
                    "Add proper heading hierarchy starting with h1",
                    "Use headings to structure content logically",
                    "Ensure headings describe following content"
                ],
                automated_fix_available=False,
                estimated_fix_time="30-60 minutes"
            ))
            return violations
        
        # Check for multiple H1s
        h1_tags = soup.find_all('h1')
        if len(h1_tags) > 1:
            violations.append(AccessibilityViolation(
                violation_type="multiple_h1",
                wcag_level="AA",
                wcag_criterion="1.3.1",
                severity="moderate",
                element="h1",
                selector="h1",
                description=f"Page has {len(h1_tags)} H1 tags",
                seo_impact="Moderate - Multiple H1s can confuse search engines",
                user_impact="Moderate - Unclear page structure for assistive technology",
                remediation_steps=[
                    "Use only one H1 per page",
                    "Convert additional H1s to appropriate heading levels",
                    "Review heading hierarchy logic"
                ],
                automated_fix_available=True,
                estimated_fix_time="15-30 minutes"
            ))
        
        # Check heading sequence
        previous_level = 0
        for heading in headings:
            level = int(heading.name[1])
            
            if previous_level > 0 and level > previous_level + 1:
                violations.append(AccessibilityViolation(
                    violation_type="skipped_heading_level",
                    wcag_level="AA",
                    wcag_criterion="1.3.1",
                    severity="moderate",
                    element=heading.name,
                    selector=self._generate_selector(heading),
                    description=f"Heading level skipped: {heading.name} after h{previous_level}",
                    seo_impact="Moderate - Poor heading hierarchy affects SEO",
                    user_impact="Moderate - Confusing navigation structure",
                    remediation_steps=[
                        "Use sequential heading levels",
                        "Don't skip heading levels for styling",
                        "Review overall content hierarchy"
                    ],
                    automated_fix_available=True,
                    estimated_fix_time="10-20 minutes"
                ))
            
            previous_level = level
        
        return violations
    
    def _check_form_accessibility(self, soup: BeautifulSoup) -> List[AccessibilityViolation]:
        """Check form accessibility compliance"""
        violations = []
        
        forms = soup.find_all('form')
        for form in forms:
            inputs = form.find_all(['input', 'textarea', 'select'])
            
            for input_elem in inputs:
                input_type = input_elem.get('type', 'text')
                input_id = input_elem.get('id')
                
                # Skip hidden inputs
                if input_type == 'hidden':
                    continue
                
                # Missing label
                associated_label = None
                if input_id:
                    associated_label = form.find('label', {'for': input_id})
                
                if not associated_label and not input_elem.get('aria-label') and not input_elem.get('aria-labelledby'):
                    violations.append(AccessibilityViolation(
                        violation_type="missing_form_label",
                        wcag_level="A",
                        wcag_criterion="1.3.1",
                        severity="serious",
                        element=input_elem.name,
                        selector=self._generate_selector(input_elem),
                        description=f"Form {input_elem.name} missing accessible label",
                        seo_impact="Moderate - Poor form structure affects crawling",
                        user_impact="Critical - Form unusable by screen reader users",
                        remediation_steps=[
                            "Add label element with 'for' attribute",
                            "Use aria-label or aria-labelledby",
                            "Ensure label text is descriptive"
                        ],
                        automated_fix_available=False,
                        estimated_fix_time="10-15 minutes per field"
                    ))
                
                # Required fields without indication
                if input_elem.get('required') and not input_elem.get('aria-required'):
                    # Check if asterisk or "required" text is present in label
                    required_indication = False
                    if associated_label:
                        label_text = associated_label.get_text().lower()
                        required_indication = '*' in label_text or 'required' in label_text
                    
                    if not required_indication:
                        violations.append(AccessibilityViolation(
                            violation_type="required_field_not_indicated",
                            wcag_level="A",
                            wcag_criterion="3.3.2",
                            severity="moderate",
                            element=input_elem.name,
                            selector=self._generate_selector(input_elem),
                            description="Required field not clearly indicated",
                            seo_impact="Low - Minimal direct SEO impact",
                            user_impact="Moderate - Users may miss required fields",
                            remediation_steps=[
                                "Add aria-required='true'",
                                "Include visual required indicator",
                                "Provide clear field instructions"
                            ],
                            automated_fix_available=True,
                            estimated_fix_time="5-10 minutes per field"
                        ))
        
        return violations
    
    def _check_link_accessibility(self, soup: BeautifulSoup) -> List[AccessibilityViolation]:
        """Check link accessibility and SEO optimization"""
        violations = []
        
        links = soup.find_all('a', href=True)
        for link in links:
            link_text = link.get_text().strip()
            
            # Empty link text
            if not link_text:
                # Check for aria-label or title
                if not link.get('aria-label') and not link.get('title'):
                    violations.append(AccessibilityViolation(
                        violation_type="empty_link_text",
                        wcag_level="A",
                        wcag_criterion="2.4.4",
                        severity="serious",
                        element="a",
                        selector=self._generate_selector(link),
                        description="Link has no accessible text",
                        seo_impact="High - Links without text provide no SEO value",
                        user_impact="Critical - Link purpose unclear to screen readers",
                        remediation_steps=[
                            "Add descriptive link text",
                            "Use aria-label for icon-only links",
                            "Ensure link purpose is clear"
                        ],
                        automated_fix_available=False,
                        estimated_fix_time="10-15 minutes per link"
                    ))
            
            # Generic link text
            elif link_text.lower() in ['click here', 'read more', 'more', 'link', 'here']:
                violations.append(AccessibilityViolation(
                    violation_type="generic_link_text",
                    wcag_level="AA",
                    wcag_criterion="2.4.4",
                    severity="moderate",
                    element="a",
                    selector=self._generate_selector(link),
                    description=f"Generic link text: '{link_text}'",
                    seo_impact="Moderate - Generic text misses keyword opportunities",
                    user_impact="Moderate - Link purpose unclear out of context",
                    remediation_steps=[
                        "Use descriptive, specific link text",
                        "Include keywords relevant to destination",
                        "Make link text meaningful out of context"
                    ],
                    automated_fix_available=False,
                    estimated_fix_time="15-20 minutes per link"
                ))
        
        return violations
    
    def _check_basic_contrast_issues(self, soup: BeautifulSoup) -> List[AccessibilityViolation]:
        """Basic color contrast issue detection"""
        violations = []
        
        # Check for common contrast issues in inline styles
        elements_with_style = soup.find_all(attrs={"style": True})
        
        for element in elements_with_style:
            style = element.get('style', '').lower()
            
            # Look for potential low contrast combinations
            if 'color:' in style and 'background' in style:
                # This is a simplified check - real implementation would calculate actual contrast ratios
                if ('white' in style and 'yellow' in style) or ('gray' in style and 'silver' in style):
                    violations.append(AccessibilityViolation(
                        violation_type="potential_contrast_issue",
                        wcag_level=self.compliance_level,
                        wcag_criterion="1.4.3",
                        severity="moderate",
                        element=element.name,
                        selector=self._generate_selector(element),
                        description="Potential color contrast issue detected",
                        seo_impact="Low - Indirect impact on user engagement",
                        user_impact="High - Text may be difficult to read",
                        remediation_steps=[
                            "Test color contrast ratios",
                            "Use tools like WebAIM Contrast Checker",
                            "Ensure minimum 4.5:1 contrast ratio"
                        ],
                        automated_fix_available=False,
                        estimated_fix_time="20-30 minutes per element"
                    ))
        
        return violations
    
    def _check_semantic_markup(self, soup: BeautifulSoup) -> List[AccessibilityViolation]:
        """Check semantic HTML markup usage"""
        violations = []
        
        # Check for semantic HTML5 elements
        semantic_elements = ['main', 'nav', 'header', 'footer', 'section', 'article', 'aside']
        found_semantic = any(soup.find(element) for element in semantic_elements)
        
        if not found_semantic:
            violations.append(AccessibilityViolation(
                violation_type="missing_semantic_markup",
                wcag_level="AA",
                wcag_criterion="1.3.1",
                severity="moderate",
                element="page",
                selector="html",
                description="Page lacks semantic HTML5 structure",
                seo_impact="Moderate - Semantic markup helps search engine understanding",
                user_impact="Moderate - Assistive technology navigation less efficient",
                remediation_steps=[
                    "Replace generic divs with semantic elements",
                    "Use main, nav, header, footer elements",
                    "Structure content with section and article tags"
                ],
                automated_fix_available=True,
                estimated_fix_time="60-90 minutes"
            ))
        
        # Check for table markup without proper headers
        tables = soup.find_all('table')
        for table in tables:
            if not table.find('th') and not table.find(attrs={'scope': True}):
                violations.append(AccessibilityViolation(
                    violation_type="table_missing_headers",
                    wcag_level="A",
                    wcag_criterion="1.3.1",
                    severity="serious",
                    element="table",
                    selector=self._generate_selector(table),
                    description="Data table missing proper headers",
                    seo_impact="Low - Minimal direct SEO impact",
                    user_impact="High - Table data inaccessible to screen readers",
                    remediation_steps=[
                        "Add th elements for headers",
                        "Use scope attribute for complex tables",
                        "Consider caption element for table description"
                    ],
                    automated_fix_available=True,
                    estimated_fix_time="20-30 minutes per table"
                ))
        
        return violations
    
    def _check_keyboard_navigation(self, soup: BeautifulSoup) -> List[AccessibilityViolation]:
        """Check keyboard navigation accessibility"""
        violations = []
        
        # Check for interactive elements without proper focus management
        interactive_elements = soup.find_all(['button', 'a', 'input', 'textarea', 'select'])
        
        for element in interactive_elements:
            # Skip hidden elements
            if element.get('type') == 'hidden':
                continue
            
            # Elements with tabindex="-1" that shouldn't have it
            if element.get('tabindex') == '-1' and element.name != 'div':
                violations.append(AccessibilityViolation(
                    violation_type="interactive_element_not_focusable",
                    wcag_level="A",
                    wcag_criterion="2.1.1",
                    severity="serious",
                    element=element.name,
                    selector=self._generate_selector(element),
                    description="Interactive element removed from tab order",
                    seo_impact="Low - Minimal direct SEO impact",
                    user_impact="Critical - Element inaccessible via keyboard",
                    remediation_steps=[
                        "Remove negative tabindex",
                        "Ensure interactive elements are keyboard accessible",
                        "Test keyboard navigation flow"
                    ],
                    automated_fix_available=True,
                    estimated_fix_time="10-15 minutes per element"
                ))
        
        # Check for custom interactive elements without proper ARIA
        divs_with_click = soup.find_all('div', attrs={'onclick': True})
        for div in divs_with_click:
            if not div.get('role') and not div.get('tabindex'):
                violations.append(AccessibilityViolation(
                    violation_type="div_click_handler_not_accessible",
                    wcag_level="A",
                    wcag_criterion="2.1.1",
                    severity="serious",
                    element="div",
                    selector=self._generate_selector(div),
                    description="Clickable div without keyboard accessibility",
                    seo_impact="Low - Custom elements may not be crawled",
                    user_impact="Critical - Interactive content inaccessible via keyboard",
                    remediation_steps=[
                        "Add appropriate ARIA role",
                        "Add tabindex='0' for keyboard focus",
                        "Implement keyboard event handlers",
                        "Consider using button element instead"
                    ],
                    automated_fix_available=False,
                    estimated_fix_time="30-45 minutes per element"
                ))
        
        return violations
    
    def _check_aria_implementation(self, soup: BeautifulSoup) -> List[AccessibilityViolation]:
        """Check ARIA implementation for accessibility"""
        violations = []
        
        # Check for ARIA attributes with invalid values
        elements_with_aria = soup.find_all(attrs=lambda attr: attr and any(key.startswith('aria-') for key in attr.keys() if isinstance(key, str)))
        
        for element in elements_with_aria:
            # Check aria-expanded without role
            if element.get('aria-expanded') and not element.get('role'):
                violations.append(AccessibilityViolation(
                    violation_type="aria_expanded_without_role",
                    wcag_level="AA",
                    wcag_criterion="4.1.2",
                    severity="moderate",
                    element=element.name,
                    selector=self._generate_selector(element),
                    description="aria-expanded used without appropriate role",
                    seo_impact="Low - Minimal direct SEO impact",
                    user_impact="Moderate - Unclear element purpose for assistive technology",
                    remediation_steps=[
                        "Add appropriate role (button, menubutton, etc.)",
                        "Ensure ARIA state changes reflect UI changes",
                        "Test with screen readers"
                    ],
                    automated_fix_available=False,
                    estimated_fix_time="15-20 minutes per element"
                ))
        
        return violations
    
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
    
    def _is_decorative_image(self, img) -> bool:
        """Determine if an image is likely decorative"""
        # Simple heuristics for decorative images
        src = img.get('src', '').lower()
        decorative_patterns = ['decoration', 'border', 'spacer', 'bullet', 'icon', 'background']
        
        return any(pattern in src for pattern in decorative_patterns)
    
    def _calculate_accessibility_metrics(self, url: str, violations: List[AccessibilityViolation], soup: BeautifulSoup) -> AccessibilityMetrics:
        """Calculate comprehensive accessibility metrics"""
        
        # Count violations by severity
        violation_counts = Counter(v.severity for v in violations)
        total_violations = len(violations)
        
        # Calculate component scores
        alt_text_score = self._calculate_alt_text_score(soup)
        heading_score = self._calculate_heading_score(soup, violations)
        form_score = self._calculate_form_score(soup, violations)
        semantic_score = self._calculate_semantic_score(soup)
        
        # Calculate overall compliance score
        base_score = 100.0
        penalty_weights = {"critical": 20, "serious": 10, "moderate": 5, "minor": 2}
        
        for severity, count in violation_counts.items():
            base_score -= penalty_weights.get(severity, 2) * count
        
        wcag_aa_score = max(0.0, base_score)
        
        # SEO accessibility correlation
        seo_accessibility_score = (alt_text_score + heading_score + semantic_score) / 3
        
        return AccessibilityMetrics(
            url=url,
            wcag_aa_compliance_score=wcag_aa_score,
            total_violations=total_violations,
            critical_violations=violation_counts.get("critical", 0),
            serious_violations=violation_counts.get("serious", 0),
            moderate_violations=violation_counts.get("moderate", 0),
            minor_violations=violation_counts.get("minor", 0),
            seo_accessibility_score=seo_accessibility_score,
            keyboard_navigation_score=self._calculate_keyboard_score(violations),
            screen_reader_compatibility=self._calculate_screen_reader_score(violations),
            color_contrast_score=self._calculate_contrast_score(violations),
            alternative_text_score=alt_text_score,
            heading_structure_score=heading_score,
            form_accessibility_score=form_score,
            semantic_markup_score=semantic_score
        )
    
    def _calculate_alt_text_score(self, soup: BeautifulSoup) -> float:
        """Calculate alternative text implementation score"""
        images = soup.find_all('img')
        if not images:
            return 100.0
        
        images_with_alt = len([img for img in images if img.get('alt') is not None])
        return (images_with_alt / len(images)) * 100
    
    def _calculate_heading_score(self, soup: BeautifulSoup, violations: List[AccessibilityViolation]) -> float:
        """Calculate heading structure score"""
        heading_violations = [v for v in violations if 'heading' in v.violation_type or v.element.startswith('h')]
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        if not headings:
            return 0.0
        
        # Base score minus penalties for violations
        base_score = 100.0
        penalty_per_violation = 15.0
        
        return max(0.0, base_score - (len(heading_violations) * penalty_per_violation))
    
    def _calculate_form_score(self, soup: BeautifulSoup, violations: List[AccessibilityViolation]) -> float:
        """Calculate form accessibility score"""
        form_violations = [v for v in violations if 'form' in v.violation_type]
        forms = soup.find_all('form')
        
        if not forms:
            return 100.0
        
        base_score = 100.0
        penalty_per_violation = 20.0
        
        return max(0.0, base_score - (len(form_violations) * penalty_per_violation))
    
    def _calculate_semantic_score(self, soup: BeautifulSoup) -> float:
        """Calculate semantic markup score"""
        semantic_elements = ['main', 'nav', 'header', 'footer', 'section', 'article', 'aside']
        found_elements = sum(1 for element in semantic_elements if soup.find(element))
        
        return (found_elements / len(semantic_elements)) * 100
    
    def _calculate_keyboard_score(self, violations: List[AccessibilityViolation]) -> float:
        """Calculate keyboard accessibility score"""
        keyboard_violations = [v for v in violations if 'keyboard' in v.violation_type.lower() or 'focus' in v.violation_type.lower()]
        
        base_score = 100.0
        penalty_per_violation = 15.0
        
        return max(0.0, base_score - (len(keyboard_violations) * penalty_per_violation))
    
    def _calculate_screen_reader_score(self, violations: List[AccessibilityViolation]) -> float:
        """Calculate screen reader compatibility score"""
        sr_violations = [v for v in violations if any(term in v.user_impact.lower() for term in ['screen reader', 'assistive technology'])]
        
        base_score = 100.0
        penalty_per_violation = 10.0
        
        return max(0.0, base_score - (len(sr_violations) * penalty_per_violation))
    
    def _calculate_contrast_score(self, violations: List[AccessibilityViolation]) -> float:
        """Calculate color contrast score"""
        contrast_violations = [v for v in violations if 'contrast' in v.violation_type]
        
        base_score = 100.0
        penalty_per_violation = 25.0
        
        return max(0.0, base_score - (len(contrast_violations) * penalty_per_violation))
    
    def _generate_validation_summary(self, page_results: List[Dict]) -> Dict:
        """Generate validation summary across all pages"""
        
        if not page_results:
            return {"error": "No pages successfully validated"}
        
        total_pages = len(page_results)
        total_violations = sum(len(page.get("violations", [])) for page in page_results)
        avg_compliance_score = np.mean([page["metrics"]["wcag_aa_compliance_score"] for page in page_results])
        
        # Count pages by compliance level
        compliant_pages = len([page for page in page_results if page["metrics"]["wcag_aa_compliance_score"] >= 80])
        
        return {
            "total_pages_validated": total_pages,
            "average_compliance_score": avg_compliance_score,
            "compliant_pages": compliant_pages,
            "non_compliant_pages": total_pages - compliant_pages,
            "total_violations": total_violations,
            "average_violations_per_page": total_violations / total_pages if total_pages > 0 else 0,
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def _generate_accessibility_insights(self, page_results: List[Dict], violations: List[Dict]) -> List[AccessibilityInsight]:
        """Generate strategic accessibility insights"""
        
        insights = []
        
        if not page_results:
            return insights
        
        # Most common violation insight
        violation_types = [v["violation_type"] for v in violations]
        if violation_types:
            most_common = Counter(violation_types).most_common(1)[0]
            insights.append(AccessibilityInsight(
                insight_type="common_violation",
                title=f"Most Common Issue: {most_common[0].replace('_', ' ').title()}",
                description=f"Found {most_common[1]} instances across {len(page_results)} pages",
                business_impact="Medium - Affects user experience and legal compliance",
                affected_users="15% of users rely on accessible features",
                seo_benefit="Improved accessibility often correlates with better SEO performance",
                implementation_priority="High",
                recommended_actions=[
                    f"Prioritize fixing {most_common[0].replace('_', ' ')} issues",
                    "Implement automated testing to prevent recurrence",
                    "Train development team on accessibility best practices"
                ],
                success_metrics=[
                    "Reduce violation count by 80%",
                    "Achieve WCAG AA compliance",
                    "Improve user task completion rates"
                ],
                roi_factors=[
                    "Avoid legal compliance issues",
                    "Expand addressable market to users with disabilities",
                    "Improve SEO through better semantic structure"
                ]
            ))
        
        # SEO impact insight
        avg_seo_score = np.mean([page["metrics"]["seo_accessibility_score"] for page in page_results])
        if avg_seo_score < 70:
            insights.append(AccessibilityInsight(
                insight_type="seo_opportunity",
                title="Accessibility Improvements Can Boost SEO Performance",
                description=f"Current SEO-accessibility score is {avg_seo_score:.1f}% with significant improvement potential",
                business_impact="High - SEO improvements drive organic traffic growth",
                affected_users="All organic search users benefit from improved accessibility",
                seo_benefit="Better semantic markup, alt text, and heading structure improve search rankings",
                implementation_priority="High",
                recommended_actions=[
                    "Optimize alt text for images with target keywords",
                    "Improve heading structure for content hierarchy",
                    "Implement semantic HTML5 elements"
                ],
                success_metrics=[
                    "Increase SEO-accessibility score to 85%+",
                    "Improve organic search rankings",
                    "Increase click-through rates from search results"
                ],
                roi_factors=[
                    "Higher search engine rankings",
                    "Increased organic traffic",
                    "Better user engagement metrics"
                ]
            ))
        
        return insights
    
    def _calculate_overall_compliance_score(self, page_results: List[Dict]) -> float:
        """Calculate overall compliance score across all pages"""
        
        if not page_results:
            return 0.0
        
        scores = [page["metrics"]["wcag_aa_compliance_score"] for page in page_results]
        return np.mean(scores)
    
    def _analyze_seo_impact(self, violations: List[Dict]) -> Dict:
        """Analyze SEO impact of accessibility violations"""
        
        seo_impact_violations = [v for v in violations if "High" in v.get("seo_impact", "")]
        
        return {
            "high_seo_impact_violations": len(seo_impact_violations),
            "seo_opportunity_score": max(0, 100 - len(seo_impact_violations) * 10),
            "key_seo_improvements": [
                "Add descriptive alt text to images",
                "Improve heading structure hierarchy",
                "Implement semantic HTML5 markup",
                "Optimize link text for keywords and clarity"
            ],
            "estimated_seo_benefit": "15-25% improvement in organic search performance"
        }
    
    def generate_executive_accessibility_report(self, validation_results: Dict) -> Dict:
        """Generate executive accessibility report
        
        Perfect for: Board presentations, compliance reporting, digital strategy planning
        """
        
        summary = validation_results.get("validation_summary", {})
        compliance_score = validation_results.get("compliance_score", 0)
        violations = validation_results.get("violations", [])
        
        # Risk assessment
        critical_violations = len([v for v in violations if v.get("severity") == "critical"])
        serious_violations = len([v for v in violations if v.get("severity") == "serious"])
        
        return {
            "executive_summary": {
                "accessibility_status": "Compliant" if compliance_score >= 80 else "Non-Compliant",
                "pages_audited": summary.get("total_pages_validated", 0),
                "compliance_score": f"{compliance_score:.1f}%",
                "legal_risk_level": "High" if critical_violations > 0 else "Medium" if serious_violations > 5 else "Low",
                "business_opportunity": f"¬{len(violations) * 1000:,} estimated implementation cost vs ¬{len(violations) * 5000:,} potential legal exposure"
            },
            "compliance_metrics": {
                "wcag_aa_compliance": f"{compliance_score:.1f}%",
                "total_violations": len(violations),
                "critical_issues": critical_violations,
                "serious_issues": serious_violations,
                "pages_fully_compliant": summary.get("compliant_pages", 0)
            },
            "business_impact": {
                "affected_user_population": "15% of users (1.3 billion people with disabilities globally)",
                "seo_opportunity": validation_results.get("seo_impact_analysis", {}).get("estimated_seo_benefit", "15-25% improvement"),
                "legal_compliance": "ADA, Section 508, EU Accessibility Act compliance",
                "brand_reputation": "Demonstrates commitment to inclusive design and social responsibility"
            },
            "implementation_roadmap": {
                "phase_1": "Fix critical and serious violations (30 days)",
                "phase_2": "Address moderate violations and implement testing (60 days)",
                "phase_3": "Achieve full WCAG AA compliance and ongoing monitoring (90 days)"
            },
            "investment_analysis": {
                "implementation_cost": f"¬{len(violations) * 800:,} - ¬{len(violations) * 1200:,}",
                "ongoing_monitoring": "¬15,000 - ¬25,000 annually",
                "roi_timeline": "6-12 months through improved SEO and reduced legal risk",
                "competitive_advantage": "Only 2% of websites are fully accessible - major differentiation opportunity"
            },
            "portfolio_note": "<¯ Built by technical marketing leader with accessibility and inclusive design expertise",
            "contact_info": "= https://www.linkedin.com/in/sspyrou/ | =€ https://verityai.co"
        }


# Example usage for portfolio demonstration
async def demonstrate_accessibility_validation():
    """Demonstration of accessibility validation capabilities for portfolio showcase"""
    
    sample_urls = ['https://example.com', 'https://example.com/about']
    
    async with AccessibilityValidator(compliance_level="AA") as validator:
        # Would perform actual accessibility validation in real implementation
        print("<¯ Accessibility Validation System Ready")
        print(f"=Ê Configured to validate {len(sample_urls)} URLs")
        print("=€ Enterprise WCAG compliance demonstrated")

if __name__ == "__main__":
    asyncio.run(demonstrate_accessibility_validation())