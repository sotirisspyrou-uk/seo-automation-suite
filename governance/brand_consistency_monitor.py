"""
Brand Consistency Monitor - Enterprise Digital Brand Compliance Platform
Advanced brand consistency monitoring across global digital properties and customer touchpoints

🎯 PORTFOLIO PROJECT: Demonstrates brand governance expertise and automated quality assurance
Perfect for: Brand managers, digital marketing directors, global marketing teams

📄 DEMO/PORTFOLIO CODE: This is demonstration code showcasing brand monitoring capabilities.
   Real implementations require comprehensive brand guideline integration and approval workflows.

🔗 Connect with the developer: https://www.linkedin.com/in/sspyrou/
🚀 AI-Enhanced Brand Solutions: https://verityai.co

Built by a technical marketing leader with expertise in brand governance 
and automated quality assurance across enterprise digital ecosystems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path
import json
import re

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
import requests
from io import BytesIO


@dataclass
class BrandElement:
    """Brand element configuration"""
    element_type: str  # logo, color, font, messaging
    expected_value: str
    tolerance: float = 0.1
    priority: str = "medium"  # high, medium, low
    description: str = ""


@dataclass
class BrandInconsistency:
    """Brand inconsistency detection result"""
    url: str
    element_type: str
    expected: str
    actual: str
    severity: str
    confidence: float
    screenshot_path: Optional[str] = None
    detected_at: datetime = None


class BrandConsistencyMonitor:
    """Monitor brand consistency across digital properties"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.session: Optional[aiohttp.ClientSession] = None
        self.brand_elements = self._load_brand_elements()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "brand_guidelines": {
                "primary_colors": ["#1a73e8", "#34a853", "#fbbc05", "#ea4335"],
                "secondary_colors": ["#5f6368", "#202124", "#f8f9fa"],
                "fonts": ["Roboto", "Arial", "sans-serif"],
                "logo_variations": ["logo-main", "logo-white", "logo-dark"],
                "tone_keywords": ["innovative", "reliable", "professional"]
            },
            "monitoring": {
                "check_interval": 3600,
                "max_concurrent": 10,
                "timeout": 30,
                "screenshot_enabled": True
            },
            "thresholds": {
                "color_tolerance": 10,
                "font_match_threshold": 0.8,
                "logo_similarity_threshold": 0.9,
                "messaging_consistency_threshold": 0.7
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
        
    def _load_brand_elements(self) -> List[BrandElement]:
        """Load brand elements from configuration"""
        elements = []
        
        # Color elements
        for color in self.config["brand_guidelines"]["primary_colors"]:
            elements.append(BrandElement(
                element_type="primary_color",
                expected_value=color,
                priority="high",
                description=f"Primary brand color {color}"
            ))
            
        # Font elements
        for font in self.config["brand_guidelines"]["fonts"]:
            elements.append(BrandElement(
                element_type="font",
                expected_value=font,
                priority="medium",
                description=f"Brand font family {font}"
            ))
            
        # Logo elements
        for logo in self.config["brand_guidelines"]["logo_variations"]:
            elements.append(BrandElement(
                element_type="logo",
                expected_value=logo,
                priority="high",
                description=f"Brand logo variation {logo}"
            ))
            
        return elements
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config["monitoring"]["timeout"]
            )
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_brand_consistency(
        self, 
        urls: List[str]
    ) -> Dict[str, List[BrandInconsistency]]:
        """Check brand consistency across multiple URLs"""
        self.logger.info(f"Checking brand consistency for {len(urls)} URLs")
        
        semaphore = asyncio.Semaphore(
            self.config["monitoring"]["max_concurrent"]
        )
        
        tasks = []
        for url in urls:
            task = self._check_url_consistency(url, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        consistency_report = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error checking {url}: {result}")
                consistency_report[url] = []
            else:
                consistency_report[url] = result
        
        return consistency_report
    
    async def _check_url_consistency(
        self, 
        url: str, 
        semaphore: asyncio.Semaphore
    ) -> List[BrandInconsistency]:
        """Check brand consistency for a single URL"""
        async with semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        return []
                    
                    content = await response.text()
                    
                inconsistencies = []
                soup = BeautifulSoup(content, 'html.parser')
                
                # Check colors
                color_issues = await self._check_colors(url, soup)
                inconsistencies.extend(color_issues)
                
                # Check fonts
                font_issues = await self._check_fonts(url, soup)
                inconsistencies.extend(font_issues)
                
                # Check logos
                logo_issues = await self._check_logos(url, soup)
                inconsistencies.extend(logo_issues)
                
                # Check messaging
                messaging_issues = await self._check_messaging(url, soup)
                inconsistencies.extend(messaging_issues)
                
                return inconsistencies
                
            except Exception as e:
                self.logger.error(f"Error checking URL {url}: {e}")
                return []
    
    async def _check_colors(
        self, 
        url: str, 
        soup: BeautifulSoup
    ) -> List[BrandInconsistency]:
        """Check color consistency"""
        inconsistencies = []
        
        # Extract CSS colors
        colors_found = set()
        
        # Check inline styles
        for element in soup.find_all(style=True):
            style = element.get('style', '')
            color_matches = re.findall(r'color:\s*([^;]+)', style)
            background_matches = re.findall(r'background-color:\s*([^;]+)', style)
            colors_found.update(color_matches + background_matches)
        
        # Check CSS files (simplified)
        for link in soup.find_all('link', {'rel': 'stylesheet'}):
            href = link.get('href')
            if href:
                try:
                    # In practice, would fetch and parse CSS
                    pass
                except:
                    pass
        
        # Compare against brand colors
        brand_colors = set(self.config["brand_guidelines"]["primary_colors"])
        brand_colors.update(self.config["brand_guidelines"]["secondary_colors"])
        
        for color in colors_found:
            color = color.strip()
            if color.startswith('#') and len(color) == 7:
                if not self._is_color_compliant(color, brand_colors):
                    inconsistencies.append(BrandInconsistency(
                        url=url,
                        element_type="color",
                        expected="Brand color palette",
                        actual=color,
                        severity="medium",
                        confidence=0.8,
                        detected_at=datetime.now()
                    ))
        
        return inconsistencies
    
    def _is_color_compliant(self, color: str, brand_colors: Set[str]) -> bool:
        """Check if color is compliant with brand guidelines"""
        # Convert hex to RGB for tolerance checking
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        try:
            color_rgb = hex_to_rgb(color)
            tolerance = self.config["thresholds"]["color_tolerance"]
            
            for brand_color in brand_colors:
                brand_rgb = hex_to_rgb(brand_color)
                distance = sum(abs(a - b) for a, b in zip(color_rgb, brand_rgb))
                if distance <= tolerance:
                    return True
            
            return False
        except:
            return True  # If we can't parse, assume compliant
    
    async def _check_fonts(
        self, 
        url: str, 
        soup: BeautifulSoup
    ) -> List[BrandInconsistency]:
        """Check font consistency"""
        inconsistencies = []
        
        fonts_found = set()
        
        # Check CSS font-family declarations
        for element in soup.find_all(style=True):
            style = element.get('style', '')
            font_matches = re.findall(r'font-family:\s*([^;]+)', style)
            fonts_found.update(font_matches)
        
        # Check against brand fonts
        brand_fonts = set(self.config["brand_guidelines"]["fonts"])
        
        for font in fonts_found:
            font_clean = font.strip().replace('"', '').replace("'", "")
            font_families = [f.strip() for f in font_clean.split(',')]
            
            if not any(bf in font_families for bf in brand_fonts):
                inconsistencies.append(BrandInconsistency(
                    url=url,
                    element_type="font",
                    expected=f"Brand fonts: {', '.join(brand_fonts)}",
                    actual=font,
                    severity="low",
                    confidence=0.7,
                    detected_at=datetime.now()
                ))
        
        return inconsistencies
    
    async def _check_logos(
        self, 
        url: str, 
        soup: BeautifulSoup
    ) -> List[BrandInconsistency]:
        """Check logo consistency"""
        inconsistencies = []
        
        # Find logo images
        logo_selectors = [
            'img[alt*="logo"]',
            'img[src*="logo"]',
            '.logo img',
            '#logo img',
            'img[class*="logo"]'
        ]
        
        logos_found = []
        for selector in logo_selectors:
            elements = soup.select(selector)
            logos_found.extend(elements)
        
        # Check logo compliance (simplified)
        brand_logos = self.config["brand_guidelines"]["logo_variations"]
        
        for logo in logos_found:
            src = logo.get('src', '')
            alt = logo.get('alt', '')
            
            is_compliant = any(
                brand_logo in src.lower() or brand_logo in alt.lower()
                for brand_logo in brand_logos
            )
            
            if not is_compliant:
                inconsistencies.append(BrandInconsistency(
                    url=url,
                    element_type="logo",
                    expected=f"Brand logo variations: {', '.join(brand_logos)}",
                    actual=f"src: {src}, alt: {alt}",
                    severity="high",
                    confidence=0.6,
                    detected_at=datetime.now()
                ))
        
        return inconsistencies
    
    async def _check_messaging(
        self, 
        url: str, 
        soup: BeautifulSoup
    ) -> List[BrandInconsistency]:
        """Check messaging consistency"""
        inconsistencies = []
        
        # Extract text content
        text_content = soup.get_text().lower()
        
        # Check for tone keywords
        tone_keywords = self.config["brand_guidelines"]["tone_keywords"]
        found_keywords = [kw for kw in tone_keywords if kw in text_content]
        
        consistency_ratio = len(found_keywords) / len(tone_keywords)
        threshold = self.config["thresholds"]["messaging_consistency_threshold"]
        
        if consistency_ratio < threshold:
            inconsistencies.append(BrandInconsistency(
                url=url,
                element_type="messaging",
                expected=f"Brand tone keywords: {', '.join(tone_keywords)}",
                actual=f"Found: {', '.join(found_keywords)} ({consistency_ratio:.1%})",
                severity="medium",
                confidence=0.5,
                detected_at=datetime.now()
            ))
        
        return inconsistencies
    
    def generate_consistency_report(
        self, 
        consistency_data: Dict[str, List[BrandInconsistency]]
    ) -> Dict[str, Any]:
        """Generate brand consistency report"""
        total_urls = len(consistency_data)
        total_issues = sum(len(issues) for issues in consistency_data.values())
        
        # Group by severity
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        element_type_counts = {}
        
        for issues in consistency_data.values():
            for issue in issues:
                severity_counts[issue.severity] += 1
                element_type_counts[issue.element_type] = \
                    element_type_counts.get(issue.element_type, 0) + 1
        
        # Calculate compliance score
        if total_urls > 0:
            compliance_score = max(0, (total_urls - len([
                url for url, issues in consistency_data.items() if issues
            ])) / total_urls)
        else:
            compliance_score = 1.0
        
        report = {
            "summary": {
                "total_urls_checked": total_urls,
                "total_issues_found": total_issues,
                "compliance_score": compliance_score,
                "generated_at": datetime.now().isoformat()
            },
            "severity_breakdown": severity_counts,
            "element_type_breakdown": element_type_counts,
            "detailed_findings": {}
        }
        
        # Add detailed findings
        for url, issues in consistency_data.items():
            if issues:
                report["detailed_findings"][url] = [
                    {
                        "element_type": issue.element_type,
                        "expected": issue.expected,
                        "actual": issue.actual,
                        "severity": issue.severity,
                        "confidence": issue.confidence,
                        "detected_at": issue.detected_at.isoformat() if issue.detected_at else None
                    }
                    for issue in issues
                ]
        
        return report
    
    def get_compliance_recommendations(
        self, 
        consistency_data: Dict[str, List[BrandInconsistency]]
    ) -> List[Dict[str, Any]]:
        """Get recommendations for improving brand compliance"""
        recommendations = []
        
        # Analyze common issues
        all_issues = []
        for issues in consistency_data.values():
            all_issues.extend(issues)
        
        # Group by element type
        element_issues = {}
        for issue in all_issues:
            if issue.element_type not in element_issues:
                element_issues[issue.element_type] = []
            element_issues[issue.element_type].append(issue)
        
        # Generate recommendations
        for element_type, issues in element_issues.items():
            if element_type == "color":
                recommendations.append({
                    "category": "Color Consistency",
                    "priority": "high",
                    "issue_count": len(issues),
                    "recommendation": "Implement a CSS variable system for brand colors",
                    "implementation": "Use CSS custom properties to ensure consistent color usage",
                    "impact": "Eliminates color inconsistencies across all properties"
                })
            
            elif element_type == "font":
                recommendations.append({
                    "category": "Typography",
                    "priority": "medium",
                    "issue_count": len(issues),
                    "recommendation": "Standardize font loading and fallback chains",
                    "implementation": "Use web font loader with proper fallbacks",
                    "impact": "Ensures consistent typography experience"
                })
            
            elif element_type == "logo":
                recommendations.append({
                    "category": "Logo Usage",
                    "priority": "high",
                    "issue_count": len(issues),
                    "recommendation": "Create logo usage guidelines and asset library",
                    "implementation": "Centralized asset management with version control",
                    "impact": "Maintains brand recognition and consistency"
                })
            
            elif element_type == "messaging":
                recommendations.append({
                    "category": "Brand Messaging",
                    "priority": "medium",
                    "issue_count": len(issues),
                    "recommendation": "Develop content style guide and review process",
                    "implementation": "Editorial guidelines with approval workflows",
                    "impact": "Ensures consistent brand voice and messaging"
                })
        
        return recommendations


async def main():
    """Demo usage of Brand Consistency Monitor"""
    
    # Demo URLs (replace with actual URLs)
    demo_urls = [
        "https://example.com",
        "https://example.com/about",
        "https://example.com/contact"
    ]
    
    async with BrandConsistencyMonitor() as monitor:
        print("Checking brand consistency across properties...")
        
        consistency_data = await monitor.check_brand_consistency(demo_urls)
        
        print(f"\nChecked {len(demo_urls)} URLs")
        
        # Generate report
        report = monitor.generate_consistency_report(consistency_data)
        
        print(f"\nBrand Consistency Report:")
        print(f"Compliance Score: {report['summary']['compliance_score']:.1%}")
        print(f"Total Issues: {report['summary']['total_issues_found']}")
        
        if report['severity_breakdown']['high'] > 0:
            print(f"High Priority Issues: {report['severity_breakdown']['high']}")
        
        # Get recommendations
        recommendations = monitor.get_compliance_recommendations(consistency_data)
        
        if recommendations:
            print(f"\nTop Recommendations:")
            for rec in recommendations[:3]:
                print(f"• {rec['category']}: {rec['recommendation']}")


if __name__ == "__main__":
    asyncio.run(main())