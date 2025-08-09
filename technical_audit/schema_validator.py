"""
Schema Validator - Enterprise Structured Data Optimization Platform  
Advanced schema markup validation and rich snippets optimization for maximum search visibility

🎯 PORTFOLIO PROJECT: Demonstrates structured data expertise and semantic SEO knowledge
Perfect for: Technical SEO specialists, frontend developers, content strategists

📄 DEMO/PORTFOLIO CODE: This is demonstration code showcasing schema validation capabilities.
   Real implementations require comprehensive schema.org integration and testing workflows.

🔗 Connect with the developer: https://www.linkedin.com/in/sspyrou/
🚀 AI-Enhanced SEO Solutions: https://verityai.co

Built by a technical marketing leader with expertise in structured data implementations
that achieved significant rich snippets visibility and search performance improvements.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import json
import re
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
import jsonschema
from jsonschema import validate, ValidationError
import extruct
import pandas as pd


@dataclass
class SchemaIssue:
    """Individual schema validation issue"""
    url: str
    schema_type: str  # JSON-LD, Microdata, RDFa
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    recommendation: str
    schema_location: str = ""  # XPath or selector
    invalid_markup: str = ""
    expected_format: str = ""
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class SchemaMarkup:
    """Parsed schema markup"""
    markup_type: str  # JSON-LD, Microdata, RDFa
    schema_org_type: str  # Organization, Product, Article, etc.
    markup_content: Dict[str, Any]
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    rich_snippet_eligible: bool = False
    completeness_score: float = 0.0  # 0-1 based on required properties


@dataclass
class PageSchemaAnalysis:
    """Schema analysis for a single page"""
    url: str
    schemas_found: List[SchemaMarkup] = field(default_factory=list)
    issues: List[SchemaIssue] = field(default_factory=list)
    rich_snippet_types: List[str] = field(default_factory=list)
    schema_coverage_score: int = 0  # 0-100
    total_schemas: int = 0
    valid_schemas: int = 0
    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class SiteSchemaAnalysis:
    """Complete site schema analysis"""
    site_url: str
    page_analyses: List[PageSchemaAnalysis] = field(default_factory=list)
    schema_type_coverage: Dict[str, int] = field(default_factory=dict)
    rich_snippet_opportunities: List[str] = field(default_factory=list)
    common_issues: List[Dict[str, Any]] = field(default_factory=list)
    overall_schema_score: int = 0
    recommendations: List[str] = field(default_factory=list)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)


class SchemaValidator:
    """Comprehensive structured data validator for SEO"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.session: Optional[aiohttp.ClientSession] = None
        self.schema_definitions = self._load_schema_definitions()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "validation": {
                "schema_org_version": "15.0",
                "validate_against_google": True,
                "check_rich_snippets": True,
                "validate_nested_schemas": True,
                "max_schema_depth": 5
            },
            "extraction": {
                "timeout": 30,
                "max_concurrent": 15,
                "user_agent": "Schema-Validator/1.0 (+https://example.com/bot)",
                "extract_jsonld": True,
                "extract_microdata": True,
                "extract_rdfa": True
            },
            "scoring": {
                "required_property_weight": 0.4,
                "recommended_property_weight": 0.3,
                "google_guidelines_weight": 0.2,
                "validation_weight": 0.1
            },
            "rich_snippets": {
                "supported_types": [
                    "Article", "BlogPosting", "NewsArticle",
                    "Product", "Offer", "Review", "AggregateRating",
                    "Organization", "LocalBusiness", "Person",
                    "Event", "Recipe", "FAQPage", "HowTo",
                    "JobPosting", "Course", "Movie", "Book"
                ],
                "google_requirements": {
                    "Article": ["headline", "image", "datePublished", "author"],
                    "Product": ["name", "image", "description", "offers"],
                    "Organization": ["name", "url"],
                    "Review": ["itemReviewed", "reviewRating", "author"],
                    "Event": ["name", "startDate", "location"]
                }
            },
            "validation_rules": {
                "require_context": True,
                "require_type": True,
                "validate_urls": True,
                "check_image_accessibility": True,
                "validate_datetime_format": True
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
    
    def _load_schema_definitions(self) -> Dict[str, Any]:
        """Load Schema.org definitions for validation"""
        # In practice, would load from schema.org JSON-LD definitions
        # For demo, we'll use simplified definitions
        return {
            "Article": {
                "required": ["@type", "headline"],
                "recommended": ["author", "datePublished", "image", "publisher"],
                "properties": {
                    "headline": {"type": "string", "maxLength": 110},
                    "author": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                    "datePublished": {"type": "string", "format": "date-time"},
                    "image": {"anyOf": [{"type": "string"}, {"type": "array"}]}
                }
            },
            "Product": {
                "required": ["@type", "name"],
                "recommended": ["image", "description", "offers", "brand"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "image": {"anyOf": [{"type": "string"}, {"type": "array"}]},
                    "offers": {"type": "object"}
                }
            },
            "Organization": {
                "required": ["@type", "name"],
                "recommended": ["url", "logo", "contactPoint"],
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string", "format": "uri"},
                    "logo": {"anyOf": [{"type": "string"}, {"type": "object"}]}
                }
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=15)
        timeout = aiohttp.ClientTimeout(total=self.config["extraction"]["timeout"])
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.config["extraction"]["user_agent"]}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def analyze_site_schemas(
        self, 
        site_url: str,
        urls_to_analyze: List[str] = None,
        sample_size: int = 100
    ) -> SiteSchemaAnalysis:
        """Analyze structured data across entire site"""
        self.logger.info(f"Starting schema analysis for {site_url}")
        
        # Get URLs to analyze
        if not urls_to_analyze:
            urls_to_analyze = await self._discover_urls_for_analysis(site_url, sample_size)
        
        # Analyze pages concurrently
        semaphore = asyncio.Semaphore(self.config["extraction"]["max_concurrent"])
        tasks = [
            self._analyze_page_schemas(url, semaphore) 
            for url in urls_to_analyze
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        page_analyses = []
        for url, result in zip(urls_to_analyze, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error analyzing {url}: {result}")
                # Create error analysis
                error_analysis = PageSchemaAnalysis(
                    url=url,
                    schema_coverage_score=0
                )
                error_analysis.issues.append(SchemaIssue(
                    url=url,
                    schema_type="unknown",
                    issue_type="analysis_error",
                    severity="high",
                    description=f"Failed to analyze schemas: {str(result)}",
                    recommendation="Check page accessibility and markup validity"
                ))
                page_analyses.append(error_analysis)
            else:
                page_analyses.append(result)
        
        # Compile site analysis
        return self._compile_site_analysis(site_url, page_analyses)
    
    async def _discover_urls_for_analysis(self, site_url: str, sample_size: int) -> List[str]:
        """Discover URLs for schema analysis"""
        urls = set([site_url.rstrip('/')])
        
        # Try sitemap first
        sitemap_urls = await self._extract_sitemap_urls(site_url)
        urls.update(sitemap_urls[:sample_size-1])
        
        return list(urls)[:sample_size]
    
    async def _extract_sitemap_urls(self, site_url: str) -> List[str]:
        """Extract URLs from sitemap"""
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
                    
                    return urls
        except Exception as e:
            self.logger.warning(f"Could not fetch sitemap: {e}")
        
        return []
    
    async def _analyze_page_schemas(
        self, 
        url: str, 
        semaphore: asyncio.Semaphore
    ) -> PageSchemaAnalysis:
        """Analyze schemas for a single page"""
        async with semaphore:
            self.logger.debug(f"Analyzing schemas for {url}")
            
            analysis = PageSchemaAnalysis(url=url)
            
            try:
                # Fetch page content
                async with self.session.get(url) as response:
                    if response.status != 200:
                        analysis.issues.append(SchemaIssue(
                            url=url,
                            schema_type="page",
                            issue_type="http_error",
                            severity="critical",
                            description=f"HTTP {response.status} error",
                            recommendation="Fix page accessibility"
                        ))
                        return analysis
                    
                    html_content = await response.text()
                
                # Extract structured data
                extracted_data = await self._extract_structured_data(html_content, url)
                
                # Process each type of structured data
                for markup_type, data_list in extracted_data.items():
                    for data in data_list:
                        schema_markup = await self._process_schema_markup(
                            markup_type, data, url
                        )
                        if schema_markup:
                            analysis.schemas_found.append(schema_markup)
                
                # Validate schemas and identify issues
                await self._validate_page_schemas(analysis)
                
                # Calculate scores
                analysis.total_schemas = len(analysis.schemas_found)
                analysis.valid_schemas = len([s for s in analysis.schemas_found if s.is_valid])
                analysis.schema_coverage_score = self._calculate_coverage_score(analysis)
                analysis.rich_snippet_types = [
                    s.schema_org_type for s in analysis.schemas_found 
                    if s.rich_snippet_eligible
                ]
                
                return analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing {url}: {e}")
                analysis.issues.append(SchemaIssue(
                    url=url,
                    schema_type="unknown",
                    issue_type="extraction_error",
                    severity="high",
                    description=f"Schema extraction failed: {str(e)}",
                    recommendation="Verify page markup and accessibility"
                ))
                return analysis
    
    async def _extract_structured_data(
        self, 
        html_content: str, 
        url: str
    ) -> Dict[str, List[Any]]:
        """Extract structured data from HTML"""
        extracted = {"jsonld": [], "microdata": [], "rdfa": []}
        
        try:
            # Use extruct library for extraction
            data = extruct.extract(
                html_content,
                base_url=url,
                syntaxes=['json-ld', 'microdata', 'rdfa']
            )
            
            if self.config["extraction"]["extract_jsonld"]:
                extracted["jsonld"] = data.get('json-ld', [])
            
            if self.config["extraction"]["extract_microdata"]:
                extracted["microdata"] = data.get('microdata', [])
            
            if self.config["extraction"]["extract_rdfa"]:
                extracted["rdfa"] = data.get('rdfa', [])
        
        except Exception as e:
            self.logger.error(f"Error extracting structured data from {url}: {e}")
        
        return extracted
    
    async def _process_schema_markup(
        self, 
        markup_type: str, 
        markup_data: Dict[str, Any], 
        url: str
    ) -> Optional[SchemaMarkup]:
        """Process and validate individual schema markup"""
        try:
            # Determine schema type
            schema_type = self._identify_schema_type(markup_data)
            if not schema_type:
                return None
            
            # Create schema markup object
            schema_markup = SchemaMarkup(
                markup_type=markup_type,
                schema_org_type=schema_type,
                markup_content=markup_data,
                is_valid=True
            )
            
            # Validate the markup
            validation_errors = await self._validate_schema_markup(
                schema_type, markup_data, url
            )
            
            if validation_errors:
                schema_markup.is_valid = False
                schema_markup.validation_errors = validation_errors
            
            # Check rich snippet eligibility
            schema_markup.rich_snippet_eligible = self._check_rich_snippet_eligibility(
                schema_type, markup_data
            )
            
            # Calculate completeness score
            schema_markup.completeness_score = self._calculate_completeness_score(
                schema_type, markup_data
            )
            
            return schema_markup
        
        except Exception as e:
            self.logger.error(f"Error processing schema markup: {e}")
            return None
    
    def _identify_schema_type(self, markup_data: Dict[str, Any]) -> Optional[str]:
        """Identify Schema.org type from markup data"""
        # Handle different markup formats
        if isinstance(markup_data, dict):
            # JSON-LD format
            schema_type = markup_data.get('@type')
            if schema_type:
                if isinstance(schema_type, list):
                    schema_type = schema_type[0]
                # Remove schema.org prefix if present
                if isinstance(schema_type, str):
                    return schema_type.split('/')[-1]
            
            # Microdata format
            if 'type' in markup_data:
                type_value = markup_data['type']
                if isinstance(type_value, list):
                    type_value = type_value[0]
                if isinstance(type_value, str):
                    return type_value.split('/')[-1]
        
        return None
    
    async def _validate_schema_markup(
        self, 
        schema_type: str, 
        markup_data: Dict[str, Any], 
        url: str
    ) -> List[str]:
        """Validate schema markup against Schema.org definitions"""
        errors = []
        
        # Get schema definition
        schema_def = self.schema_definitions.get(schema_type, {})
        if not schema_def:
            return [f"Unknown schema type: {schema_type}"]
        
        # Check required properties
        required_props = schema_def.get("required", [])
        for prop in required_props:
            if prop not in markup_data:
                errors.append(f"Missing required property: {prop}")
        
        # Validate property formats
        properties = schema_def.get("properties", {})
        for prop, value in markup_data.items():
            if prop in properties:
                prop_def = properties[prop]
                validation_error = self._validate_property(prop, value, prop_def)
                if validation_error:
                    errors.append(validation_error)
        
        # Validate URLs if required
        if self.config["validation_rules"]["validate_urls"]:
            url_errors = self._validate_urls_in_markup(markup_data, url)
            errors.extend(url_errors)
        
        # Validate datetime formats
        if self.config["validation_rules"]["validate_datetime_format"]:
            datetime_errors = self._validate_datetime_formats(markup_data)
            errors.extend(datetime_errors)
        
        return errors
    
    def _validate_property(
        self, 
        prop_name: str, 
        value: Any, 
        prop_def: Dict[str, Any]
    ) -> Optional[str]:
        """Validate individual property"""
        try:
            # Use jsonschema for validation
            validate(instance=value, schema=prop_def)
            return None
        except ValidationError as e:
            return f"Property '{prop_name}': {e.message}"
        except Exception:
            return None
    
    def _validate_urls_in_markup(
        self, 
        markup_data: Dict[str, Any], 
        base_url: str
    ) -> List[str]:
        """Validate URLs in markup"""
        errors = []
        url_properties = ['url', 'image', 'logo', 'sameAs']
        
        for prop in url_properties:
            if prop in markup_data:
                urls = markup_data[prop]
                if isinstance(urls, str):
                    urls = [urls]
                elif isinstance(urls, list):
                    pass
                else:
                    continue
                
                for url in urls:
                    if isinstance(url, str) and not self._is_valid_url(url):
                        errors.append(f"Invalid URL in {prop}: {url}")
        
        return errors
    
    def _validate_datetime_formats(self, markup_data: Dict[str, Any]) -> List[str]:
        """Validate datetime format compliance"""
        errors = []
        datetime_properties = [
            'datePublished', 'dateModified', 'dateCreated',
            'startDate', 'endDate', 'validFrom', 'validThrough'
        ]
        
        for prop in datetime_properties:
            if prop in markup_data:
                datetime_value = markup_data[prop]
                if isinstance(datetime_value, str):
                    if not self._is_valid_datetime_format(datetime_value):
                        errors.append(f"Invalid datetime format in {prop}: {datetime_value}")
        
        return errors
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False
    
    def _is_valid_datetime_format(self, datetime_str: str) -> bool:
        """Check if datetime string is valid ISO 8601"""
        # Simplified check - would use proper datetime parsing in practice
        iso_pattern = r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{3})?([+-]\d{2}:\d{2}|Z)?)?$'
        return bool(re.match(iso_pattern, datetime_str))
    
    def _check_rich_snippet_eligibility(
        self, 
        schema_type: str, 
        markup_data: Dict[str, Any]
    ) -> bool:
        """Check if markup is eligible for rich snippets"""
        if schema_type not in self.config["rich_snippets"]["supported_types"]:
            return False
        
        # Check Google-specific requirements
        google_requirements = self.config["rich_snippets"]["google_requirements"]
        required_props = google_requirements.get(schema_type, [])
        
        for prop in required_props:
            if prop not in markup_data:
                return False
        
        return True
    
    def _calculate_completeness_score(
        self, 
        schema_type: str, 
        markup_data: Dict[str, Any]
    ) -> float:
        """Calculate markup completeness score"""
        schema_def = self.schema_definitions.get(schema_type, {})
        if not schema_def:
            return 0.0
        
        scoring = self.config["scoring"]
        total_score = 0.0
        
        # Required properties
        required_props = schema_def.get("required", [])
        if required_props:
            required_present = sum(1 for prop in required_props if prop in markup_data)
            required_score = (required_present / len(required_props)) * scoring["required_property_weight"]
            total_score += required_score
        
        # Recommended properties
        recommended_props = schema_def.get("recommended", [])
        if recommended_props:
            recommended_present = sum(1 for prop in recommended_props if prop in markup_data)
            recommended_score = (recommended_present / len(recommended_props)) * scoring["recommended_property_weight"]
            total_score += recommended_score
        
        # Google guidelines compliance
        google_requirements = self.config["rich_snippets"]["google_requirements"].get(schema_type, [])
        if google_requirements:
            google_present = sum(1 for prop in google_requirements if prop in markup_data)
            google_score = (google_present / len(google_requirements)) * scoring["google_guidelines_weight"]
            total_score += google_score
        
        # Validation score
        validation_score = scoring["validation_weight"]  # Full score if no validation errors
        total_score += validation_score
        
        return min(1.0, total_score)
    
    async def _validate_page_schemas(self, analysis: PageSchemaAnalysis):
        """Validate all schemas on a page and identify issues"""
        url = analysis.url
        
        # Check for missing schemas on content-rich pages
        if not analysis.schemas_found:
            analysis.issues.append(SchemaIssue(
                url=url,
                schema_type="missing",
                issue_type="no_structured_data",
                severity="medium",
                description="No structured data found on page",
                recommendation="Implement relevant Schema.org markup for better search visibility"
            ))
        
        # Validate individual schemas
        for schema in analysis.schemas_found:
            if not schema.is_valid:
                for error in schema.validation_errors:
                    analysis.issues.append(SchemaIssue(
                        url=url,
                        schema_type=schema.markup_type,
                        issue_type="validation_error",
                        severity="high",
                        description=f"{schema.schema_org_type} validation error: {error}",
                        recommendation="Fix markup validation errors",
                        invalid_markup=json.dumps(schema.markup_content, indent=2)
                    ))
            
            # Check completeness
            if schema.completeness_score < 0.7:
                analysis.issues.append(SchemaIssue(
                    url=url,
                    schema_type=schema.markup_type,
                    issue_type="incomplete_markup",
                    severity="medium",
                    description=f"{schema.schema_org_type} markup incomplete ({schema.completeness_score:.1%})",
                    recommendation="Add missing required and recommended properties"
                ))
            
            # Check rich snippet eligibility
            if not schema.rich_snippet_eligible and schema.schema_org_type in self.config["rich_snippets"]["supported_types"]:
                analysis.issues.append(SchemaIssue(
                    url=url,
                    schema_type=schema.markup_type,
                    issue_type="rich_snippet_ineligible",
                    severity="medium",
                    description=f"{schema.schema_org_type} markup not eligible for rich snippets",
                    recommendation="Add missing properties required by Google for rich snippets"
                ))
        
        # Check for duplicate schemas
        schema_types = [s.schema_org_type for s in analysis.schemas_found]
        duplicates = [t for t in set(schema_types) if schema_types.count(t) > 1]
        
        for duplicate_type in duplicates:
            analysis.issues.append(SchemaIssue(
                url=url,
                schema_type="multiple",
                issue_type="duplicate_schema",
                severity="medium",
                description=f"Multiple {duplicate_type} schemas found",
                recommendation="Consolidate duplicate schemas or ensure they serve different purposes"
            ))
    
    def _calculate_coverage_score(self, analysis: PageSchemaAnalysis) -> int:
        """Calculate schema coverage score for a page"""
        if not analysis.schemas_found:
            return 0
        
        base_score = 50  # Base score for having any schema
        
        # Add points for valid schemas
        valid_ratio = analysis.valid_schemas / analysis.total_schemas if analysis.total_schemas > 0 else 0
        base_score += valid_ratio * 30
        
        # Add points for rich snippet eligibility
        eligible_count = len([s for s in analysis.schemas_found if s.rich_snippet_eligible])
        eligible_ratio = eligible_count / analysis.total_schemas if analysis.total_schemas > 0 else 0
        base_score += eligible_ratio * 20
        
        # Deduct points for issues
        critical_issues = len([i for i in analysis.issues if i.severity == "critical"])
        high_issues = len([i for i in analysis.issues if i.severity == "high"])
        
        base_score -= critical_issues * 15
        base_score -= high_issues * 8
        
        return max(0, min(100, int(base_score)))
    
    def _compile_site_analysis(
        self, 
        site_url: str, 
        page_analyses: List[PageSchemaAnalysis]
    ) -> SiteSchemaAnalysis:
        """Compile site-wide schema analysis"""
        
        # Calculate schema type coverage
        schema_type_counts = {}
        for analysis in page_analyses:
            for schema in analysis.schemas_found:
                schema_type = schema.schema_org_type
                schema_type_counts[schema_type] = schema_type_counts.get(schema_type, 0) + 1
        
        # Find rich snippet opportunities
        all_schema_types = set(schema_type_counts.keys())
        supported_types = set(self.config["rich_snippets"]["supported_types"])
        opportunities = list(supported_types - all_schema_types)
        
        # Identify common issues
        common_issues = self._identify_common_schema_issues(page_analyses)
        
        # Calculate overall score
        if page_analyses:
            overall_score = sum(p.schema_coverage_score for p in page_analyses) // len(page_analyses)
        else:
            overall_score = 0
        
        # Generate recommendations
        recommendations = self._generate_schema_recommendations(
            page_analyses, schema_type_counts, opportunities
        )
        
        # Create analysis summary
        total_pages = len(page_analyses)
        pages_with_schemas = len([p for p in page_analyses if p.schemas_found])
        total_schemas = sum(len(p.schemas_found) for p in page_analyses)
        valid_schemas = sum(p.valid_schemas for p in page_analyses)
        
        analysis_summary = {
            "pages_with_schemas": pages_with_schemas,
            "pages_without_schemas": total_pages - pages_with_schemas,
            "schema_coverage_percentage": (pages_with_schemas / total_pages * 100) if total_pages > 0 else 0,
            "total_schemas_found": total_schemas,
            "valid_schemas": valid_schemas,
            "validation_success_rate": (valid_schemas / total_schemas * 100) if total_schemas > 0 else 0,
            "rich_snippet_eligible_pages": len([p for p in page_analyses if p.rich_snippet_types])
        }
        
        return SiteSchemaAnalysis(
            site_url=site_url,
            page_analyses=page_analyses,
            schema_type_coverage=schema_type_counts,
            rich_snippet_opportunities=opportunities,
            common_issues=common_issues,
            overall_schema_score=overall_score,
            recommendations=recommendations,
            analysis_summary=analysis_summary
        )
    
    def _identify_common_schema_issues(
        self, 
        page_analyses: List[PageSchemaAnalysis]
    ) -> List[Dict[str, Any]]:
        """Identify most common schema issues across site"""
        issue_counts = {}
        total_pages = len(page_analyses)
        
        for analysis in page_analyses:
            for issue in analysis.issues:
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
        
        # Sort by frequency and add percentage
        common_issues = sorted(issue_counts.values(), key=lambda x: x["count"], reverse=True)
        
        for issue in common_issues:
            issue["percentage"] = (issue["count"] / total_pages * 100) if total_pages > 0 else 0
        
        return common_issues[:10]  # Return top 10 most common issues
    
    def _generate_schema_recommendations(
        self, 
        page_analyses: List[PageSchemaAnalysis],
        schema_type_counts: Dict[str, int],
        opportunities: List[str]
    ) -> List[str]:
        """Generate schema optimization recommendations"""
        recommendations = []
        
        # Pages without schemas
        pages_without_schemas = [p for p in page_analyses if not p.schemas_found]
        if pages_without_schemas:
            recommendations.append(
                f"Add structured data to {len(pages_without_schemas)} pages lacking any schema markup"
            )
        
        # Validation issues
        pages_with_errors = [p for p in page_analyses if any(i.severity in ["critical", "high"] for i in p.issues)]
        if pages_with_errors:
            recommendations.append(
                f"Fix validation errors on {len(pages_with_errors)} pages with schema issues"
            )
        
        # Rich snippet opportunities
        if opportunities:
            recommendations.append(
                f"Implement {', '.join(opportunities[:3])} schemas for rich snippet opportunities"
            )
        
        # Missing properties
        incomplete_schemas = []
        for analysis in page_analyses:
            for schema in analysis.schemas_found:
                if schema.completeness_score < 0.8:
                    incomplete_schemas.append(schema.schema_org_type)
        
        if incomplete_schemas:
            common_incomplete = max(set(incomplete_schemas), key=incomplete_schemas.count)
            recommendations.append(
                f"Complete {common_incomplete} schema properties for better rich snippet eligibility"
            )
        
        # Content-specific recommendations
        if "Article" not in schema_type_counts:
            recommendations.append("Add Article schema to blog posts and news content")
        
        if "Product" not in schema_type_counts:
            recommendations.append("Implement Product schema for e-commerce pages")
        
        if "Organization" not in schema_type_counts:
            recommendations.append("Add Organization schema to homepage and about page")
        
        recommendations.extend([
            "Regularly validate structured data using Google's Rich Results Test",
            "Monitor search performance for pages with rich snippets",
            "Keep schema markup updated with latest Schema.org specifications"
        ])
        
        return recommendations[:8]  # Return top 8 recommendations
    
    def export_schema_analysis(
        self, 
        analysis: SiteSchemaAnalysis, 
        output_path: str,
        format: str = "json"
    ):
        """Export schema analysis results"""
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
        
        self.logger.info(f"Schema analysis exported to {output_path}")
    
    def _export_json(self, analysis: SiteSchemaAnalysis, path: Path):
        """Export analysis as JSON"""
        export_data = {
            "site_url": analysis.site_url,
            "overall_schema_score": analysis.overall_schema_score,
            "summary": analysis.analysis_summary,
            "schema_coverage": analysis.schema_type_coverage,
            "opportunities": analysis.rich_snippet_opportunities,
            "common_issues": analysis.common_issues,
            "recommendations": analysis.recommendations,
            "page_analyses": []
        }
        
        for page in analysis.page_analyses:
            page_data = {
                "url": page.url,
                "schema_coverage_score": page.schema_coverage_score,
                "total_schemas": page.total_schemas,
                "valid_schemas": page.valid_schemas,
                "rich_snippet_types": page.rich_snippet_types,
                "schemas": [
                    {
                        "type": schema.schema_org_type,
                        "markup_type": schema.markup_type,
                        "is_valid": schema.is_valid,
                        "completeness_score": schema.completeness_score,
                        "rich_snippet_eligible": schema.rich_snippet_eligible,
                        "validation_errors": schema.validation_errors
                    }
                    for schema in page.schemas_found
                ],
                "issues": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "recommendation": issue.recommendation
                    }
                    for issue in page.issues
                ]
            }
            export_data["page_analyses"].append(page_data)
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _export_csv(self, analysis: SiteSchemaAnalysis, path: Path):
        """Export analysis as CSV"""
        rows = []
        for page in analysis.page_analyses:
            rows.append({
                "url": page.url,
                "schema_score": page.schema_coverage_score,
                "total_schemas": page.total_schemas,
                "valid_schemas": page.valid_schemas,
                "rich_snippet_eligible": len(page.rich_snippet_types),
                "critical_issues": len([i for i in page.issues if i.severity == "critical"]),
                "high_issues": len([i for i in page.issues if i.severity == "high"]),
                "total_issues": len(page.issues),
                "schema_types": ", ".join([s.schema_org_type for s in page.schemas_found])
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    
    def _export_html(self, analysis: SiteSchemaAnalysis, path: Path):
        """Export analysis as HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Schema Markup Analysis Report</title>
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
            <h1>Schema Markup Analysis Report</h1>
            
            <div class="summary">
                <h2>Overall Schema Score: <span class="score">{analysis.overall_schema_score}</span></h2>
                <p><strong>Pages Analyzed:</strong> {len(analysis.page_analyses)}</p>
                <p><strong>Pages with Schemas:</strong> {analysis.analysis_summary['pages_with_schemas']}</p>
                <p><strong>Schema Coverage:</strong> {analysis.analysis_summary['schema_coverage_percentage']:.1f}%</p>
                <p><strong>Validation Success Rate:</strong> {analysis.analysis_summary['validation_success_rate']:.1f}%</p>
                <p><strong>Rich Snippet Eligible Pages:</strong> {analysis.analysis_summary['rich_snippet_eligible_pages']}</p>
            </div>
            
            <div class="recommendations">
                <h2>Key Recommendations</h2>
                <ul>
        """
        
        for rec in analysis.recommendations[:5]:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <h2>Schema Type Coverage</h2>
            <table>
                <tr><th>Schema Type</th><th>Pages</th></tr>
        """
        
        for schema_type, count in sorted(analysis.schema_type_coverage.items(), key=lambda x: x[1], reverse=True):
            html_content += f"<tr><td>{schema_type}</td><td>{count}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Page Analysis Results</h2>
            <table>
                <tr>
                    <th>URL</th>
                    <th>Schema Score</th>
                    <th>Total Schemas</th>
                    <th>Valid Schemas</th>
                    <th>Issues</th>
                    <th>Schema Types</th>
                </tr>
        """
        
        for page in analysis.page_analyses[:20]:  # Limit to first 20
            schema_types = ", ".join([s.schema_org_type for s in page.schemas_found])
            issues_count = len(page.issues)
            html_content += f"""
                <tr>
                    <td>{page.url}</td>
                    <td>{page.schema_coverage_score}</td>
                    <td>{page.total_schemas}</td>
                    <td>{page.valid_schemas}</td>
                    <td>{issues_count}</td>
                    <td>{schema_types}</td>
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
    """Demo usage of Schema Validator"""
    
    async with SchemaValidator() as validator:
        print("Schema Markup Validator Demo")
        
        # Analyze site schemas
        site_url = "https://example.com"
        sample_urls = [
            "https://example.com",
            "https://example.com/about",
            "https://example.com/products/widget-1",
            "https://example.com/blog/how-to-guide"
        ]
        
        print(f"\n🔍 Analyzing schema markup for {site_url}...")
        
        analysis = await validator.analyze_site_schemas(
            site_url=site_url,
            urls_to_analyze=sample_urls
        )
        
        print(f"\nSchema Analysis Results:")
        print(f"Overall Schema Score: {analysis.overall_schema_score}/100")
        print(f"Pages with Schemas: {analysis.analysis_summary['pages_with_schemas']}/{len(analysis.page_analyses)}")
        print(f"Schema Coverage: {analysis.analysis_summary['schema_coverage_percentage']:.1f}%")
        print(f"Validation Success Rate: {analysis.analysis_summary['validation_success_rate']:.1f}%")
        
        if analysis.schema_type_coverage:
            print(f"\n📋 Schema Types Found:")
            for schema_type, count in list(analysis.schema_type_coverage.items())[:5]:
                print(f"• {schema_type}: {count} pages")
        
        if analysis.rich_snippet_opportunities:
            print(f"\n🎯 Rich Snippet Opportunities:")
            for opportunity in analysis.rich_snippet_opportunities[:3]:
                print(f"• {opportunity}")
        
        if analysis.common_issues:
            print(f"\n⚠️  Most Common Issues:")
            for issue in analysis.common_issues[:3]:
                print(f"• {issue['issue_type']}: {issue['count']} pages ({issue['percentage']:.1f}%)")
        
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(analysis.recommendations[:3], 1):
            print(f"{i}. {rec}")
        
        # Export results
        validator.export_schema_analysis(analysis, "schema_analysis.json", "json")
        print(f"\n✅ Analysis exported to schema_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())