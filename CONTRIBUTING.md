# Contributing to SEO Automation Suite

## Welcome Contributors! 🚀

The SEO Automation Suite represents 27 years of SEO expertise codified for enterprise-scale operations. We welcome contributions that maintain our high standards of technical excellence and business impact.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Architecture Guidelines](#architecture-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Performance Standards](#performance-standards)
- [Security Guidelines](#security-guidelines)

## Code of Conduct

### Our Standards

- **Technical Excellence**: All contributions must demonstrate enterprise-grade quality
- **Business Focus**: Code should solve real SEO challenges faced by Fortune 500 companies
- **Knowledge Sharing**: Share the "why" behind SEO decisions, not just the "how"
- **Respectful Collaboration**: Professional communication in all interactions
- **Data Privacy**: Strict adherence to GDPR and enterprise data protection standards

### Unacceptable Behavior

- Submitting code that could be used for black-hat SEO practices
- Exposing API keys or sensitive configuration data
- Contributing modules without proper error handling or logging
- Bypassing rate limiting or API usage guidelines

## Getting Started

### Prerequisites

```bash
# Python 3.9+ required
python --version  # Should be 3.9+

# Install dependencies
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Set up environment
cp .env.example .env
# Configure your API keys in .env
```

### API Keys Required

- **Google Search Console API**: For performance data
- **Google PageSpeed Insights API**: For Core Web Vitals
- **SEMrush API**: For competitive analysis
- **Ahrefs API**: For backlink and keyword data
- **Screaming Frog API**: For technical crawling

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd seo-automation-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run tests to verify setup
pytest tests/
```

## Development Process

### Branch Naming Convention

- `feature/core-web-vitals-enhancement`
- `bugfix/crawl-budget-memory-leak`
- `docs/api-reference-update`
- `performance/query-optimization`

### Commit Message Format

```
type(scope): brief description

Longer explanation of the change, including:
- Business impact of the change
- SEO expertise demonstrated
- Performance implications
- Any breaking changes

Fixes #123
```

**Types:**
- `feat`: New SEO functionality
- `fix`: Bug fix that improves SEO performance
- `docs`: Documentation improvements
- `perf`: Performance optimization
- `refactor`: Code improvement without changing functionality
- `test`: Test additions or improvements

### Pull Request Process

1. **Branch from main**: Always create feature branches from the latest main
2. **Implement with tests**: Include comprehensive tests for all new functionality
3. **Update documentation**: Ensure all public APIs are documented
4. **Performance benchmarks**: Include performance impact analysis
5. **Security review**: Verify no sensitive data exposure
6. **Business justification**: Explain the SEO value of your contribution

### Code Review Checklist

#### Technical Requirements
- [ ] All functions have type hints
- [ ] Error handling covers edge cases
- [ ] Logging provides actionable insights
- [ ] Rate limiting implemented for API calls
- [ ] Memory usage optimized for large datasets
- [ ] Async/await used where appropriate

#### SEO Expertise Requirements
- [ ] Solution addresses real enterprise SEO challenges
- [ ] Follows current Google guidelines and best practices
- [ ] Considers international SEO implications
- [ ] Handles large-scale website scenarios (100k+ pages)
- [ ] Provides actionable, prioritized recommendations

#### Business Impact Requirements
- [ ] Clear ROI potential for Fortune 500 implementations
- [ ] Executive-level reporting capabilities
- [ ] Scales to multi-domain portfolio management
- [ ] Integrates with enterprise marketing stacks

## Architecture Guidelines

### Module Structure

```
module_name/
├── __init__.py
├── core.py           # Main functionality
├── models.py         # Data classes and schemas
├── exceptions.py     # Custom exceptions
├── utils.py          # Helper functions
└── tests/
    ├── test_core.py
    ├── test_models.py
    └── fixtures/
```

### Design Principles

1. **Single Responsibility**: Each module solves one SEO challenge exceptionally well
2. **Enterprise Scalability**: Design for 100k+ pages, multi-domain scenarios
3. **API-First**: All functionality accessible via clean APIs
4. **Async by Default**: Use asyncio for I/O operations
5. **Data Validation**: Strict input validation with Pydantic models
6. **Comprehensive Logging**: Structured logging for audit trails

### Error Handling Standards

```python
import structlog
from typing import Optional

logger = structlog.get_logger()

class SEOAnalysisError(Exception):
    """Base exception for SEO analysis errors"""
    def __init__(self, message: str, domain: Optional[str] = None):
        self.domain = domain
        super().__init__(message)
        logger.error("seo_analysis_error", message=message, domain=domain)

def analyze_domain(domain: str) -> DomainAnalysis:
    """Analyze domain with comprehensive error handling"""
    try:
        # Validate input
        if not domain or not isinstance(domain, str):
            raise ValueError(f"Invalid domain: {domain}")
            
        # Perform analysis
        result = perform_analysis(domain)
        
        logger.info("domain_analysis_complete", 
                   domain=domain, 
                   metrics_count=len(result.metrics))
        
        return result
        
    except requests.RequestException as e:
        raise SEOAnalysisError(f"API request failed: {str(e)}", domain)
    except Exception as e:
        logger.exception("unexpected_error", domain=domain)
        raise SEOAnalysisError(f"Analysis failed: {str(e)}", domain)
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Individual function testing
2. **Integration Tests**: API integration testing
3. **Performance Tests**: Large dataset handling
4. **Security Tests**: Data protection validation

### Test Structure

```python
import pytest
import asyncio
from unittest.mock import Mock, patch

class TestCoreWebVitalsMonitor:
    """Test Core Web Vitals monitoring functionality"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance with test configuration"""
        return CoreWebVitalsMonitor("test_api_key", ["example.com"])
    
    @pytest.mark.asyncio
    async def test_analyze_domain_success(self, monitor):
        """Test successful domain analysis"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock API response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {"lighthouseResult": {}}
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test analysis
            result = await monitor.analyze_domain("example.com")
            
            # Assertions
            assert result["domain"] == "example.com"
            assert "web_vitals" in result
            assert "recommendations" in result
    
    def test_calculate_opportunity_score_high_volume(self, monitor):
        """Test opportunity scoring for high-volume keywords"""
        opportunity = KeywordOpportunity(
            keyword="enterprise seo software",
            search_volume=50000,
            difficulty=45.0,
            # ... other fields
        )
        
        scored = monitor._score_opportunities([opportunity])
        assert scored[0].opportunity_score > 70
```

### Performance Testing

```python
import time
import pytest
from memory_profiler import profile

def test_large_dataset_performance():
    """Test performance with enterprise-scale datasets"""
    # Test with 100k URLs
    large_dataset = generate_test_urls(100000)
    
    start_time = time.time()
    result = analyze_crawl_budget(large_dataset)
    execution_time = time.time() - start_time
    
    # Performance requirements
    assert execution_time < 30.0  # Must complete within 30 seconds
    assert len(result.recommendations) > 0
    assert result.memory_usage < 500  # MB

@profile
def test_memory_usage():
    """Profile memory usage for optimization"""
    analyzer = KeywordGapAnalyzer("example.com", ["competitor.com"], {})
    result = analyzer.analyze_keyword_gaps()
    # Memory profiling output will be generated
```

## Documentation Standards

### Docstring Format

```python
def analyze_keyword_gaps(self, target_country: str = "us") -> List[KeywordOpportunity]:
    """Comprehensive keyword gap analysis for enterprise SEO.
    
    Identifies content gaps and opportunities by analyzing competitor
    keyword rankings using advanced machine learning clustering and
    search intent classification.
    
    Args:
        target_country: Country code for localized analysis (default: "us")
        
    Returns:
        List of keyword opportunities ranked by business impact potential.
        Each opportunity includes search volume, difficulty, competitor
        rankings, and actionable content recommendations.
        
    Raises:
        SEOAnalysisError: When API requests fail or data is insufficient
        ValueError: When target_country format is invalid
        
    Example:
        >>> analyzer = KeywordGapAnalyzer("mysite.com", ["competitor.com"], api_keys)
        >>> opportunities = await analyzer.analyze_keyword_gaps("uk")
        >>> high_value = [o for o in opportunities if o.opportunity_score > 80]
        >>> print(f"Found {len(high_value)} high-value opportunities")
        
    Business Impact:
        This analysis typically identifies 200-500 keyword opportunities
        with combined monthly search volume of 1M+ searches. Implementation
        of top 20 recommendations usually results in 30-50% organic traffic
        increase within 6 months for enterprise websites.
        
    SEO Expertise:
        Uses semantic clustering to identify topical content gaps,
        search intent classification for content type recommendations,
        and competitive analysis to prioritize quick wins vs. long-term
        strategic opportunities.
    """
```

### API Documentation

All public APIs must include:
- Clear business value proposition
- Expected ROI and implementation timeline
- Enterprise scalability considerations
- Integration examples with popular marketing tools

## Performance Standards

### Response Time Requirements

- **Real-time APIs**: < 500ms response time
- **Batch Analysis**: < 30 seconds for 10k URLs
- **Report Generation**: < 2 minutes for comprehensive analysis
- **Memory Usage**: < 500MB for typical enterprise datasets

### Scalability Requirements

- **Concurrent Requests**: Handle 100+ concurrent API calls
- **Data Volume**: Process 1M+ URLs without memory issues
- **Multi-domain**: Support 1000+ domains in single analysis
- **Rate Limiting**: Respect all third-party API limits

## Security Guidelines

### Data Protection

```python
import os
from cryptography.fernet import Fernet

class SecureAPIClient:
    """Secure API client with encrypted credential storage"""
    
    def __init__(self):
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        if not self.encryption_key:
            raise ValueError("ENCRYPTION_KEY environment variable required")
        self.cipher = Fernet(self.encryption_key)
    
    def store_api_key(self, service: str, api_key: str) -> None:
        """Store API key with encryption"""
        encrypted_key = self.cipher.encrypt(api_key.encode())
        # Store encrypted_key securely
    
    def get_api_key(self, service: str) -> str:
        """Retrieve and decrypt API key"""
        encrypted_key = self.load_encrypted_key(service)
        return self.cipher.decrypt(encrypted_key).decode()
```

### Sensitive Data Handling

- **Never log API keys or credentials**
- **Encrypt sensitive data at rest**
- **Use environment variables for configuration**
- **Implement proper access controls**
- **Regular security audits of dependencies**

### GDPR Compliance

```python
class GDPRCompliantAnalyzer:
    """SEO analyzer with GDPR compliance built-in"""
    
    def __init__(self, data_retention_days: int = 30):
        self.data_retention_days = data_retention_days
        
    def analyze_with_consent(self, domain: str, user_consent: bool) -> Analysis:
        """Perform analysis only with explicit user consent"""
        if not user_consent:
            raise ConsentRequiredError("User consent required for analysis")
            
        # Perform analysis
        result = self.analyze(domain)
        
        # Schedule data deletion
        self.schedule_data_deletion(domain, self.data_retention_days)
        
        return result
```

## Contribution Workflow

### 1. Issue Creation
- Use issue templates for bug reports and feature requests
- Include business justification for new features
- Provide reproduction steps for bugs
- Tag with appropriate labels (performance, security, etc.)

### 2. Development
- Fork the repository
- Create feature branch from main
- Implement with comprehensive tests
- Ensure all checks pass locally

### 3. Pull Request
- Use PR template
- Include performance benchmarks
- Link to related issues
- Request review from code owners

### 4. Review Process
- Technical review by maintainers
- SEO expertise validation
- Performance impact assessment
- Security review if applicable

### 5. Merge Requirements
- All tests passing
- Code coverage > 90%
- Performance benchmarks met
- Documentation updated
- At least two approving reviews

## Recognition

Contributors who demonstrate exceptional SEO expertise and technical excellence will be:

- **Featured in documentation** with SEO insights shared
- **Invited to speak** at SEO conferences and webinars
- **Credited in case studies** showing business impact
- **Considered for consulting** opportunities with enterprise clients

## Questions?

- **Technical Questions**: Open a GitHub issue with `question` label
- **SEO Strategy**: Contact the maintainers for expert consultation
- **Enterprise Integration**: Reach out for custom implementation support

## Links

- [VerityAI AI SEO Services](https://verityai.co/landing/ai-seo-services)
- [SEO Best Practices Documentation](./docs/seo-best-practices.md)
- [API Reference](./docs/api-reference.md)
- [Enterprise Integration Guide](./docs/enterprise-integration.md)

---

**Thank you for contributing to the advancement of enterprise SEO automation! 🎯**