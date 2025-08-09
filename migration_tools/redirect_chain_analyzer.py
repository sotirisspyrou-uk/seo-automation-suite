"""
Redirect Chain Analyzer - Enterprise SEO Migration Safety Platform
Advanced redirect chain analysis and optimization for zero-loss website migrations

🎯 PORTFOLIO PROJECT: Demonstrates migration expertise and SEO technical knowledge
Perfect for: Technical SEO specialists, enterprise developers, migration project managers

📄 DEMO/PORTFOLIO CODE: This is demonstration code showcasing redirect analysis capabilities.
   Real implementations require comprehensive testing environments and rollback procedures.

🔗 Connect with the developer: https://www.linkedin.com/in/sspyrou/
🚀 AI-Enhanced Migration Solutions: https://verityai.co

Built by a technical marketing leader with 27 years of SEO expertise,
specializing in zero-loss migrations that protected £12M+ ARR during platform changes.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path

import aiohttp
import pandas as pd
from aiohttp import ClientResponseError


@dataclass
class RedirectHop:
    """Single redirect hop in a chain"""
    from_url: str
    to_url: str
    status_code: int
    response_time: float
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RedirectChain:
    """Complete redirect chain analysis"""
    original_url: str
    final_url: str
    hops: List[RedirectHop] = field(default_factory=list)
    total_hops: int = 0
    total_time: float = 0.0
    status: str = "unknown"  # success, loop, broken, timeout
    issues: List[str] = field(default_factory=list)
    seo_impact: str = "low"  # low, medium, high
    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class RedirectAnalysis:
    """Comprehensive redirect analysis results"""
    chains: List[RedirectChain]
    summary_stats: Dict[str, Any]
    performance_metrics: Dict[str, float]
    seo_recommendations: List[Dict[str, Any]]
    redirect_map: Dict[str, str]


class RedirectChainAnalyzer:
    """Analyze and optimize redirect chains"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "analysis": {
                "max_hops": 10,
                "timeout": 30,
                "max_concurrent": 20,
                "follow_redirects": False,  # We handle manually
                "user_agent": "SEO-Redirect-Analyzer/1.0"
            },
            "performance": {
                "warning_hop_count": 3,
                "critical_hop_count": 5,
                "slow_response_threshold": 2.0,
                "very_slow_threshold": 5.0
            },
            "seo": {
                "preserve_parameters": True,
                "check_canonical": True,
                "monitor_status_codes": [301, 302, 303, 307, 308],
                "preferred_redirect_code": 301
            },
            "reporting": {
                "include_headers": ["location", "cache-control", "expires"],
                "group_by_domain": True,
                "export_formats": ["json", "csv", "html"]
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
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(
            total=self.config["analysis"]["timeout"]
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.config["analysis"]["user_agent"]}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def analyze_redirect_chains(
        self, 
        urls: List[str]
    ) -> RedirectAnalysis:
        """Analyze redirect chains for multiple URLs"""
        self.logger.info(f"Analyzing redirect chains for {len(urls)} URLs")
        
        semaphore = asyncio.Semaphore(
            self.config["analysis"]["max_concurrent"]
        )
        
        tasks = []
        for url in urls:
            task = self._analyze_single_chain(url, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        chains = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error analyzing {url}: {result}")
                # Create error chain
                error_chain = RedirectChain(
                    original_url=url,
                    final_url=url,
                    status="error",
                    issues=[f"Analysis failed: {str(result)}"],
                    seo_impact="high"
                )
                chains.append(error_chain)
            else:
                chains.append(result)
        
        return self._compile_analysis_results(chains)
    
    async def _analyze_single_chain(
        self, 
        url: str, 
        semaphore: asyncio.Semaphore
    ) -> RedirectChain:
        """Analyze redirect chain for a single URL"""
        async with semaphore:
            chain = RedirectChain(original_url=url, final_url=url)
            visited_urls = set()
            current_url = url
            
            try:
                while len(chain.hops) < self.config["analysis"]["max_hops"]:
                    # Check for redirect loops
                    if current_url in visited_urls:
                        chain.status = "loop"
                        chain.issues.append(f"Redirect loop detected at {current_url}")
                        chain.seo_impact = "high"
                        break
                    
                    visited_urls.add(current_url)
                    start_time = datetime.now()
                    
                    try:
                        async with self.session.get(
                            current_url,
                            allow_redirects=False
                        ) as response:
                            response_time = (datetime.now() - start_time).total_seconds()
                            
                            # Check if this is a redirect
                            if response.status in self.config["seo"]["monitor_status_codes"]:
                                location = response.headers.get('location', '')
                                if not location:
                                    chain.status = "broken"
                                    chain.issues.append(f"Redirect without location header at {current_url}")
                                    chain.seo_impact = "high"
                                    break
                                
                                # Resolve relative URLs
                                next_url = urljoin(current_url, location)
                                
                                # Record the hop
                                hop = RedirectHop(
                                    from_url=current_url,
                                    to_url=next_url,
                                    status_code=response.status,
                                    response_time=response_time,
                                    headers={k: v for k, v in response.headers.items() 
                                           if k.lower() in self.config["reporting"]["include_headers"]}
                                )
                                chain.hops.append(hop)
                                chain.total_time += response_time
                                
                                current_url = next_url
                                
                            else:
                                # Final destination reached
                                chain.final_url = current_url
                                chain.status = "success"
                                
                                # Check final response
                                if response.status >= 400:
                                    chain.status = "broken"
                                    chain.issues.append(f"Final URL returns {response.status}")
                                    chain.seo_impact = "high"
                                
                                break
                    
                    except ClientResponseError as e:
                        chain.status = "broken"
                        chain.issues.append(f"HTTP error {e.status} at {current_url}")
                        chain.seo_impact = "high"
                        break
                    
                    except Exception as e:
                        chain.status = "error"
                        chain.issues.append(f"Request failed: {str(e)}")
                        chain.seo_impact = "high"
                        break
                
                # Check if we hit max hops
                if len(chain.hops) >= self.config["analysis"]["max_hops"]:
                    chain.status = "too_many_hops"
                    chain.issues.append(f"Exceeded maximum hops ({self.config['analysis']['max_hops']})")
                    chain.seo_impact = "high"
                
                chain.total_hops = len(chain.hops)
                
                # Analyze performance and SEO impact
                self._analyze_chain_performance(chain)
                self._analyze_seo_impact(chain)
                
                return chain
                
            except Exception as e:
                self.logger.error(f"Unexpected error analyzing {url}: {e}")
                chain.status = "error"
                chain.issues.append(f"Analysis error: {str(e)}")
                chain.seo_impact = "high"
                return chain
    
    def _analyze_chain_performance(self, chain: RedirectChain):
        """Analyze performance impact of redirect chain"""
        if not chain.hops:
            return
        
        # Check hop count thresholds
        warning_hops = self.config["performance"]["warning_hop_count"]
        critical_hops = self.config["performance"]["critical_hop_count"]
        
        if chain.total_hops >= critical_hops:
            chain.issues.append(f"Critical: {chain.total_hops} redirect hops")
            if chain.seo_impact == "low":
                chain.seo_impact = "high"
        elif chain.total_hops >= warning_hops:
            chain.issues.append(f"Warning: {chain.total_hops} redirect hops")
            if chain.seo_impact == "low":
                chain.seo_impact = "medium"
        
        # Check response times
        slow_threshold = self.config["performance"]["slow_response_threshold"]
        very_slow_threshold = self.config["performance"]["very_slow_threshold"]
        
        if chain.total_time >= very_slow_threshold:
            chain.issues.append(f"Very slow total time: {chain.total_time:.2f}s")
            chain.seo_impact = "high"
        elif chain.total_time >= slow_threshold:
            chain.issues.append(f"Slow total time: {chain.total_time:.2f}s")
            if chain.seo_impact == "low":
                chain.seo_impact = "medium"
        
        # Check individual hop performance
        for i, hop in enumerate(chain.hops):
            if hop.response_time >= slow_threshold:
                chain.issues.append(f"Slow hop {i+1}: {hop.response_time:.2f}s")
    
    def _analyze_seo_impact(self, chain: RedirectChain):
        """Analyze SEO impact of redirect chain"""
        if not chain.hops:
            return
        
        preferred_code = self.config["seo"]["preferred_redirect_code"]
        
        # Check redirect types
        for i, hop in enumerate(chain.hops):
            if hop.status_code == 302:
                chain.issues.append(f"Temporary redirect (302) at hop {i+1}")
                if chain.seo_impact == "low":
                    chain.seo_impact = "medium"
            elif hop.status_code not in [301, 302, 307, 308]:
                chain.issues.append(f"Unusual redirect code {hop.status_code} at hop {i+1}")
                if chain.seo_impact == "low":
                    chain.seo_impact = "medium"
        
        # Check for domain changes
        original_domain = urlparse(chain.original_url).netloc
        final_domain = urlparse(chain.final_url).netloc
        
        if original_domain != final_domain:
            chain.issues.append(f"Domain change: {original_domain} → {final_domain}")
            if chain.seo_impact == "low":
                chain.seo_impact = "medium"
        
        # Check for HTTPS/HTTP changes
        original_scheme = urlparse(chain.original_url).scheme
        final_scheme = urlparse(chain.final_url).scheme
        
        if original_scheme == "http" and final_scheme == "https":
            # This is good - upgrading to HTTPS
            pass
        elif original_scheme == "https" and final_scheme == "http":
            chain.issues.append("Downgrading from HTTPS to HTTP")
            chain.seo_impact = "high"
    
    def _compile_analysis_results(self, chains: List[RedirectChain]) -> RedirectAnalysis:
        """Compile comprehensive analysis results"""
        total_chains = len(chains)
        
        # Calculate summary statistics
        successful_chains = len([c for c in chains if c.status == "success"])
        broken_chains = len([c for c in chains if c.status in ["broken", "error"]])
        loop_chains = len([c for c in chains if c.status == "loop"])
        long_chains = len([c for c in chains if c.total_hops >= 3])
        
        summary_stats = {
            "total_chains": total_chains,
            "successful_chains": successful_chains,
            "broken_chains": broken_chains,
            "redirect_loops": loop_chains,
            "long_chains": long_chains,
            "success_rate": successful_chains / total_chains if total_chains > 0 else 0
        }
        
        # Calculate performance metrics
        valid_chains = [c for c in chains if c.status == "success" and c.hops]
        if valid_chains:
            hop_counts = [c.total_hops for c in valid_chains]
            response_times = [c.total_time for c in valid_chains]
            
            performance_metrics = {
                "avg_hops": sum(hop_counts) / len(hop_counts),
                "max_hops": max(hop_counts),
                "avg_response_time": sum(response_times) / len(response_times),
                "max_response_time": max(response_times)
            }
        else:
            performance_metrics = {
                "avg_hops": 0,
                "max_hops": 0,
                "avg_response_time": 0,
                "max_response_time": 0
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(chains, summary_stats)
        
        # Create redirect map
        redirect_map = {}
        for chain in chains:
            if chain.status == "success":
                redirect_map[chain.original_url] = chain.final_url
        
        return RedirectAnalysis(
            chains=chains,
            summary_stats=summary_stats,
            performance_metrics=performance_metrics,
            seo_recommendations=recommendations,
            redirect_map=redirect_map
        )
    
    def _generate_recommendations(
        self, 
        chains: List[RedirectChain],
        summary_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # High-level issues
        if summary_stats["success_rate"] < 0.95:
            recommendations.append({
                "category": "Broken Redirects",
                "priority": "high",
                "issue": f"{summary_stats['broken_chains']} broken redirect chains",
                "recommendation": "Fix broken redirects immediately",
                "impact": "Critical SEO impact - lost link equity and poor UX",
                "action": "Review error chains and fix destination URLs"
            })
        
        if summary_stats["redirect_loops"] > 0:
            recommendations.append({
                "category": "Redirect Loops",
                "priority": "high",
                "issue": f"{summary_stats['redirect_loops']} redirect loops detected",
                "recommendation": "Fix redirect loops immediately",
                "impact": "Prevents page access and wastes crawl budget",
                "action": "Map redirect paths and eliminate circular references"
            })
        
        if summary_stats["long_chains"] > 0:
            recommendations.append({
                "category": "Chain Length",
                "priority": "medium",
                "issue": f"{summary_stats['long_chains']} chains with 3+ hops",
                "recommendation": "Reduce redirect chains to 1-2 hops maximum",
                "impact": "Slow page loads and potential link equity loss",
                "action": "Update redirects to point directly to final destinations"
            })
        
        # Performance issues
        slow_chains = [c for c in chains if c.total_time > 2.0]
        if slow_chains:
            recommendations.append({
                "category": "Performance",
                "priority": "medium",
                "issue": f"{len(slow_chains)} slow redirect chains (>2s)",
                "recommendation": "Optimize server response times",
                "impact": "Poor user experience and potential ranking impact",
                "action": "Review server configuration and CDN setup"
            })
        
        # SEO-specific issues
        temp_redirects = []
        for chain in chains:
            for hop in chain.hops:
                if hop.status_code == 302:
                    temp_redirects.append(chain)
                    break
        
        if temp_redirects:
            recommendations.append({
                "category": "SEO",
                "priority": "medium", 
                "issue": f"{len(temp_redirects)} chains using temporary redirects",
                "recommendation": "Use 301 redirects for permanent moves",
                "impact": "May not pass full link equity",
                "action": "Change 302 redirects to 301 for permanent moves"
            })
        
        return recommendations
    
    def export_analysis(
        self, 
        analysis: RedirectAnalysis, 
        output_path: str, 
        format: str = "json"
    ):
        """Export analysis results to file"""
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
        
        self.logger.info(f"Analysis exported to {output_path}")
    
    def _export_json(self, analysis: RedirectAnalysis, path: Path):
        """Export analysis as JSON"""
        export_data = {
            "summary": analysis.summary_stats,
            "performance": analysis.performance_metrics,
            "recommendations": analysis.seo_recommendations,
            "chains": []
        }
        
        for chain in analysis.chains:
            chain_data = {
                "original_url": chain.original_url,
                "final_url": chain.final_url,
                "total_hops": chain.total_hops,
                "total_time": chain.total_time,
                "status": chain.status,
                "issues": chain.issues,
                "seo_impact": chain.seo_impact,
                "hops": [
                    {
                        "from_url": hop.from_url,
                        "to_url": hop.to_url,
                        "status_code": hop.status_code,
                        "response_time": hop.response_time
                    }
                    for hop in chain.hops
                ]
            }
            export_data["chains"].append(chain_data)
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _export_csv(self, analysis: RedirectAnalysis, path: Path):
        """Export analysis as CSV"""
        rows = []
        for chain in analysis.chains:
            rows.append({
                "original_url": chain.original_url,
                "final_url": chain.final_url,
                "hops": chain.total_hops,
                "total_time": chain.total_time,
                "status": chain.status,
                "seo_impact": chain.seo_impact,
                "issues": "; ".join(chain.issues)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    
    def _export_html(self, analysis: RedirectAnalysis, path: Path):
        """Export analysis as HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Redirect Chain Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
                .issue {{ color: #d32f2f; }}
                .success {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Redirect Chain Analysis Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total URLs Analyzed: {analysis.summary_stats['total_chains']}</p>
                <p>Success Rate: {analysis.summary_stats['success_rate']:.1%}</p>
                <p>Broken Chains: {analysis.summary_stats['broken_chains']}</p>
                <p>Redirect Loops: {analysis.summary_stats['redirect_loops']}</p>
            </div>
            
            <h2>Recommendations</h2>
            <ul>
        """
        
        for rec in analysis.seo_recommendations:
            html_content += f"<li><strong>{rec['category']}</strong>: {rec['recommendation']}</li>"
        
        html_content += """
            </ul>
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Original URL</th>
                    <th>Final URL</th>
                    <th>Hops</th>
                    <th>Status</th>
                    <th>Issues</th>
                </tr>
        """
        
        for chain in analysis.chains:
            status_class = "success" if chain.status == "success" else "issue"
            html_content += f"""
                <tr>
                    <td>{chain.original_url}</td>
                    <td>{chain.final_url}</td>
                    <td>{chain.total_hops}</td>
                    <td class="{status_class}">{chain.status}</td>
                    <td>{'; '.join(chain.issues)}</td>
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
    """Demo usage of Redirect Chain Analyzer"""
    
    # Demo URLs with various redirect scenarios
    demo_urls = [
        "http://example.com",  # HTTP to HTTPS
        "https://www.example.com/old-page",  # Standard redirect
        "https://example.com/redirect-loop1",  # Potential loop
        "https://example.com/broken-redirect",  # Broken redirect
    ]
    
    async with RedirectChainAnalyzer() as analyzer:
        print("Analyzing redirect chains...")
        
        analysis = await analyzer.analyze_redirect_chains(demo_urls)
        
        print(f"\nRedirect Chain Analysis Results:")
        print(f"URLs Analyzed: {analysis.summary_stats['total_chains']}")
        print(f"Success Rate: {analysis.summary_stats['success_rate']:.1%}")
        print(f"Average Hops: {analysis.performance_metrics['avg_hops']:.1f}")
        print(f"Average Response Time: {analysis.performance_metrics['avg_response_time']:.2f}s")
        
        if analysis.summary_stats['broken_chains'] > 0:
            print(f"⚠️  {analysis.summary_stats['broken_chains']} broken redirect chains found")
        
        if analysis.summary_stats['redirect_loops'] > 0:
            print(f"🔄 {analysis.summary_stats['redirect_loops']} redirect loops detected")
        
        print(f"\nTop Recommendations:")
        for rec in analysis.seo_recommendations[:3]:
            print(f"• {rec['category']}: {rec['recommendation']}")
        
        # Export results
        analyzer.export_analysis(analysis, "redirect_analysis.json", "json")
        print(f"\n✅ Analysis exported to redirect_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())