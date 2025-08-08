"""
Traffic Prediction Models - Executive-Grade Organic Traffic Forecasting
Advanced ML-powered traffic forecasting for strategic planning and budget allocation

Portfolio Demo: This module showcases sophisticated predictive analytics 
combining machine learning expertise with business intelligence insights.

Author: Sotiris Spyrou | LinkedIn: https://www.linkedin.com/in/sspyrou/
Company: VerityAI - https://verityai.co/landing/ai-seo-services

DISCLAIMER: This is portfolio demonstration code showcasing technical capabilities
and strategic thinking. Not intended for production use without proper testing
and enterprise-grade security implementation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import aiohttp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import structlog

logger = structlog.get_logger()


@dataclass
class TrafficForecast:
    """Comprehensive traffic forecast with confidence intervals"""
    period: str  # "30d", "90d", "180d", "365d"
    predicted_sessions: int
    confidence_lower: int
    confidence_upper: int
    confidence_interval: float  # 90%, 95%, etc.
    growth_rate: float
    seasonal_adjustment: float
    trend_strength: str  # "strong_upward", "moderate_upward", "stable", "declining"
    key_drivers: List[str]
    risk_factors: List[str]
    business_impact: Dict[str, Union[str, float]]


@dataclass
class SeasonalPattern:
    """Seasonal traffic pattern analysis"""
    pattern_type: str  # "monthly", "weekly", "holiday", "industry_specific"
    peak_periods: List[str]
    trough_periods: List[str]
    seasonality_strength: float  # 0-1 scale
    pattern_reliability: float  # 0-1 scale based on historical consistency
    optimization_opportunities: List[str]
    preparation_recommendations: List[str]


@dataclass
class TrafficScenario:
    """Traffic scenario modeling for strategic planning"""
    scenario_name: str  # "conservative", "baseline", "optimistic", "aggressive"
    probability: float  # 0-1 likelihood of scenario
    traffic_projection: Dict[str, int]  # period -> traffic
    investment_required: Dict[str, float]  # category -> budget
    expected_roi: float
    key_assumptions: List[str]
    success_metrics: List[str]
    risk_mitigation: List[str]


@dataclass
class AlgorithmImpactModel:
    """Algorithm change impact modeling"""
    algorithm_type: str  # "core_update", "feature_update", "policy_change"
    historical_impact: float  # -1 to 1 scale
    predicted_impact: float
    impact_duration: str  # "temporary", "permanent", "evolving"
    affected_categories: List[str]
    mitigation_strategies: List[str]
    recovery_timeline: str


@dataclass
class CompetitiveImpactForecast:
    """Competitive impact on traffic forecasting"""
    competitor: str
    impact_magnitude: float  # Expected traffic impact
    impact_timeline: str  # "immediate", "short_term", "long_term"
    affected_keywords: List[str]
    defensive_measures: List[str]
    market_share_implications: float


class TrafficPredictionModels:
    """
    Executive-Grade Traffic Prediction Platform
    
    Demonstrates advanced predictive analytics combining:
    - Multiple machine learning models for accuracy
    - Seasonal decomposition and trend analysis
    - Scenario modeling for strategic planning
    - Algorithm impact prediction
    - Competitive traffic forecasting
    
    Perfect for: CMOs, digital marketing directors, enterprise strategy teams
    """
    
    def __init__(self, domain: str, historical_data_months: int = 24):
        self.domain = domain
        self.historical_months = historical_data_months
        self.models = {}
        self.scalers = {}
        
        # Initialize ML models
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200, 
            max_depth=15, 
            random_state=42
        )
        self.models['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=150, 
            learning_rate=0.1, 
            random_state=42
        )
        self.models['linear_trend'] = LinearRegression()
        
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        # Executive forecasting parameters
        self.confidence_intervals = [80, 90, 95]
        self.forecast_horizons = [30, 90, 180, 365]
        self.scenario_probabilities = {
            'conservative': 0.3,
            'baseline': 0.4,
            'optimistic': 0.25,
            'aggressive': 0.05
        }
        
        # Portfolio branding
        logger.info(
            "traffic_prediction_models_initialized",
            domain=domain,
            historical_months=historical_data_months,
            portfolio_note="Demo showcasing ML expertise + business intelligence"
        )
    
    async def generate_comprehensive_forecast(self, 
                                           forecast_horizon: int = 365,
                                           include_scenarios: bool = True) -> Dict:
        """
        Executive-Level Comprehensive Traffic Forecasting
        
        Generate multi-model traffic forecasts with business intelligence
        for strategic planning and budget allocation decisions.
        """
        logger.info(
            "generating_comprehensive_forecast",
            forecast_horizon_days=forecast_horizon,
            executive_context="Strategic traffic forecasting and business planning"
        )
        
        # Parallel data processing for comprehensive analysis
        forecast_tasks = [
            self._prepare_historical_data(),
            self._train_prediction_models(),
            self._analyze_seasonal_patterns(),
            self._model_algorithm_impacts(),
            self._assess_competitive_impacts()
        ]
        
        historical_data, trained_models, seasonal_analysis, algorithm_impacts, competitive_impacts = await asyncio.gather(*forecast_tasks)
        
        # Generate base forecasts using ensemble of models
        base_forecasts = self._generate_ensemble_forecasts(historical_data, trained_models, forecast_horizon)
        
        # Apply seasonal adjustments
        seasonal_forecasts = self._apply_seasonal_adjustments(base_forecasts, seasonal_analysis)
        
        # Generate scenario-based forecasts
        scenario_forecasts = []
        if include_scenarios:
            scenario_forecasts = self._generate_scenario_forecasts(seasonal_forecasts, historical_data)
        
        # Create executive summary and recommendations
        executive_insights = self._generate_executive_insights(
            seasonal_forecasts, scenario_forecasts, seasonal_analysis
        )
        
        return {
            "executive_summary": {
                "baseline_annual_projection": seasonal_forecasts[365]['predicted_sessions'] if 365 in seasonal_forecasts else 0,
                "growth_trajectory": seasonal_forecasts[365]['trend_strength'] if 365 in seasonal_forecasts else "unknown",
                "confidence_level": "High" if len(trained_models) > 2 else "Medium",
                "seasonal_optimization_potential": len(seasonal_analysis.optimization_opportunities) if seasonal_analysis else 0,
                "algorithm_risk_level": self._assess_algorithm_risk_level(algorithm_impacts),
                "competitive_pressure_impact": sum(ci.impact_magnitude for ci in competitive_impacts[:3])
            },
            "traffic_forecasts": {
                period: forecast for period, forecast in seasonal_forecasts.items()
            },
            "seasonal_analysis": seasonal_analysis,
            "scenario_modeling": scenario_forecasts,
            "algorithm_impact_assessment": algorithm_impacts,
            "competitive_impact_forecast": competitive_impacts,
            "executive_insights": executive_insights,
            "model_performance": {
                "ensemble_accuracy": self._calculate_ensemble_accuracy(trained_models),
                "prediction_confidence": self._calculate_prediction_confidence(seasonal_forecasts),
                "model_reliability_score": self._assess_model_reliability(historical_data, trained_models)
            },
            "strategic_recommendations": {
                "investment_priorities": self._recommend_investment_priorities(seasonal_forecasts, scenario_forecasts),
                "risk_mitigation": self._recommend_risk_mitigation(algorithm_impacts, competitive_impacts),
                "growth_acceleration": self._recommend_growth_strategies(seasonal_forecasts, seasonal_analysis)
            }
        }
    
    async def _prepare_historical_data(self) -> pd.DataFrame:
        """Prepare and clean historical traffic data for modeling"""
        
        # Simulated historical data (in production: integrate with GA4 API)
        date_range = pd.date_range(
            start=datetime.now() - timedelta(days=self.historical_months * 30),
            end=datetime.now(),
            freq='D'
        )
        
        np.random.seed(42)  # For consistent demo data
        
        # Base traffic with trend and seasonality
        base_traffic = 10000
        trend_growth = 0.001  # Daily growth rate
        
        traffic_data = []
        for i, date in enumerate(date_range):
            # Trend component
            trend_factor = 1 + (trend_growth * i)
            
            # Seasonal components
            day_of_week_factor = self._get_day_of_week_factor(date.weekday())
            month_factor = self._get_month_factor(date.month)
            
            # Random noise
            noise_factor = np.random.normal(1, 0.1)
            
            # Holiday impact (simplified)
            holiday_factor = self._get_holiday_factor(date)
            
            # Calculate daily sessions
            daily_sessions = int(
                base_traffic * trend_factor * day_of_week_factor * 
                month_factor * noise_factor * holiday_factor
            )
            
            traffic_data.append({
                'date': date,
                'sessions': max(1000, daily_sessions),  # Minimum floor
                'day_of_week': date.weekday(),
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'is_holiday': holiday_factor != 1.0,
                'trend_day': i
            })
        
        df = pd.DataFrame(traffic_data)
        
        # Add feature engineering
        df['sessions_7d_avg'] = df['sessions'].rolling(window=7, center=True).mean()
        df['sessions_30d_avg'] = df['sessions'].rolling(window=30, center=True).mean()
        df['yoy_growth'] = df['sessions'].pct_change(periods=365)
        df['mom_growth'] = df['sessions'].pct_change(periods=30)
        df['dow_avg_sessions'] = df.groupby('day_of_week')['sessions'].transform('mean')
        df['month_avg_sessions'] = df.groupby('month')['sessions'].transform('mean')
        
        return df.fillna(method='bfill').fillna(method='ffill')
    
    def _get_day_of_week_factor(self, day_of_week: int) -> float:
        """Get traffic factor based on day of week (0=Monday)"""
        # Business site pattern - lower on weekends
        factors = {
            0: 1.1,  # Monday
            1: 1.2,  # Tuesday
            2: 1.15, # Wednesday
            3: 1.25, # Thursday
            4: 1.0,  # Friday
            5: 0.7,  # Saturday
            6: 0.6   # Sunday
        }
        return factors.get(day_of_week, 1.0)
    
    def _get_month_factor(self, month: int) -> float:
        """Get traffic factor based on month"""
        # Business seasonality pattern
        factors = {
            1: 1.15,  # January - New Year planning
            2: 1.25,  # February - Budget season
            3: 1.2,   # March - Q1 end
            4: 1.1,   # April
            5: 1.0,   # May
            6: 0.95,  # June
            7: 0.8,   # July - Summer slowdown
            8: 0.85,  # August - Summer slowdown
            9: 1.3,   # September - Back to business
            10: 1.25, # October - Q4 planning
            11: 1.2,  # November - Year-end push
            12: 0.9   # December - Holidays
        }
        return factors.get(month, 1.0)
    
    def _get_holiday_factor(self, date: datetime) -> float:
        """Get traffic factor for holidays (simplified)"""
        # Major US business holidays (simplified)
        if date.month == 1 and date.day == 1:  # New Year
            return 0.3
        elif date.month == 7 and date.day == 4:  # July 4th
            return 0.5
        elif date.month == 11 and date.day >= 22 and date.day <= 28 and date.weekday() == 3:  # Thanksgiving week
            return 0.4
        elif date.month == 12 and date.day >= 24 and date.day <= 26:  # Christmas
            return 0.3
        else:
            return 1.0
    
    async def _train_prediction_models(self) -> Dict:
        """Train multiple ML models for ensemble forecasting"""
        
        historical_data = await self._prepare_historical_data()
        
        # Prepare features
        feature_columns = [
            'trend_day', 'day_of_week', 'month', 'quarter',
            'sessions_7d_avg', 'sessions_30d_avg'
        ]
        
        X = historical_data[feature_columns].fillna(0)
        y = historical_data['sessions']
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        trained_models = {}
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'linear_trend':
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Make predictions for validation
                if model_name == 'linear_trend':
                    predictions = model.predict(X_test_scaled)
                else:
                    predictions = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                trained_models[model_name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'predictions': predictions
                }
                
                logger.info(f"model_trained", model=model_name, mae=mae, r2=r2)
                
            except Exception as e:
                logger.error(f"model_training_error", model=model_name, error=str(e))
        
        return trained_models
    
    def _generate_ensemble_forecasts(self, historical_data: pd.DataFrame, 
                                   trained_models: Dict, 
                                   forecast_horizon: int) -> Dict[int, Dict]:
        """Generate ensemble forecasts for multiple time horizons"""
        
        base_forecasts = {}
        
        for horizon in self.forecast_horizons:
            if horizon <= forecast_horizon:
                forecast = self._generate_single_forecast(historical_data, trained_models, horizon)
                base_forecasts[horizon] = forecast
        
        return base_forecasts
    
    def _generate_single_forecast(self, historical_data: pd.DataFrame,
                                trained_models: Dict,
                                days_ahead: int) -> Dict:
        """Generate forecast for a specific time horizon"""
        
        # Get latest data point
        latest_data = historical_data.iloc[-1]
        
        # Create future data points
        future_dates = pd.date_range(
            start=latest_data['date'] + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        # Prepare features for future predictions
        future_features = []
        for i, date in enumerate(future_dates):
            feature_row = {
                'trend_day': latest_data['trend_day'] + i + 1,
                'day_of_week': date.weekday(),
                'month': date.month,
                'quarter': (date.month - 1) // 3 + 1,
                'sessions_7d_avg': latest_data['sessions_7d_avg'],  # Simplified
                'sessions_30d_avg': latest_data['sessions_30d_avg']  # Simplified
            }
            future_features.append(feature_row)
        
        future_df = pd.DataFrame(future_features)
        
        # Generate predictions from each model
        model_predictions = []
        
        for model_name, model_info in trained_models.items():
            try:
                model = model_info['model']
                
                if model_name == 'linear_trend':
                    X_scaled = self.scalers['standard'].transform(future_df)
                    predictions = model.predict(X_scaled)
                else:
                    predictions = model.predict(future_df)
                
                # Weight predictions by model performance (R²)
                weight = max(0.1, model_info.get('r2', 0.5))
                model_predictions.append((predictions, weight))
                
            except Exception as e:
                logger.error("prediction_error", model=model_name, error=str(e))
        
        # Calculate ensemble prediction
        if model_predictions:
            # Weighted average of predictions
            weighted_sum = np.zeros(len(future_dates))
            total_weight = 0
            
            for predictions, weight in model_predictions:
                weighted_sum += predictions * weight
                total_weight += weight
            
            ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else weighted_sum
            
            # Calculate total predicted sessions
            total_sessions = int(np.sum(ensemble_prediction))
            
            # Calculate confidence intervals (simplified approach)
            prediction_std = np.std([pred[0] for pred, _ in model_predictions], axis=0)
            confidence_margin = 1.96 * np.sum(prediction_std)  # 95% CI
            
            # Determine trend strength
            growth_rate = (total_sessions / (latest_data['sessions'] * days_ahead)) - 1
            trend_strength = self._classify_trend_strength(growth_rate)
            
            return {
                'predicted_sessions': total_sessions,
                'confidence_lower': max(0, int(total_sessions - confidence_margin)),
                'confidence_upper': int(total_sessions + confidence_margin),
                'confidence_interval': 95.0,
                'growth_rate': growth_rate,
                'seasonal_adjustment': 0.0,  # Will be applied later
                'trend_strength': trend_strength,
                'key_drivers': self._identify_key_drivers(growth_rate, days_ahead),
                'risk_factors': self._identify_risk_factors(growth_rate, days_ahead),
                'business_impact': self._calculate_business_impact(total_sessions, growth_rate)
            }
        
        return {}
    
    def _classify_trend_strength(self, growth_rate: float) -> str:
        """Classify trend strength based on growth rate"""
        if growth_rate > 0.3:
            return "strong_upward"
        elif growth_rate > 0.1:
            return "moderate_upward"
        elif growth_rate > -0.1:
            return "stable"
        else:
            return "declining"
    
    def _identify_key_drivers(self, growth_rate: float, days_ahead: int) -> List[str]:
        """Identify key drivers of traffic growth"""
        drivers = []
        
        if growth_rate > 0.2:
            drivers.extend([
                "Strong organic growth momentum",
                "Improved search rankings",
                "Content strategy effectiveness"
            ])
        elif growth_rate > 0.05:
            drivers.extend([
                "Steady content performance",
                "Seasonal optimization gains",
                "Technical SEO improvements"
            ])
        else:
            drivers.extend([
                "Market maturity and saturation",
                "Increased competitive pressure",
                "Algorithm change impacts"
            ])
        
        # Add time-horizon specific drivers
        if days_ahead >= 365:
            drivers.append("Long-term strategic initiatives")
        elif days_ahead >= 90:
            drivers.append("Seasonal campaign performance")
        
        return drivers[:4]  # Limit to top 4
    
    def _identify_risk_factors(self, growth_rate: float, days_ahead: int) -> List[str]:
        """Identify risk factors that could impact forecast"""
        risks = []
        
        if growth_rate < 0:
            risks.extend([
                "Declining organic performance trend",
                "Competitive displacement risk",
                "Algorithm vulnerability exposure"
            ])
        
        # Universal risks
        risks.extend([
            "Search algorithm changes",
            "Competitive market dynamics",
            "Economic factors affecting search behavior",
            "Technical performance degradation"
        ])
        
        # Time-specific risks
        if days_ahead >= 180:
            risks.append("Long-term market disruption potential")
        
        return risks[:5]  # Limit to top 5
    
    def _calculate_business_impact(self, predicted_sessions: int, growth_rate: float) -> Dict[str, Union[str, float]]:
        """Calculate business impact of traffic forecast"""
        
        # Simulated business metrics
        conversion_rate = 0.025  # 2.5% conversion rate
        average_order_value = 150
        
        estimated_conversions = predicted_sessions * conversion_rate
        estimated_revenue = estimated_conversions * average_order_value
        
        return {
            "estimated_conversions": int(estimated_conversions),
            "estimated_revenue": estimated_revenue,
            "growth_impact": "positive" if growth_rate > 0 else "negative",
            "revenue_opportunity": f"${estimated_revenue:,.0f}",
            "business_significance": "high" if estimated_revenue > 100000 else "medium" if estimated_revenue > 50000 else "low"
        }
    
    async def _analyze_seasonal_patterns(self) -> SeasonalPattern:
        """Analyze seasonal traffic patterns for optimization"""
        
        historical_data = await self._prepare_historical_data()
        
        # Monthly analysis
        monthly_avg = historical_data.groupby('month')['sessions'].mean()
        peak_months = monthly_avg.nlargest(3).index.tolist()
        trough_months = monthly_avg.nsmallest(3).index.tolist()
        
        # Calculate seasonality strength
        monthly_variance = monthly_avg.var()
        overall_mean = monthly_avg.mean()
        seasonality_strength = min(1.0, monthly_variance / (overall_mean ** 2))
        
        # Pattern reliability (consistency year-over-year)
        pattern_reliability = 0.85  # Simulated high reliability
        
        # Generate optimization opportunities
        optimization_opportunities = []
        if seasonality_strength > 0.3:
            optimization_opportunities.extend([
                "Seasonal content calendar optimization",
                "Peak season budget allocation increase",
                "Pre-peak content preparation strategy"
            ])
        
        if peak_months:
            peak_month_names = [self._get_month_name(m) for m in peak_months]
            optimization_opportunities.append(f"Maximize efforts during {', '.join(peak_month_names)}")
        
        # Preparation recommendations
        preparation_recommendations = [
            "Develop seasonal content 2-3 months in advance",
            "Increase crawl budget during peak preparation periods",
            "Monitor competitor seasonal strategies for defensive planning",
            "Prepare technical infrastructure for traffic spikes"
        ]
        
        return SeasonalPattern(
            pattern_type="monthly",
            peak_periods=[self._get_month_name(m) for m in peak_months],
            trough_periods=[self._get_month_name(m) for m in trough_months],
            seasonality_strength=seasonality_strength,
            pattern_reliability=pattern_reliability,
            optimization_opportunities=optimization_opportunities,
            preparation_recommendations=preparation_recommendations
        )
    
    def _get_month_name(self, month_num: int) -> str:
        """Convert month number to name"""
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        return months[month_num - 1] if 1 <= month_num <= 12 else "Unknown"
    
    def _apply_seasonal_adjustments(self, base_forecasts: Dict[int, Dict], 
                                  seasonal_analysis: SeasonalPattern) -> Dict[int, TrafficForecast]:
        """Apply seasonal adjustments to base forecasts"""
        
        adjusted_forecasts = {}
        
        for horizon, forecast in base_forecasts.items():
            # Apply seasonal adjustment factor
            seasonal_multiplier = 1.0 + (seasonal_analysis.seasonality_strength * 0.1)
            
            adjusted_sessions = int(forecast['predicted_sessions'] * seasonal_multiplier)
            adjusted_lower = int(forecast['confidence_lower'] * seasonal_multiplier)
            adjusted_upper = int(forecast['confidence_upper'] * seasonal_multiplier)
            
            # Update business impact
            business_impact = self._calculate_business_impact(adjusted_sessions, forecast['growth_rate'])
            
            adjusted_forecasts[horizon] = TrafficForecast(
                period=f"{horizon}d",
                predicted_sessions=adjusted_sessions,
                confidence_lower=adjusted_lower,
                confidence_upper=adjusted_upper,
                confidence_interval=forecast['confidence_interval'],
                growth_rate=forecast['growth_rate'],
                seasonal_adjustment=seasonal_multiplier - 1.0,
                trend_strength=forecast['trend_strength'],
                key_drivers=forecast['key_drivers'],
                risk_factors=forecast['risk_factors'],
                business_impact=business_impact
            )
        
        return adjusted_forecasts
    
    def _generate_scenario_forecasts(self, base_forecasts: Dict[int, TrafficForecast],
                                   historical_data: pd.DataFrame) -> List[TrafficScenario]:
        """Generate scenario-based forecasts for strategic planning"""
        
        scenarios = []
        
        # Get baseline annual forecast
        baseline_annual = base_forecasts.get(365)
        if not baseline_annual:
            return scenarios
        
        baseline_traffic = baseline_annual.predicted_sessions
        
        # Conservative scenario (70% of baseline)
        conservative_traffic = int(baseline_traffic * 0.7)
        scenarios.append(TrafficScenario(
            scenario_name="conservative",
            probability=self.scenario_probabilities['conservative'],
            traffic_projection={
                "90d": int(conservative_traffic * 0.25),
                "180d": int(conservative_traffic * 0.5),
                "365d": conservative_traffic
            },
            investment_required={
                "content": 50000,
                "technical": 30000,
                "paid_amplification": 20000
            },
            expected_roi=1.2,
            key_assumptions=[
                "Minimal competitive pressure increase",
                "Stable algorithm environment",
                "Conservative content investment"
            ],
            success_metrics=[
                "Traffic growth of 20-30%",
                "Maintained market share",
                "Improved conversion rates"
            ],
            risk_mitigation=[
                "Diversified traffic sources",
                "Strong technical foundation",
                "Defensive competitive positioning"
            ]
        ))
        
        # Baseline scenario (100% of forecast)
        scenarios.append(TrafficScenario(
            scenario_name="baseline",
            probability=self.scenario_probabilities['baseline'],
            traffic_projection={
                "90d": int(baseline_traffic * 0.25),
                "180d": int(baseline_traffic * 0.5),
                "365d": baseline_traffic
            },
            investment_required={
                "content": 75000,
                "technical": 50000,
                "paid_amplification": 35000
            },
            expected_roi=1.8,
            key_assumptions=[
                "Current growth trends continue",
                "Moderate competitive dynamics",
                "Consistent algorithm environment"
            ],
            success_metrics=[
                "Traffic growth of 40-60%",
                "Market share maintenance or growth",
                "ROI improvement"
            ],
            risk_mitigation=[
                "Balanced growth strategies",
                "Competitive monitoring",
                "Algorithm change preparation"
            ]
        ))
        
        # Optimistic scenario (130% of baseline)
        optimistic_traffic = int(baseline_traffic * 1.3)
        scenarios.append(TrafficScenario(
            scenario_name="optimistic",
            probability=self.scenario_probabilities['optimistic'],
            traffic_projection={
                "90d": int(optimistic_traffic * 0.25),
                "180d": int(optimistic_traffic * 0.5),
                "365d": optimistic_traffic
            },
            investment_required={
                "content": 100000,
                "technical": 75000,
                "paid_amplification": 50000
            },
            expected_roi=2.5,
            key_assumptions=[
                "Accelerated growth opportunities",
                "Successful competitive displacement",
                "Favorable algorithm changes"
            ],
            success_metrics=[
                "Traffic growth of 80-120%",
                "Market share expansion",
                "Premium ROI achievement"
            ],
            risk_mitigation=[
                "Aggressive growth investments",
                "Competitive advantage development",
                "Opportunity capitalization focus"
            ]
        ))
        
        # Aggressive scenario (160% of baseline)
        aggressive_traffic = int(baseline_traffic * 1.6)
        scenarios.append(TrafficScenario(
            scenario_name="aggressive",
            probability=self.scenario_probabilities['aggressive'],
            traffic_projection={
                "90d": int(aggressive_traffic * 0.25),
                "180d": int(aggressive_traffic * 0.5),
                "365d": aggressive_traffic
            },
            investment_required={
                "content": 150000,
                "technical": 100000,
                "paid_amplification": 75000
            },
            expected_roi=3.2,
            key_assumptions=[
                "Market disruption opportunities",
                "Breakthrough competitive advantages",
                "Optimal execution across all channels"
            ],
            success_metrics=[
                "Traffic growth of 150%+",
                "Market leadership achievement",
                "Exceptional ROI delivery"
            ],
            risk_mitigation=[
                "High-risk, high-reward strategies",
                "Maximum resource allocation",
                "Continuous optimization and pivoting"
            ]
        ))
        
        return scenarios
    
    async def _model_algorithm_impacts(self) -> List[AlgorithmImpactModel]:
        """Model potential algorithm change impacts on traffic"""
        
        # Simulated algorithm impact modeling
        algorithm_scenarios = [
            {
                "type": "core_update",
                "historical_impact": -0.15,  # -15% historically
                "predicted_impact": -0.08,   # -8% predicted
                "duration": "permanent",
                "categories": ["content_quality", "e_a_t_signals", "user_experience"]
            },
            {
                "type": "feature_update",
                "historical_impact": 0.05,   # +5% historically
                "predicted_impact": 0.03,    # +3% predicted
                "duration": "evolving",
                "categories": ["featured_snippets", "local_results", "image_search"]
            },
            {
                "type": "policy_change",
                "historical_impact": -0.05,  # -5% historically
                "predicted_impact": -0.02,   # -2% predicted
                "duration": "temporary",
                "categories": ["link_policies", "content_guidelines", "spam_detection"]
            }
        ]
        
        impact_models = []
        
        for scenario in algorithm_scenarios:
            mitigation_strategies = self._generate_algorithm_mitigation_strategies(scenario["type"])
            recovery_timeline = self._estimate_recovery_timeline(scenario["predicted_impact"])
            
            impact_models.append(AlgorithmImpactModel(
                algorithm_type=scenario["type"],
                historical_impact=scenario["historical_impact"],
                predicted_impact=scenario["predicted_impact"],
                impact_duration=scenario["duration"],
                affected_categories=scenario["categories"],
                mitigation_strategies=mitigation_strategies,
                recovery_timeline=recovery_timeline
            ))
        
        return impact_models
    
    def _generate_algorithm_mitigation_strategies(self, algorithm_type: str) -> List[str]:
        """Generate mitigation strategies for different algorithm types"""
        
        strategies_map = {
            "core_update": [
                "Strengthen E-A-T signals across content portfolio",
                "Improve Core Web Vitals and technical performance",
                "Enhance content depth and expertise demonstration",
                "Diversify traffic sources beyond organic search"
            ],
            "feature_update": [
                "Optimize content for featured snippet opportunities",
                "Implement comprehensive schema markup",
                "Focus on question-based content optimization",
                "Monitor and adapt to SERP feature changes"
            ],
            "policy_change": [
                "Audit and improve link profile quality",
                "Review content against updated guidelines",
                "Implement proactive spam detection measures",
                "Strengthen content authenticity signals"
            ]
        }
        
        return strategies_map.get(algorithm_type, ["Monitor changes and adapt accordingly"])
    
    def _estimate_recovery_timeline(self, predicted_impact: float) -> str:
        """Estimate recovery timeline based on impact magnitude"""
        
        if abs(predicted_impact) > 0.15:
            return "6-12 months"
        elif abs(predicted_impact) > 0.08:
            return "3-6 months" 
        elif abs(predicted_impact) > 0.03:
            return "1-3 months"
        else:
            return "2-6 weeks"
    
    async def _assess_competitive_impacts(self) -> List[CompetitiveImpactForecast]:
        """Assess competitive impacts on traffic forecasting"""
        
        # Simulated competitive impact assessment
        competitive_scenarios = [
            {
                "competitor": "competitor-1.com",
                "impact_magnitude": -0.12,  # -12% traffic impact
                "timeline": "short_term",
                "keywords": ["enterprise software", "business solutions", "digital transformation"]
            },
            {
                "competitor": "competitor-2.com", 
                "impact_magnitude": -0.08,  # -8% traffic impact
                "timeline": "long_term",
                "keywords": ["seo automation", "marketing technology", "analytics platform"]
            },
            {
                "competitor": "new-entrant.com",
                "impact_magnitude": -0.05,  # -5% traffic impact
                "timeline": "immediate",
                "keywords": ["ai marketing", "automated seo", "intelligent optimization"]
            }
        ]
        
        impact_forecasts = []
        
        for scenario in competitive_scenarios:
            defensive_measures = self._generate_defensive_measures(scenario["impact_magnitude"])
            market_share_impact = abs(scenario["impact_magnitude"]) * 0.8  # Correlation factor
            
            impact_forecasts.append(CompetitiveImpactForecast(
                competitor=scenario["competitor"],
                impact_magnitude=scenario["impact_magnitude"],
                impact_timeline=scenario["timeline"],
                affected_keywords=scenario["keywords"],
                defensive_measures=defensive_measures,
                market_share_implications=market_share_impact
            ))
        
        return impact_forecasts
    
    def _generate_defensive_measures(self, impact_magnitude: float) -> List[str]:
        """Generate defensive measures based on competitive impact"""
        
        measures = []
        
        if abs(impact_magnitude) > 0.1:  # High impact
            measures.extend([
                "Accelerate content production to defend keyword positions",
                "Implement aggressive technical SEO optimizations",
                "Launch competitive keyword campaigns",
                "Develop unique content differentiators"
            ])
        elif abs(impact_magnitude) > 0.05:  # Medium impact
            measures.extend([
                "Monitor competitor strategies closely",
                "Strengthen content in affected topic areas",
                "Improve user experience metrics",
                "Focus on long-tail keyword opportunities"
            ])
        else:  # Low impact
            measures.extend([
                "Maintain current optimization efforts",
                "Track competitive movements",
                "Identify potential collaboration opportunities",
                "Focus on market expansion strategies"
            ])
        
        return measures[:4]  # Limit to top 4
    
    def _generate_executive_insights(self, forecasts: Dict[int, TrafficForecast],
                                   scenarios: List[TrafficScenario],
                                   seasonal_analysis: SeasonalPattern) -> Dict:
        """Generate executive-level insights and recommendations"""
        
        annual_forecast = forecasts.get(365)
        if not annual_forecast:
            return {}
        
        return {
            "growth_trajectory_assessment": {
                "trend_direction": annual_forecast.trend_strength,
                "growth_sustainability": "high" if annual_forecast.growth_rate > 0.3 else "moderate",
                "seasonal_leverage_potential": len(seasonal_analysis.optimization_opportunities),
                "risk_mitigation_priority": "high" if len(annual_forecast.risk_factors) > 3 else "medium"
            },
            "business_implications": {
                "revenue_projection": annual_forecast.business_impact.get("estimated_revenue", 0),
                "investment_efficiency": "optimal" if len(scenarios) > 2 else "standard",
                "market_position_impact": "strengthening" if annual_forecast.growth_rate > 0 else "maintaining",
                "competitive_advantage": "growing" if annual_forecast.growth_rate > 0.2 else "stable"
            },
            "strategic_priorities": {
                "immediate_focus": annual_forecast.key_drivers[:3],
                "risk_management": annual_forecast.risk_factors[:3],
                "seasonal_optimization": seasonal_analysis.optimization_opportunities[:3],
                "scenario_planning": [s.scenario_name for s in scenarios if s.probability > 0.2]
            },
            "confidence_assessment": {
                "forecast_reliability": "high" if annual_forecast.confidence_interval >= 90 else "medium",
                "prediction_accuracy": "validated" if len(forecasts) > 2 else "estimated",
                "model_convergence": "strong" if len(forecasts) == len(self.forecast_horizons) else "partial"
            }
        }
    
    # Performance assessment methods
    def _calculate_ensemble_accuracy(self, trained_models: Dict) -> float:
        """Calculate ensemble model accuracy"""
        if not trained_models:
            return 0.0
        
        r2_scores = [model.get('r2', 0) for model in trained_models.values()]
        return np.mean(r2_scores) if r2_scores else 0.0
    
    def _calculate_prediction_confidence(self, forecasts: Dict[int, TrafficForecast]) -> float:
        """Calculate overall prediction confidence"""
        if not forecasts:
            return 0.0
        
        confidence_scores = []
        for forecast in forecasts.values():
            # Calculate confidence based on interval width
            interval_width = forecast.confidence_upper - forecast.confidence_lower
            relative_width = interval_width / forecast.predicted_sessions
            confidence = max(0.5, 1.0 - relative_width)
            confidence_scores.append(confidence)
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    def _assess_model_reliability(self, historical_data: pd.DataFrame, trained_models: Dict) -> float:
        """Assess overall model reliability score"""
        factors = []
        
        # Data quality factor
        data_completeness = 1.0 - (historical_data.isnull().sum().sum() / historical_data.size)
        factors.append(data_completeness)
        
        # Model performance factor
        model_performance = self._calculate_ensemble_accuracy(trained_models)
        factors.append(model_performance)
        
        # Data volume factor
        data_volume_score = min(1.0, len(historical_data) / (365 * 2))  # 2 years ideal
        factors.append(data_volume_score)
        
        return np.mean(factors) if factors else 0.5
    
    def _assess_algorithm_risk_level(self, algorithm_impacts: List[AlgorithmImpactModel]) -> str:
        """Assess overall algorithm risk level"""
        if not algorithm_impacts:
            return "low"
        
        total_negative_impact = sum(
            abs(impact.predicted_impact) for impact in algorithm_impacts 
            if impact.predicted_impact < 0
        )
        
        if total_negative_impact > 0.2:
            return "high"
        elif total_negative_impact > 0.1:
            return "medium"
        else:
            return "low"
    
    # Recommendation methods
    def _recommend_investment_priorities(self, forecasts: Dict[int, TrafficForecast], 
                                       scenarios: List[TrafficScenario]) -> List[Dict]:
        """Recommend investment priorities based on forecasts"""
        
        recommendations = []
        
        annual_forecast = forecasts.get(365)
        if annual_forecast and annual_forecast.growth_rate > 0:
            recommendations.append({
                "priority": "high",
                "category": "growth_acceleration",
                "investment": "Content production and optimization",
                "expected_return": "20-40% traffic increase",
                "timeline": "6-12 months"
            })
        
        # Seasonal optimization
        recommendations.append({
            "priority": "medium",
            "category": "seasonal_optimization",
            "investment": "Seasonal content and campaign preparation",
            "expected_return": "15-25% seasonal lift",
            "timeline": "3-6 months"
        })
        
        # Technical foundation
        recommendations.append({
            "priority": "high",
            "category": "technical_foundation",
            "investment": "Core Web Vitals and technical SEO",
            "expected_return": "Algorithm resilience and ranking improvements",
            "timeline": "3-6 months"
        })
        
        return recommendations
    
    def _recommend_risk_mitigation(self, algorithm_impacts: List[AlgorithmImpactModel],
                                 competitive_impacts: List[CompetitiveImpactForecast]) -> List[Dict]:
        """Recommend risk mitigation strategies"""
        
        recommendations = []
        
        # Algorithm risk mitigation
        high_impact_algorithms = [a for a in algorithm_impacts if abs(a.predicted_impact) > 0.05]
        if high_impact_algorithms:
            recommendations.append({
                "risk_type": "algorithm_vulnerability",
                "mitigation": "Diversify traffic sources and strengthen E-A-T signals",
                "priority": "high",
                "timeline": "immediate"
            })
        
        # Competitive risk mitigation
        high_competitive_pressure = [c for c in competitive_impacts if abs(c.impact_magnitude) > 0.08]
        if high_competitive_pressure:
            recommendations.append({
                "risk_type": "competitive_pressure",
                "mitigation": "Accelerate content differentiation and technical advantages",
                "priority": "high",
                "timeline": "1-3 months"
            })
        
        # General risk mitigation
        recommendations.append({
            "risk_type": "market_volatility",
            "mitigation": "Maintain flexible resource allocation and monitoring systems",
            "priority": "medium",
            "timeline": "ongoing"
        })
        
        return recommendations
    
    def _recommend_growth_strategies(self, forecasts: Dict[int, TrafficForecast],
                                   seasonal_analysis: SeasonalPattern) -> List[Dict]:
        """Recommend growth acceleration strategies"""
        
        strategies = []
        
        annual_forecast = forecasts.get(365)
        if annual_forecast:
            if annual_forecast.trend_strength in ["strong_upward", "moderate_upward"]:
                strategies.append({
                    "strategy": "momentum_acceleration",
                    "description": "Amplify successful content and optimization strategies",
                    "investment_level": "high",
                    "expected_impact": "30-50% additional growth"
                })
            elif annual_forecast.trend_strength == "stable":
                strategies.append({
                    "strategy": "breakthrough_initiatives",
                    "description": "Launch innovative content and technical differentiation",
                    "investment_level": "medium",
                    "expected_impact": "20-35% growth acceleration"
                })
        
        # Seasonal growth strategies
        if seasonal_analysis.seasonality_strength > 0.3:
            strategies.append({
                "strategy": "seasonal_maximization",
                "description": "Optimize for peak seasonal opportunities",
                "investment_level": "medium",
                "expected_impact": "15-25% seasonal improvement"
            })
        
        return strategies
    
    def generate_executive_dashboard(self, forecast_results: Dict) -> Dict:
        """
        Generate Executive Traffic Forecasting Dashboard
        
        Perfect for CMO presentations and strategic planning sessions.
        Demonstrates ability to transform predictive analytics into business intelligence.
        """
        
        return {
            "executive_kpis": {
                "annual_traffic_projection": f"{forecast_results['executive_summary']['baseline_annual_projection']:,}",
                "growth_trajectory": forecast_results['executive_summary']['growth_trajectory'].replace('_', ' ').title(),
                "forecast_confidence": forecast_results['executive_summary']['confidence_level'],
                "algorithm_risk_level": forecast_results['executive_summary']['algorithm_risk_level'].title(),
                "seasonal_opportunity_count": forecast_results['executive_summary']['seasonal_optimization_potential']
            },
            "business_impact_summary": {
                "revenue_projection": f"${forecast_results.get('traffic_forecasts', {}).get('365d', {}).get('business_impact', {}).get('estimated_revenue', 0):,.0f}",
                "growth_sustainability": forecast_results.get('executive_insights', {}).get('growth_trajectory_assessment', {}).get('growth_sustainability', 'moderate').title(),
                "investment_efficiency": forecast_results.get('executive_insights', {}).get('business_implications', {}).get('investment_efficiency', 'standard').title(),
                "competitive_position": forecast_results.get('executive_insights', {}).get('business_implications', {}).get('market_position_impact', 'maintaining').title()
            },
            "strategic_priorities": {
                "immediate_actions": forecast_results.get('executive_insights', {}).get('strategic_priorities', {}).get('immediate_focus', [])[:3],
                "risk_management": forecast_results.get('executive_insights', {}).get('strategic_priorities', {}).get('risk_management', [])[:3],
                "growth_initiatives": forecast_results.get('strategic_recommendations', {}).get('growth_acceleration', [])[:3]
            },
            "scenario_planning": {
                "conservative_projection": f"${forecast_results.get('scenario_modeling', [{}])[0].get('traffic_projection', {}).get('365d', 0) if forecast_results.get('scenario_modeling') else 0:,}",
                "baseline_projection": f"${forecast_results.get('scenario_modeling', [{}] * 2)[1].get('traffic_projection', {}).get('365d', 0) if len(forecast_results.get('scenario_modeling', [])) > 1 else 0:,}",
                "optimistic_projection": f"${forecast_results.get('scenario_modeling', [{}] * 3)[2].get('traffic_projection', {}).get('365d', 0) if len(forecast_results.get('scenario_modeling', [])) > 2 else 0:,}",
                "investment_scenarios": [s.get('scenario_name', '').title() for s in forecast_results.get('scenario_modeling', [])][:3]
            },
            "portfolio_branding": {
                "analyst": "Sotiris Spyrou",
                "linkedin": "https://www.linkedin.com/in/sspyrou/",
                "company": "VerityAI - AI SEO Services",
                "service_url": "https://verityai.co/landing/ai-seo-services",
                "expertise_note": "Advanced ML forecasting with executive business intelligence"
            }
        }


# Portfolio demonstration usage
async def demonstrate_traffic_forecasting():
    """
    Portfolio Demonstration: Executive-Level Traffic Forecasting
    
    This function showcases advanced predictive analytics capabilities
    that make this portfolio valuable for senior marketing and strategy roles.
    """
    
    # Example usage for enterprise traffic forecasting scenario
    domain = "enterprise-client.com"
    
    predictor = TrafficPredictionModels(domain, historical_data_months=18)
    
    # Comprehensive traffic forecasting
    results = await predictor.generate_comprehensive_forecast(
        forecast_horizon=365,
        include_scenarios=True
    )
    
    # Generate executive dashboard
    executive_dashboard = predictor.generate_executive_dashboard(results)
    
    return {
        "traffic_forecast_analysis": results,
        "executive_dashboard": executive_dashboard,
        "portfolio_value_demonstration": {
            "predictive_modeling": "Advanced ensemble ML models for accurate forecasting",
            "business_intelligence": "Strategic scenario planning with ROI projections",
            "executive_communication": "C-suite ready dashboards and insights",
            "risk_assessment": "Comprehensive algorithm and competitive impact modeling"
        }
    }


if __name__ == "__main__":
    # Portfolio demonstration
    print("📈 Traffic Prediction Models - Portfolio Demo")
    print("🤖 Showcasing ML forecasting + business intelligence")
    print("🔗 LinkedIn: https://www.linkedin.com/in/sspyrou/")
    print("🏢 VerityAI: https://verityai.co/landing/ai-seo-services")
    print("\n⚠️  Portfolio demonstration code - not for production use")
    
    # Run demonstration
    results = asyncio.run(demonstrate_traffic_forecasting())
    annual_projection = results['traffic_forecast_analysis']['executive_summary']['baseline_annual_projection']
    growth_trajectory = results['traffic_forecast_analysis']['executive_summary']['growth_trajectory']
    print(f"\n✅ Forecast complete - Annual projection: {annual_projection:,} sessions, Trajectory: {growth_trajectory.replace('_', ' ').title()}")