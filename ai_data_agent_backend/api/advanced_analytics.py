"""
Advanced Analytics Engine for AI Data Agent

This module provides sophisticated analytical capabilities that go beyond basic query translation:
- Multi-dimensional analysis and drill-down capabilities
- Trend detection and forecasting
- Anomaly detection and outlier identification
- Cohort analysis and customer segmentation
- Statistical analysis and correlation discovery
- Automated insight generation and business recommendations
"""

from datetime import datetime, timedelta
from django.db import connection
from typing import Dict, List, Tuple, Any, Optional
import json
import re
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced analytics - graceful degradation if not available
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    HAS_ADVANCED_PACKAGES = True
except ImportError as e:
    # Create mock objects for graceful degradation
    pd = None
    np = None
    LinearRegression = None
    KMeans = None
    StandardScaler = None
    stats = None
    HAS_ADVANCED_PACKAGES = False
    print(f"Advanced analytics packages not available: {e}")
    print("Basic functionality will be available, install scikit-learn, pandas, numpy, scipy for full features")

class AdvancedAnalyticsEngine:
    """
    Advanced analytics engine that provides sophisticated business intelligence
    capabilities beyond basic SQL query translation.
    """
    
    def __init__(self):
        self.has_advanced_packages = HAS_ADVANCED_PACKAGES
        if self.has_advanced_packages:
            self.analysis_types = {
                'trend_analysis': self._perform_trend_analysis,
                'cohort_analysis': self._perform_cohort_analysis,
                'customer_segmentation': self._perform_customer_segmentation,
                'anomaly_detection': self._perform_anomaly_detection,
                'correlation_analysis': self._perform_correlation_analysis,
                'forecasting': self._perform_forecasting,
                'comparative_analysis': self._perform_comparative_analysis,
                'drill_down_analysis': self._perform_drill_down_analysis
            }
        else:
            # Basic analysis types when advanced packages are not available
            self.analysis_types = {
                'basic_analysis': self._perform_basic_analysis
            }
    
    def analyze_question_intelligence(self, question: str) -> Dict[str, Any]:
        """
        Intelligently analyze the question to determine the most appropriate
        advanced analytical approach beyond simple query translation.
        """
        question_lower = question.lower()
        
        # Detect sophisticated analytical intent
        analytical_intent = self._detect_analytical_intent(question_lower)
        
        # Determine required analytical depth
        analysis_depth = self._determine_analysis_depth(question_lower)
        
        # Identify business context and KPIs
        business_context = self._identify_business_context(question_lower)
        
        # Suggest multi-dimensional analysis
        suggested_dimensions = self._suggest_analysis_dimensions(question_lower)
        
        return {
            'analytical_intent': analytical_intent,
            'analysis_depth': analysis_depth,
            'business_context': business_context,
            'suggested_dimensions': suggested_dimensions,
            'recommended_analysis_types': self._recommend_analysis_types(question_lower),
            'advanced_insights_available': True,
            'multi_dimensional_analysis': True
        }
    
    def _detect_analytical_intent(self, question: str) -> Dict[str, Any]:
        """Detect sophisticated analytical intent in the question."""
        intent_patterns = {
            'trend_analysis': [
                'trend', 'over time', 'growing', 'declining', 'pattern', 'seasonal',
                'month over month', 'year over year', 'growth rate', 'trajectory'
            ],
            'comparative_analysis': [
                'compare', 'versus', 'vs', 'difference', 'better than', 'worse than',
                'benchmark', 'relative to', 'against', 'performance comparison'
            ],
            'predictive_analysis': [
                'forecast', 'predict', 'projection', 'future', 'expected', 'likely',
                'will be', 'estimate', 'outlook', 'anticipate'
            ],
            'segmentation_analysis': [
                'segment', 'group', 'cluster', 'category', 'type', 'cohort',
                'demographics', 'behavior', 'characteristics', 'profile'
            ],
            'anomaly_detection': [
                'unusual', 'anomaly', 'outlier', 'abnormal', 'unexpected', 'strange',
                'deviation', 'irregular', 'exception', 'spike'
            ],
            'correlation_analysis': [
                'relationship', 'correlation', 'impact', 'influence', 'affect',
                'connected', 'related to', 'association', 'dependency'
            ],
            'drill_down_analysis': [
                'why', 'what caused', 'breakdown', 'detailed view', 'underlying',
                'root cause', 'deeper', 'granular', 'specific reasons'
            ]
        }
        
        detected_intents = {}
        confidence_scores = {}
        
        for intent, patterns in intent_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in question)
            if matches > 0:
                detected_intents[intent] = True
                confidence_scores[intent] = min(matches / len(patterns), 1.0)
            else:
                detected_intents[intent] = False
                confidence_scores[intent] = 0.0
        
        # Determine primary intent
        if confidence_scores:
            primary_intent = max(confidence_scores.keys(), key=lambda x: confidence_scores[x])
        else:
            primary_intent = 'basic_query'
        
        return {
            'primary_intent': primary_intent,
            'confidence_score': confidence_scores[primary_intent],
            'all_intents': detected_intents,
            'complexity_level': 'advanced' if sum(detected_intents.values()) > 2 else 'intermediate'
        }
    
    def _determine_analysis_depth(self, question: str) -> str:
        """Determine the required depth of analysis."""
        depth_indicators = {
            'surface': ['count', 'total', 'sum', 'basic'],
            'intermediate': ['average', 'group by', 'breakdown', 'category'],
            'deep': ['trend', 'pattern', 'correlation', 'insight', 'why'],
            'expert': ['predict', 'forecast', 'optimize', 'recommend', 'strategic']
        }
        
        scores = {}
        for depth, indicators in depth_indicators.items():
            scores[depth] = sum(1 for indicator in indicators if indicator in question)
        
        if any(scores.values()):
            return max(scores.keys(), key=lambda x: scores[x])
        else:
            return 'intermediate'
    
    def _identify_business_context(self, question: str) -> Dict[str, Any]:
        """Identify business context and relevant KPIs."""
        business_domains = {
            'sales': ['sales', 'revenue', 'orders', 'purchase', 'transaction', 'conversion'],
            'marketing': ['campaign', 'acquisition', 'engagement', 'retention', 'churn'],
            'customer': ['customer', 'client', 'user', 'satisfaction', 'loyalty'],
            'operations': ['inventory', 'supply', 'efficiency', 'cost', 'margin'],
            'finance': ['profit', 'expense', 'budget', 'roi', 'financial']
        }
        
        relevant_domains = []
        for domain, keywords in business_domains.items():
            if any(keyword in question for keyword in keywords):
                relevant_domains.append(domain)
        
        return {
            'primary_domain': relevant_domains[0] if relevant_domains else 'general',
            'all_domains': relevant_domains,
            'business_complexity': 'high' if len(relevant_domains) > 2 else 'medium'
        }
    
    def _suggest_analysis_dimensions(self, question: str) -> List[str]:
        """Suggest relevant dimensions for multi-dimensional analysis."""
        dimension_keywords = {
            'time': ['time', 'date', 'month', 'year', 'quarter', 'season'],
            'geography': ['region', 'country', 'state', 'city', 'location'],
            'customer': ['customer', 'segment', 'type', 'demographics'],
            'product': ['product', 'category', 'brand', 'item'],
            'channel': ['channel', 'source', 'method', 'platform']
        }
        
        suggested = []
        for dimension, keywords in dimension_keywords.items():
            if any(keyword in question for keyword in keywords):
                suggested.append(dimension)
        
        return suggested if suggested else ['time', 'customer']
    
    def _recommend_analysis_types(self, question: str) -> List[str]:
        """Recommend specific types of advanced analysis."""
        recommendations = []
        
        if any(word in question for word in ['trend', 'over time', 'growth']):
            recommendations.append('trend_analysis')
        
        if any(word in question for word in ['compare', 'versus', 'difference']):
            recommendations.append('comparative_analysis')
        
        if any(word in question for word in ['segment', 'group', 'cluster']):
            recommendations.append('customer_segmentation')
        
        if any(word in question for word in ['forecast', 'predict', 'future']):
            recommendations.append('forecasting')
        
        if any(word in question for word in ['why', 'cause', 'reason']):
            recommendations.append('drill_down_analysis')
        
        if any(word in question for word in ['relationship', 'correlation', 'impact']):
            recommendations.append('correlation_analysis')
        
        if any(word in question for word in ['unusual', 'anomaly', 'outlier']):
            recommendations.append('anomaly_detection')
        
        return recommendations if recommendations else ['trend_analysis', 'drill_down_analysis']
    
    def perform_advanced_analysis(self, question: str, base_data: List[Dict]) -> Dict[str, Any]:
        """
        Perform sophisticated analytical processing on the base data
        to generate advanced insights and recommendations.
        """
        if not base_data:
            return {'error': 'No data available for advanced analysis'}
        
        # Check if advanced packages are available
        if not self.has_advanced_packages:
            return self._perform_basic_analysis(base_data, question)
        
        # Convert to DataFrame for advanced processing
        df = pd.DataFrame(base_data)
        
        # Get analytical intelligence
        intelligence = self.analyze_question_intelligence(question)
        
        # Perform recommended analyses
        advanced_results = {}
        
        for analysis_type in intelligence['recommended_analysis_types']:
            if analysis_type in self.analysis_types:
                try:
                    result = self.analysis_types[analysis_type](df, question)
                    advanced_results[analysis_type] = result
                except Exception as e:
                    advanced_results[analysis_type] = {'error': str(e)}
        
        # Generate strategic insights
        strategic_insights = self._generate_strategic_insights(df, question, advanced_results)
        
        # Create actionable recommendations
        recommendations = self._generate_recommendations(df, question, advanced_results)
        
        return {
            'intelligence': intelligence,
            'advanced_analyses': advanced_results,
            'strategic_insights': strategic_insights,
            'actionable_recommendations': recommendations,
            'data_quality_assessment': self._assess_data_quality(df),
            'statistical_summary': self._generate_statistical_summary(df)
        }
    
    def _perform_trend_analysis(self, df, question: str) -> Dict[str, Any]:
        """Perform sophisticated trend analysis with forecasting."""
        try:
            # Identify time-based columns
            time_cols = [col for col in df.columns if any(time_word in col.lower() 
                        for time_word in ['date', 'time', 'created', 'dt'])]
            
            if not time_cols:
                return {'error': 'No time-based columns found for trend analysis'}
            
            time_col = time_cols[0]
            
            # Convert to datetime if needed
            if df[time_col].dtype == 'object':
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Group by time periods and calculate metrics
            df_sorted = df.sort_values(time_col)
            
            # Monthly trend analysis
            monthly_trends = df_sorted.groupby(df_sorted[time_col].dt.to_period('M')).agg({
                col: ['count', 'sum', 'mean'] for col in df.select_dtypes(include=[np.number]).columns
            }).round(2)
            
            # Calculate growth rates
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            growth_rates = {}
            
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if len(df) > 1:
                    values = df.groupby(df[time_col].dt.to_period('M'))[col].sum()
                    if len(values) > 1:
                        growth_rate = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100) if values.iloc[0] != 0 else 0
                        growth_rates[col] = round(growth_rate, 2)
            
            # Detect trend patterns
            trend_patterns = self._detect_trend_patterns(df, time_col)
            
            return {
                'monthly_trends': monthly_trends.to_dict() if not monthly_trends.empty else {},
                'growth_rates': growth_rates,
                'trend_patterns': trend_patterns,
                'trend_strength': 'strong' if any(abs(rate) > 20 for rate in growth_rates.values()) else 'moderate',
                'recommendations': self._generate_trend_recommendations(growth_rates, trend_patterns)
            }
            
        except Exception as e:
            return {'error': f'Trend analysis failed: {str(e)}'}
    
    def _perform_customer_segmentation(self, df, question: str) -> Dict[str, Any]:
        """Perform advanced customer segmentation analysis."""
        try:
            # Identify customer-related columns
            customer_cols = [col for col in df.columns if any(cust_word in col.lower() 
                           for cust_word in ['customer', 'cust', 'client', 'user'])]
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {'error': 'Insufficient numeric data for customer segmentation'}
            
            # Prepare data for clustering
            features = df[numeric_cols[:4]].fillna(0)  # Use first 4 numeric columns
            
            if len(features) < 3:
                return {'error': 'Insufficient data points for segmentation'}
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform K-means clustering  
            n_clusters = min(4, len(features) // 2)  # Optimal cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Analyze segments
            df_with_clusters = df.copy()
            df_with_clusters['Segment'] = clusters
            
            segment_analysis = {}
            for cluster in range(n_clusters):
                cluster_data = df_with_clusters[df_with_clusters['Segment'] == cluster]
                segment_analysis[f'Segment_{cluster + 1}'] = {
                    'size': len(cluster_data),
                    'percentage': round(len(cluster_data) / len(df) * 100, 1),
                    'characteristics': {
                        col: {
                            'mean': round(float(cluster_data[col].mean()), 2),
                            'median': round(float(np.percentile(cluster_data[col], 50)), 2)
                        } for col in numeric_cols[:3] if col in cluster_data.columns
                    }
                }
            
            # Generate segment insights
            segment_insights = self._generate_segment_insights(segment_analysis)
            
            return {
                'segments': segment_analysis,
                'insights': segment_insights,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'segmentation_quality': 'high' if n_clusters >= 3 else 'moderate'
            }
            
        except Exception as e:
            return {'error': f'Customer segmentation failed: {str(e)}'}
    
    def _perform_anomaly_detection(self, df, question: str) -> Dict[str, Any]:
        """Detect anomalies and outliers in the data."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {'error': 'No numeric columns available for anomaly detection'}
            
            anomalies = {}
            
            for col in numeric_cols[:3]:  # Analyze first 3 numeric columns
                if df[col].std() == 0:  # Skip columns with no variation
                    continue
                    
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                
                # Identify outliers (z-score > 2.5)
                outlier_indices = np.where(z_scores > 2.5)[0]
                
                if len(outlier_indices) > 0:
                    anomalies[col] = {
                        'outlier_count': len(outlier_indices),
                        'outlier_percentage': round(len(outlier_indices) / len(df) * 100, 2),
                        'outlier_values': df.iloc[outlier_indices][col].tolist()[:5],  # First 5 outliers
                        'threshold': round(df[col].mean() + 2.5 * df[col].std(), 2)
                    }
            
            # Generate anomaly insights
            anomaly_insights = self._generate_anomaly_insights(anomalies)
            
            return {
                'anomalies_detected': anomalies,
                'insights': anomaly_insights,
                'severity': 'high' if any(info['outlier_percentage'] > 5 for info in anomalies.values()) else 'low'
            }
            
        except Exception as e:
            return {'error': f'Anomaly detection failed: {str(e)}'}
    
    def _perform_correlation_analysis(self, df, question: str) -> Dict[str, Any]:
        """Perform correlation analysis to identify relationships."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {'error': 'Insufficient numeric columns for correlation analysis'}
            
            # Calculate correlation matrix
            correlation_matrix = df[numeric_cols].corr()
            
            # Find strong correlations (> 0.7 or < -0.7)
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            'variable_1': correlation_matrix.columns[i],
                            'variable_2': correlation_matrix.columns[j],
                            'correlation': round(corr_value, 3),
                            'strength': 'very strong' if abs(corr_value) > 0.9 else 'strong',
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        })
            
            return {
                'correlation_matrix': correlation_matrix.round(3).to_dict(),
                'strong_correlations': strong_correlations,
                'insights': self._generate_correlation_insights(strong_correlations),
                'relationships_found': len(strong_correlations)
            }
            
        except Exception as e:
            return {'error': f'Correlation analysis failed: {str(e)}'}
    
    def _perform_forecasting(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Perform basic forecasting using linear regression."""
        try:
            # Find time and numeric columns
            time_cols = [col for col in df.columns if any(time_word in col.lower() 
                        for time_word in ['date', 'time', 'created', 'dt'])]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if not time_cols or len(numeric_cols) == 0:
                return {'error': 'Insufficient data for forecasting'}
            
            time_col = time_cols[0]
            
            # Convert to datetime and create numeric time feature
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col])
            
            if len(df) < 3:
                return {'error': 'Insufficient data points for forecasting'}
            
            # Create time-based features
            df['time_numeric'] = (df[time_col] - df[time_col].min()).dt.days
            
            forecasts = {}
            
            for col in numeric_cols[:2]:  # Forecast for first 2 numeric columns
                if df[col].std() == 0:  # Skip columns with no variation
                    continue
                
                # Prepare data
                X = df[['time_numeric']].values
                y = df[col].values
                
                # Fit linear regression
                model = LinearRegression()
                model.fit(X, y)
                
                # Make predictions for next 3 time periods
                future_time = np.array([[X[-1][0] + i] for i in range(1, 4)])
                future_predictions = model.predict(future_time)
                
                forecasts[col] = {
                    'predictions': [round(pred, 2) for pred in future_predictions],
                    'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing',
                    'slope': round(model.coef_[0], 4),
                    'r_squared': round(float(model.score(X, y)), 3)
                }
            
            return {
                'forecasts': forecasts,
                'forecast_horizon': '3 periods',
                'model_type': 'linear_regression',
                'insights': self._generate_forecast_insights(forecasts)
            }
            
        except Exception as e:
            return {'error': f'Forecasting failed: {str(e)}'}
    
    def _generate_strategic_insights(self, df: pd.DataFrame, question: str, analyses: Dict) -> List[str]:
        """Generate strategic business insights from the analyses."""
        insights = []
        
        # Analyze the overall data patterns
        if not df.empty:
            insights.append(f"Dataset contains {len(df)} records with {len(df.columns)} dimensions")
            
            # Data quality insights
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_percentage > 10:
                insights.append(f"Data quality concern: {missing_percentage:.1f}% missing values detected")
            
            # Growth insights from trend analysis
            if 'trend_analysis' in analyses and 'growth_rates' in analyses['trend_analysis']:
                growth_rates = analyses['trend_analysis']['growth_rates']
                if growth_rates:
                    avg_growth = np.mean(list(growth_rates.values()))
                    if avg_growth > 15:
                        insights.append(f"Strong growth trajectory detected: {avg_growth:.1f}% average growth")
                    elif avg_growth < -15:
                        insights.append(f"Declining trend alert: {avg_growth:.1f}% average decline")
            
            # Segmentation insights
            if 'customer_segmentation' in analyses and 'segments' in analyses['customer_segmentation']:
                segments = analyses['customer_segmentation']['segments']
                insights.append(f"Customer base shows {len(segments)} distinct behavioral segments")
                
                # Find dominant segment
                largest_segment = max(segments.items(), key=lambda x: x[1]['size'])
                insights.append(f"Dominant segment: {largest_segment[0]} represents {largest_segment[1]['percentage']}% of customers")
            
            # Anomaly insights
            if 'anomaly_detection' in analyses and 'anomalies_detected' in analyses['anomaly_detection']:
                anomalies = analyses['anomaly_detection']['anomalies_detected']
                if anomalies:
                    total_outliers = sum(info['outlier_count'] for info in anomalies.values())
                    insights.append(f"Anomaly alert: {total_outliers} outliers detected across key metrics")
        
        return insights
    
    def _generate_recommendations(self, df: pd.DataFrame, question: str, analyses: Dict) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        # Based on trend analysis
        if 'trend_analysis' in analyses and 'growth_rates' in analyses['trend_analysis']:
            growth_rates = analyses['trend_analysis']['growth_rates']
            for metric, rate in growth_rates.items():
                if rate > 20:
                    recommendations.append(f"Capitalize on {metric} growth momentum - consider scaling operations")
                elif rate < -10:
                    recommendations.append(f"Address declining {metric} - investigate root causes and implement corrective measures")
        
        # Based on segmentation analysis
        if 'customer_segmentation' in analyses and 'segments' in analyses['customer_segmentation']:
            segments = analyses['customer_segmentation']['segments']
            if len(segments) > 2:
                recommendations.append("Implement targeted marketing strategies for each customer segment")
                recommendations.append("Develop segment-specific products or services to maximize value")
        
        # Based on anomaly detection
        if 'anomaly_detection' in analyses and 'severity' in analyses['anomaly_detection']:
            if analyses['anomaly_detection']['severity'] == 'high':
                recommendations.append("Investigate high-severity anomalies immediately - potential fraud or system issues")
                recommendations.append("Implement automated monitoring for early anomaly detection")
        
        # Based on correlation analysis
        if 'correlation_analysis' in analyses and 'strong_correlations' in analyses['correlation_analysis']:
            correlations = analyses['correlation_analysis']['strong_correlations']
            for corr in correlations[:2]:  # Top 2 correlations
                if corr['direction'] == 'positive':
                    recommendations.append(f"Leverage positive relationship between {corr['variable_1']} and {corr['variable_2']} for strategic planning")
        
        # Generic data-driven recommendations
        if not recommendations:
            recommendations.append("Continue monitoring key metrics for emerging patterns")
            recommendations.append("Consider implementing automated reporting for regular insights")
        
        return recommendations
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the dataset."""
        return {
            'completeness': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            'total_records': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'quality_score': 'high' if df.isnull().sum().sum() / (len(df) * len(df.columns)) < 0.1 else 'medium'
        }
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'error': 'No numeric columns for statistical analysis'}
        
        summary = {}
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            summary[col] = {
                'count': int(df[col].count()),
                'mean': round(df[col].mean(), 2),
                'median': round(df[col].median(), 2),
                'std': round(df[col].std(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2),
                'skewness': round(df[col].skew(), 2),
                'kurtosis': round(df[col].kurtosis(), 2)
            }
        
        return summary
    
    # Helper methods for specific analysis types
    def _detect_trend_patterns(self, df: pd.DataFrame, time_col: str) -> Dict[str, str]:
        """Detect various trend patterns in time series data."""
        patterns = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:2]:
            if len(df) > 3:
                values = df.groupby(df[time_col].dt.to_period('M'))[col].sum()
                if len(values) > 2:
                    # Simple trend detection
                    if values.iloc[-1] > values.iloc[0] * 1.1:
                        patterns[col] = 'upward_trend'
                    elif values.iloc[-1] < values.iloc[0] * 0.9:
                        patterns[col] = 'downward_trend'
                    else:
                        patterns[col] = 'stable'
        
        return patterns
    
    def _generate_trend_recommendations(self, growth_rates: Dict, patterns: Dict) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        for metric, rate in growth_rates.items():
            if rate > 20:
                recommendations.append(f"Strong {metric} growth detected - consider expansion strategies")
            elif rate < -15:
                recommendations.append(f"Declining {metric} requires immediate attention")
        
        return recommendations
    
    def _generate_segment_insights(self, segments: Dict) -> List[str]:
        """Generate insights from customer segmentation."""
        insights = []
        
        # Find largest and smallest segments
        largest = max(segments.items(), key=lambda x: x[1]['size'])
        smallest = min(segments.items(), key=lambda x: x[1]['size'])
        
        insights.append(f"Largest segment: {largest[0]} ({largest[1]['percentage']}%)")
        insights.append(f"Smallest segment: {smallest[0]} ({smallest[1]['percentage']}%)")
        
        return insights
    
    def _generate_anomaly_insights(self, anomalies: Dict) -> List[str]:
        """Generate insights from anomaly detection."""
        insights = []
        
        for col, info in anomalies.items():
            if info['outlier_percentage'] > 5:
                insights.append(f"High anomaly rate in {col}: {info['outlier_percentage']}%")
            else:
                insights.append(f"Normal anomaly rate in {col}: {info['outlier_percentage']}%")
        
        return insights
    
    def _generate_correlation_insights(self, correlations: List[Dict]) -> List[str]:
        """Generate insights from correlation analysis."""
        insights = []
        
        for corr in correlations:
            strength = corr['strength']
            direction = corr['direction']
            var1, var2 = corr['variable_1'], corr['variable_2']
            
            insights.append(f"{strength.title()} {direction} correlation between {var1} and {var2}")
        
        return insights
    
    def _generate_forecast_insights(self, forecasts: Dict) -> List[str]:
        """Generate insights from forecasting analysis."""
        insights = []
        
        for metric, forecast in forecasts.items():
            trend = forecast['trend']
            r_squared = forecast['r_squared']
            
            accuracy = 'high' if r_squared > 0.8 else 'moderate' if r_squared > 0.5 else 'low'
            insights.append(f"{metric} shows {trend} trend with {accuracy} forecast accuracy")
        
        return insights
    
    # Additional advanced analysis methods
    def _perform_comparative_analysis(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Perform comparative analysis between different segments or time periods."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {'error': 'No numeric data for comparison'}
            
            # Simple comparative analysis
            comparisons = {}
            for col in numeric_cols[:3]:
                comparisons[col] = {
                    'mean': round(df[col].mean(), 2),
                    'median': round(df[col].median(), 2),
                    'std': round(df[col].std(), 2),
                    'coefficient_of_variation': round(df[col].std() / df[col].mean() * 100, 2) if df[col].mean() != 0 else 0
                }
            
            return {
                'comparisons': comparisons,
                'insights': ['Comparative analysis completed for key metrics']
            }
            
        except Exception as e:
            return {'error': f'Comparative analysis failed: {str(e)}'}
    
    def _perform_drill_down_analysis(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Perform drill-down analysis to identify root causes."""
        try:
            # Identify categorical columns for drill-down
            categorical_cols = df.select_dtypes(include=['object']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) == 0 or len(numeric_cols) == 0:
                return {'error': 'Insufficient data for drill-down analysis'}
            
            drill_down_results = {}
            
            # Analyze by categorical breakdowns
            for cat_col in categorical_cols[:2]:  # First 2 categorical columns
                for num_col in numeric_cols[:2]:  # First 2 numeric columns
                    breakdown = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'sum']).round(2)
                    
                    drill_down_results[f'{num_col}_by_{cat_col}'] = {
                        'breakdown': breakdown.to_dict(),
                        'top_category': breakdown.index[breakdown['sum'].values.argmax()] if not breakdown.empty else None,
                        'bottom_category': breakdown.index[breakdown['sum'].values.argmin()] if not breakdown.empty else None
                    }
            
            return {
                'drill_down_results': drill_down_results,
                'insights': ['Drill-down analysis reveals categorical performance differences']
            }
            
        except Exception as e:
            return {'error': f'Drill-down analysis failed: {str(e)}'}
    
    def _perform_cohort_analysis(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Perform cohort analysis for customer behavior tracking."""
        try:
            # This is a simplified cohort analysis
            # In a real implementation, you'd need customer registration dates and activity dates
            
            time_cols = [col for col in df.columns if any(time_word in col.lower() 
                        for time_word in ['date', 'time', 'created', 'dt'])]
            
            if not time_cols:
                return {'error': 'No time columns found for cohort analysis'}
            
            time_col = time_cols[0]
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Simple cohort analysis by month
            df['cohort_month'] = df[time_col].dt.to_period('M')
            cohort_data = df.groupby('cohort_month').size()
            
            return {
                'cohort_sizes': cohort_data.to_dict(),
                'total_cohorts': len(cohort_data),
                'insights': ['Basic cohort analysis shows customer acquisition patterns']
            }
            
        except Exception as e:
            return {'error': f'Cohort analysis failed: {str(e)}'}
    
    def _perform_basic_analysis(self, data: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """
        Perform basic analysis when advanced packages are not available.
        """
        try:
            if not data:
                return {'error': 'No data provided for analysis'}
            
            # Basic statistical analysis using built-in Python functions
            numeric_fields = []
            for row in data[:5]:  # Check first 5 rows
                for key, value in row.items():
                    if isinstance(value, (int, float)) and key not in numeric_fields:
                        numeric_fields.append(key)
            
            basic_stats = {}
            for field in numeric_fields:
                values = [row.get(field, 0) for row in data if isinstance(row.get(field), (int, float))]
                if values:
                    basic_stats[field] = {
                        'count': len(values),
                        'sum': sum(values),
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
            
            insights = [
                f"Analyzed {len(data)} records with {len(numeric_fields)} numeric fields",
                "Advanced analytics require pandas, numpy, and scikit-learn packages"
            ]
            
            return {
                'analysis_type': 'basic_statistical',
                'basic_statistics': basic_stats,
                'record_count': len(data),
                'numeric_fields': numeric_fields,
                'insights': insights,
                'recommendation': 'Install advanced packages for sophisticated analytics'
            }
            
        except Exception as e:
            return {'error': f'Basic analysis failed: {str(e)}'}