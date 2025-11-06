"""
Parameter Analyzer Module

Analyzes optimization results and parameter importance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import optuna
import optuna.importance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.utils.exceptions import ModelError


class ParameterAnalyzer:
    """
    Analyzes optimization results and generates insights.
    """
    
    def __init__(self, study: optuna.Study, config, logger):
        """
        Initialize parameter analyzer.
        
        Args:
            study: Completed Optuna study
            config: Configuration object
            logger: Logger instance
        """
        self.study = study
        self.config = config
        self.logger = logger
        
        self.logger.info("ParameterAnalyzer initialized")
    
    def analyze_importance(self) -> Dict[str, float]:
        """
        Calculate and return parameter importance ranking.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        try:
            importances = optuna.importance.get_param_importances(self.study)
            return importances
        except Exception as e:
            self.logger.warning(f"Failed to calculate parameter importance: {str(e)}")
            return {}
    
    def plot_param_relationships(self) -> go.Figure:
        """
        Generate correlation heatmap of parameters vs objective.
        
        Returns:
            Plotly figure object
        """
        # Get completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        
        if len(completed_trials) == 0:
            raise ModelError("No completed trials to analyze")
        
        # Extract parameters and values
        param_names = set()
        for trial in completed_trials:
            param_names.update(trial.params.keys())
        
        param_names = sorted(list(param_names))
        
        # Build correlation matrix
        data = []
        for trial in completed_trials:
            row = [trial.value]
            for param_name in param_names:
                row.append(trial.params.get(param_name, np.nan))
            data.append(row)
        
        df = pd.DataFrame(data, columns=['objective'] + param_names)
        
        # Calculate correlation
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Parameter-Objective Correlation Matrix',
            xaxis_title='Parameters',
            yaxis_title='Parameters',
            width=800,
            height=600
        )
        
        return fig
    
    def find_optimal_regions(self) -> Dict[str, Dict]:
        """
        Identify parameter ranges that consistently perform well.
        
        Returns:
            Dictionary with optimal ranges for each parameter
        """
        completed_trials = [t for t in self.study.trials 
                           if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        
        if len(completed_trials) == 0:
            return {}
        
        # Sort by objective value (best first)
        if self.study.direction == 'maximize':
            sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        else:
            sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        
        # Get top 25% of trials
        top_25_pct = int(len(sorted_trials) * 0.25)
        top_trials = sorted_trials[:max(top_25_pct, 1)]
        
        # Extract parameter ranges from top trials
        optimal_ranges = {}
        param_names = set()
        for trial in top_trials:
            param_names.update(trial.params.keys())
        
        for param_name in param_names:
            values = [t.params[param_name] for t in top_trials if param_name in t.params]
            if values:
                optimal_ranges[param_name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'median': np.median(values)
                }
        
        return optimal_ranges
    
    def generate_insights(self) -> str:
        """
        Create text report with actionable recommendations.
        
        Returns:
            Insights text report
        """
        insights = []
        insights.append("=" * 60)
        insights.append("HYPERPARAMETER OPTIMIZATION INSIGHTS")
        insights.append("=" * 60)
        insights.append("")
        
        # Parameter importance
        importances = self.analyze_importance()
        if importances:
            insights.append("PARAMETER IMPORTANCE:")
            insights.append("-" * 60)
            sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            for param_name, importance in sorted_importances:
                if importance > 0.2:
                    insights.append(f"  • {param_name}: {importance:.3f} (HIGH IMPACT)")
                elif importance < 0.05:
                    insights.append(f"  • {param_name}: {importance:.3f} (LOW IMPACT)")
                else:
                    insights.append(f"  • {param_name}: {importance:.3f} (MODERATE IMPACT)")
            
            insights.append("")
        
        # Optimal regions
        optimal_ranges = self.find_optimal_regions()
        if optimal_ranges:
            insights.append("OPTIMAL PARAMETER REGIONS (Top 25% of trials):")
            insights.append("-" * 60)
            
            for param_name, ranges in optimal_ranges.items():
                insights.append(f"  • {param_name}:")
                insights.append(f"    - Range: [{ranges['min']:.3f}, {ranges['max']:.3f}]")
                insights.append(f"    - Mean: {ranges['mean']:.3f}")
                insights.append(f"    - Median: {ranges['median']:.3f}")
            
            insights.append("")
        
        # Low sensitivity parameters
        if importances:
            insights.append("LOW SENSITIVITY PARAMETERS (can be fixed):")
            insights.append("-" * 60)
            low_sensitivity = [name for name, imp in importances.items() if imp < 0.05]
            if low_sensitivity:
                for param_name in low_sensitivity:
                    insights.append(f"  • {param_name} (importance < 0.05)")
            else:
                insights.append("  • None found")
            insights.append("")
        
        # Recommendations
        insights.append("RECOMMENDATIONS:")
        insights.append("-" * 60)
        
        if importances:
            most_important = max(importances.items(), key=lambda x: x[1])
            insights.append(f"  1. Focus on {most_important[0]} (most important parameter)")
        
        if optimal_ranges:
            insights.append("  2. Consider constraining search space to optimal regions")
        
        if importances:
            low_sensitivity = [name for name, imp in importances.items() if imp < 0.05]
            if low_sensitivity:
                insights.append(f"  3. Fix low-sensitivity parameters to reduce search space")
        
        insights.append("")
        insights.append("=" * 60)
        
        return "\n".join(insights)
    
    def plot_parallel_coordinate(self) -> go.Figure:
        """
        Generate parallel coordinate plot showing parameter interactions.
        
        Returns:
            Plotly figure object
        """
        completed_trials = [t for t in self.study.trials 
                           if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        
        if len(completed_trials) == 0:
            raise ModelError("No completed trials to plot")
        
        # Get all parameter names
        param_names = set()
        for trial in completed_trials:
            param_names.update(trial.params.keys())
        
        param_names = sorted(list(param_names))
        
        if len(param_names) == 0:
            raise ModelError("No parameters found in trials")
        
        # Prepare data
        dimensions = []
        for param_name in param_names:
            values = [t.params.get(param_name, np.nan) for t in completed_trials]
            if all(not np.isnan(v) for v in values):
                dimensions.append({
                    'label': param_name,
                    'values': values
                })
        
        # Add objective
        objective_values = [t.value for t in completed_trials]
        dimensions.append({
            'label': 'Objective',
            'values': objective_values
        })
        
        # Create parallel coordinate plot
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=objective_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Objective")
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title='Parallel Coordinate Plot - Parameter Interactions',
            width=1200,
            height=600
        )
        
        return fig
    
    def plot_contour(self, param1: str, param2: str) -> go.Figure:
        """
        Generate 2D contour plots for parameter pairs.
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            
        Returns:
            Plotly figure object
        """
        completed_trials = [t for t in self.study.trials 
                           if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        
        if len(completed_trials) == 0:
            raise ModelError("No completed trials to plot")
        
        # Extract parameter values
        param1_values = []
        param2_values = []
        objective_values = []
        
        for trial in completed_trials:
            if param1 in trial.params and param2 in trial.params:
                param1_values.append(trial.params[param1])
                param2_values.append(trial.params[param2])
                objective_values.append(trial.value)
        
        if len(param1_values) == 0:
            raise ModelError(f"Parameters {param1} and {param2} not found in trials")
        
        # Create contour plot
        fig = go.Figure(data=go.Contour(
            x=param1_values,
            y=param2_values,
            z=objective_values,
            colorscale='Viridis',
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))
        
        fig.update_layout(
            title=f'Contour Plot: {param1} vs {param2}',
            xaxis_title=param1,
            yaxis_title=param2,
            width=800,
            height=600
        )
        
        return fig
    
    def get_importance_dataframe(self) -> pd.DataFrame:
        """
        Return DataFrame with parameter importance scores.
        
        Returns:
            DataFrame with parameter names and importance scores
        """
        importances = self.analyze_importance()
        
        if not importances:
            return pd.DataFrame(columns=['parameter', 'importance'])
        
        df = pd.DataFrame([
            {'parameter': param, 'importance': importance}
            for param, importance in importances.items()
        ])
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df



