"""
Health Manager Module

Monitors model health and determines when models need retraining.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score


class HealthManager:
    """
    Monitors model health and determines retraining needs.
    """
    
    def __init__(self, performance_threshold=0.05, time_threshold_days=30):
        """
        Initialize HealthManager.
        
        Parameters:
        -----------
        performance_threshold : float
            Maximum allowed performance degradation (e.g., 0.05 = 5% drop)
        time_threshold_days : int
            Maximum days since last training before recommending retrain
        """
        self.performance_threshold = performance_threshold
        self.time_threshold_days = time_threshold_days
        self.health_history = []
        self.baseline_metrics = {}
    
    def set_baseline(self, model_name, metrics, timestamp=None):
        """
        Set baseline performance metrics for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        metrics : dict
            Dictionary of metric_name -> value (e.g., {'accuracy': 0.85, 'f1': 0.82})
        timestamp : datetime (optional)
            Timestamp of baseline (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.baseline_metrics[model_name] = {
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        print(f"✓ Baseline set for {model_name}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Metrics: {metrics}")
    
    def check_health(self, model_name, current_metrics, timestamp=None):
        """
        Check if a model needs retraining based on current performance.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        current_metrics : dict
            Current performance metrics
        timestamp : datetime (optional)
            Timestamp of current metrics (defaults to now)
            
        Returns:
        --------
        dict : Health check results with recommendations
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if model_name not in self.baseline_metrics:
            return {
                'needs_retraining': True,
                'reason': 'No baseline metrics found',
                'recommendations': ['Set baseline metrics first']
            }
        
        baseline = self.baseline_metrics[model_name]
        baseline_metrics = baseline['metrics']
        baseline_timestamp = baseline['timestamp']
        
        # Check performance degradation
        performance_issues = []
        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                degradation = baseline_value - current_value
                degradation_pct = (degradation / baseline_value) * 100 if baseline_value > 0 else 0
                
                if degradation > self.performance_threshold:
                    performance_issues.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation': degradation,
                        'degradation_pct': degradation_pct
                    })
        
        # Check time since last training
        days_since_training = (timestamp - baseline_timestamp).days
        time_exceeded = days_since_training > self.time_threshold_days
        
        # Determine if retraining is needed
        needs_retraining = len(performance_issues) > 0 or time_exceeded
        
        # Build recommendations
        recommendations = []
        if performance_issues:
            recommendations.append(f"Performance degraded on {len(performance_issues)} metric(s)")
        if time_exceeded:
            recommendations.append(f"Model is {days_since_training} days old (threshold: {self.time_threshold_days})")
        
        if not needs_retraining:
            recommendations.append("Model is healthy - no retraining needed")
        
        # Create health report
        health_report = {
            'model_name': model_name,
            'timestamp': timestamp,
            'needs_retraining': needs_retraining,
            'performance_issues': performance_issues,
            'days_since_training': days_since_training,
            'time_exceeded': time_exceeded,
            'recommendations': recommendations
        }
        
        # Store in history
        self.health_history.append(health_report)
        
        return health_report
    
    def print_health_report(self, health_report):
        """
        Print a formatted health report.
        
        Parameters:
        -----------
        health_report : dict
            Health report from check_health()
        """
        print(f"\n{'='*70}")
        print(f"MODEL HEALTH REPORT: {health_report['model_name']}")
        print(f"{'='*70}")
        print(f"Timestamp: {health_report['timestamp']}")
        print(f"Days since training: {health_report['days_since_training']}")
        print(f"Needs retraining: {'YES' if health_report['needs_retraining'] else 'NO'}")
        
        if health_report['performance_issues']:
            print(f"\nPerformance Issues:")
            for issue in health_report['performance_issues']:
                print(f"  • {issue['metric']}: {issue['baseline']:.4f} → {issue['current']:.4f} "
                      f"({issue['degradation_pct']:.1f}% degradation)")
        
        print(f"\nRecommendations:")
        for rec in health_report['recommendations']:
            print(f"  • {rec}")
        
        print(f"{'='*70}\n")
    
    def monitor_online_performance(self, model, X_recent, y_recent, model_name, 
                                   window_size=100, alert_threshold=0.1):
        """
        Monitor model performance on recent data (online monitoring).
        
        Parameters:
        -----------
        model : trained model
            The model to monitor
        X_recent : pd.DataFrame or np.ndarray
            Recent feature data
        y_recent : pd.Series or np.ndarray
            Recent true labels
        model_name : str
            Name of the model
        window_size : int
            Size of rolling window for monitoring
        alert_threshold : float
            Threshold for triggering alerts
            
        Returns:
        --------
        dict : Monitoring results
        """
        if len(X_recent) < window_size:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least {window_size} samples for monitoring'
            }
        
        # Calculate metrics on recent data
        y_pred = model.predict(X_recent)
        current_accuracy = accuracy_score(y_recent, y_pred)
        current_f1 = f1_score(y_recent, y_pred, average='weighted', zero_division=0)
        
        current_metrics = {
            'accuracy': current_accuracy,
            'f1': current_f1
        }
        
        # Check against baseline
        if model_name in self.baseline_metrics:
            baseline_accuracy = self.baseline_metrics[model_name]['metrics'].get('accuracy', 0)
            baseline_f1 = self.baseline_metrics[model_name]['metrics'].get('f1', 0)
            
            accuracy_drop = baseline_accuracy - current_accuracy
            f1_drop = baseline_f1 - current_f1
            
            alert = accuracy_drop > alert_threshold or f1_drop > alert_threshold
            
            return {
                'status': 'alert' if alert else 'healthy',
                'current_metrics': current_metrics,
                'baseline_metrics': self.baseline_metrics[model_name]['metrics'],
                'accuracy_drop': accuracy_drop,
                'f1_drop': f1_drop,
                'alert': alert,
                'message': 'Performance degradation detected!' if alert else 'Model performing well'
            }
        else:
            return {
                'status': 'no_baseline',
                'current_metrics': current_metrics,
                'message': 'No baseline set - cannot compare performance'
            }
    
    def get_health_history(self, model_name=None):
        """
        Get health check history.
        
        Parameters:
        -----------
        model_name : str (optional)
            Filter by model name
            
        Returns:
        --------
        list : List of health reports
        """
        if model_name is None:
            return self.health_history
        
        return [h for h in self.health_history if h['model_name'] == model_name]
    
    def export_health_report(self, filepath='health_report.csv'):
        """
        Export health history to CSV.
        
        Parameters:
        -----------
        filepath : str
            Path to save CSV file
        """
        if not self.health_history:
            print("No health history to export.")
            return
        
        # Flatten health history for CSV
        rows = []
        for report in self.health_history:
            row = {
                'model_name': report['model_name'],
                'timestamp': report['timestamp'],
                'needs_retraining': report['needs_retraining'],
                'days_since_training': report['days_since_training'],
                'time_exceeded': report['time_exceeded'],
                'num_performance_issues': len(report['performance_issues'])
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"✓ Health report exported to {filepath}")
    
    def reset_baseline(self, model_name):
        """
        Reset baseline for a model (useful after retraining).
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        """
        if model_name in self.baseline_metrics:
            del self.baseline_metrics[model_name]
            print(f"✓ Baseline reset for {model_name}")
        else:
            print(f"No baseline found for {model_name}")
    
    def get_retraining_priority(self):
        """
        Get list of models prioritized by retraining urgency.
        
        Returns:
        --------
        list : List of (model_name, urgency_score) tuples, sorted by urgency
        """
        if not self.health_history:
            return []
        
        # Get latest report for each model
        latest_reports = {}
        for report in self.health_history:
            model_name = report['model_name']
            if model_name not in latest_reports or report['timestamp'] > latest_reports[model_name]['timestamp']:
                latest_reports[model_name] = report
        
        # Calculate urgency scores
        urgency_scores = []
        for model_name, report in latest_reports.items():
            if not report['needs_retraining']:
                continue
            
            # Urgency based on performance degradation and time
            urgency = 0
            
            # Add points for performance issues
            for issue in report['performance_issues']:
                urgency += issue['degradation_pct']
            
            # Add points for time
            if report['time_exceeded']:
                days_over = report['days_since_training'] - self.time_threshold_days
                urgency += days_over * 0.5
            
            urgency_scores.append((model_name, urgency))
        
        # Sort by urgency (highest first)
        return sorted(urgency_scores, key=lambda x: x[1], reverse=True)
