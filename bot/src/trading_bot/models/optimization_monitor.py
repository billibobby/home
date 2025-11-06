"""
Optimization Monitor Module

Tracks optimization progress with callbacks and checkpointing.
"""

import os
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import optuna
import joblib
from pathlib import Path

from trading_bot.utils.exceptions import ModelError


class OptimizationMonitor:
    """
    Monitors optimization progress with real-time tracking and checkpointing.
    """
    
    def __init__(self, config, logger, checkpoint_dir: str = 'optimization_checkpoints'):
        """
        Initialize optimization monitor.
        
        Args:
            config: Configuration object
            logger: Logger instance
            checkpoint_dir: Directory for checkpoint files
        """
        self.config = config
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get monitoring config
        monitoring_config = config.get('models.optimization.monitoring', {})
        self.checkpoint_interval = monitoring_config.get('checkpoint_interval', 10)
        
        # Track state
        self.start_time = None
        self.trial_durations = []
        self.completed_trials = 0
        self.best_value = None
        self.current_trial = None
        
        # Notification config
        notifications_config = config.get('models.optimization.notifications', {})
        self.notifications_enabled = notifications_config.get('enabled', False)
        self.email_config = notifications_config.get('email', {})
        self.slack_config = notifications_config.get('slack', {})
        
        self.logger.info(f"OptimizationMonitor initialized with checkpoint_dir={checkpoint_dir}")
    
    def create_callback(self) -> Callable:
        """
        Return Optuna callback function for trial updates.
        
        Returns:
            Callback function
        """
        def callback(study: optuna.Study, trial: optuna.Trial):
            """Optuna callback for trial completion."""
            self._on_trial_complete(study, trial)
        
        return callback
    
    def _on_trial_complete(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback invoked after each trial completion.
        
        Args:
            study: Optuna study
            trial: Completed trial
        """
        # Track completion
        self.completed_trials += 1
        self.current_trial = trial.number
        
        # Track best value
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            if self.best_value is None:
                self.best_value = trial.value
            else:
                if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                    if trial.value > self.best_value:
                        self.best_value = trial.value
                else:
                    if trial.value < self.best_value:
                        self.best_value = trial.value
        
        # Track trial duration
        if trial.datetime_start and trial.datetime_complete:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
            self.trial_durations.append(duration)
        
        # Save checkpoint periodically
        if self.completed_trials % self.checkpoint_interval == 0:
            self._save_checkpoint(study, trial.number)
        
        # Update dashboard (if implemented)
        self._update_dashboard(study, trial)
        
        # Send notification if enabled
        if self.notifications_enabled and self.completed_trials % 50 == 0:
            best_str = f"{self.best_value:.4f}" if self.best_value is not None else "N/A"
            message = f"Optimization progress: {self.completed_trials} trials completed, best value: {best_str}"
            self.send_notification(message, level='info')
    
    def _save_checkpoint(self, study: optuna.Study, trial_number: int) -> None:
        """
        Save study state periodically.
        
        Args:
            study: Optuna study
            trial_number: Current trial number
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_trial_{trial_number}.pkl"
            joblib.dump(study, checkpoint_path)
            
            # Keep only last 5 checkpoints
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    old_checkpoint.unlink()
            
            self.logger.info(f"Checkpoint saved at trial {trial_number}")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {str(e)}")
    
    def _update_dashboard(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Update real-time metrics (current best, trials completed, ETA).
        
        Args:
            study: Optuna study
            trial: Current trial
        """
        # This method can be extended to update GUI or external dashboard
        # For now, just log progress
        if self.completed_trials % 10 == 0:
            eta = self.estimate_time_remaining(study, self.start_time, len(study.trials))
            best_str = f"{self.best_value:.4f}" if self.best_value is not None else "N/A"
            self.logger.info(
                f"Progress: {self.completed_trials} trials, "
                f"Best: {best_str}, "
                f"ETA: {eta}"
            )
    
    def get_progress_info(self) -> Dict:
        """
        Return dictionary with current optimization status.
        
        Returns:
            Dictionary with progress information
        """
        avg_duration = np.mean(self.trial_durations) if self.trial_durations else 0.0
        
        return {
            'completed_trials': self.completed_trials,
            'current_trial': self.current_trial,
            'best_value': self.best_value,
            'average_trial_duration': avg_duration,
            'total_duration': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0,
            'checkpoint_dir': str(self.checkpoint_dir)
        }
    
    def estimate_time_remaining(self, study: optuna.Study, start_time: Optional[datetime],
                               n_trials: int) -> str:
        """
        Calculate ETA based on average trial duration.
        
        Args:
            study: Optuna study
            start_time: Optimization start time
            n_trials: Total number of trials planned
            
        Returns:
            Formatted ETA string
        """
        if not self.trial_durations or len(self.trial_durations) == 0:
            return "Unknown"
        
        avg_duration = np.mean(self.trial_durations)
        remaining_trials = max(0, n_trials - self.completed_trials)
        remaining_seconds = remaining_trials * avg_duration
        
        if remaining_seconds < 60:
            return f"{int(remaining_seconds)}s"
        elif remaining_seconds < 3600:
            return f"{int(remaining_seconds / 60)}m"
        else:
            hours = int(remaining_seconds / 3600)
            minutes = int((remaining_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def send_notification(self, message: str, level: str = 'info') -> None:
        """
        Send notification via configured channel (email/Slack).
        
        Args:
            message: Notification message
            level: Notification level ('info', 'warning', 'error')
        """
        if not self.notifications_enabled:
            return
        
        # Email notification
        if self.email_config.get('enabled', False):
            try:
                self._send_email(message, level)
            except Exception as e:
                self.logger.warning(f"Failed to send email notification: {str(e)}")
        
        # Slack notification
        if self.slack_config.get('enabled', False):
            try:
                self._send_slack(message, level)
            except Exception as e:
                self.logger.warning(f"Failed to send Slack notification: {str(e)}")
    
    def _send_email(self, message: str, level: str) -> None:
        """Send email notification using smtplib."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.email_config.get('smtp_port', 587)
            from_address = self.email_config.get('from_address', '')
            to_address = self.email_config.get('to_address', '')
            
            if not from_address or not to_address:
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_address
            msg['To'] = to_address
            msg['Subject'] = f"Trading Bot Optimization - {level.upper()}"
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email (would need authentication in production)
            # For now, just log
            self.logger.info(f"Email notification: {message}")
            
        except ImportError:
            self.logger.warning("smtplib not available for email notifications")
        except Exception as e:
            self.logger.warning(f"Email notification failed: {str(e)}")
    
    def _send_slack(self, message: str, level: str) -> None:
        """Send Slack notification using webhook."""
        try:
            import requests
            
            webhook_url = self.slack_config.get('webhook_url', '')
            if not webhook_url:
                return
            
            payload = {
                'text': f"[{level.upper()}] {message}",
                'username': 'Trading Bot Optimizer'
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except ImportError:
            self.logger.warning("requests not available for Slack notifications")
        except Exception as e:
            self.logger.warning(f"Slack notification failed: {str(e)}")
    
    def generate_progress_report(self) -> str:
        """
        Create HTML report with current optimization state.
        
        Returns:
            HTML string
        """
        progress_info = self.get_progress_info()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Progress Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>Optimization Progress Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Progress Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Completed Trials</td><td>{progress_info['completed_trials']}</td></tr>
                <tr><td>Current Trial</td><td>{progress_info['current_trial']}</td></tr>
                <tr><td>Best Value</td><td>{f"{progress_info['best_value']:.4f}" if progress_info['best_value'] is not None else 'N/A'}</td></tr>
                <tr><td>Average Trial Duration</td><td>{progress_info['average_trial_duration']:.2f}s</td></tr>
                <tr><td>Total Duration</td><td>{progress_info['total_duration']:.2f}s</td></tr>
            </table>
        </body>
        </html>
        """
        
        return html

