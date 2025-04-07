"""
MPC Thermal Model Validation Script

This script performs comprehensive validation of the MPC thermal model
by comparing predicted temperature trajectories with actual measurements.

The script:
1. Collects MPC predictions and actual temperature measurements
2. Matches predictions with corresponding measurements
3. Calculates prediction errors and performance metrics
4. Generates visualizations and detailed Excel reports
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MPC_Validation")

# Define constants
ZONE_NAMES = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
PREDICTION_HORIZON = 12  # Number of steps in prediction horizon (default: 12)
TIME_STEP = 300  # Seconds per step (default: 300s = 5min)
COMFORT_MIN = 70.0  # Minimum comfort temperature (°F)
COMFORT_MAX = 80.0  # Maximum comfort temperature (°F)

class MPCValidator:
    """Class to validate MPC thermal model predictions against actual measurements"""
    
    def __init__(self, data_dir="data", output_dir="validation_results"):
        """Initialize the validator with directories for data and output"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.predictions_dir = os.path.join(data_dir, "mpc_predictions")
        self.temperatures_dir = os.path.join(data_dir, "temperatures")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage
        self.predictions = []
        self.actual_temperatures = []
        self.matched_data = []
        self.errors = None
        
        logger.info(f"MPC Validator initialized. Data dir: {data_dir}, Output dir: {output_dir}")
    
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string from various formats"""
        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            try:
                # Try with milliseconds
                return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                try:
                    # Try standard format
                    return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                except ValueError:
                    # Extract with regex or other methods if needed
                    logger.error(f"Could not parse timestamp: {timestamp_str}")
                    return None
    
    def extract_timestamp_from_filename(self, filename):
        """Extract timestamp from filename"""
        try:
            base = os.path.basename(filename)
            # Look for patterns like temps_2025-03-26_22-30-15.json
            parts = base.split('_')
            if len(parts) >= 2:
                # Join date and time parts if they exist
                timestamp_str = '_'.join(parts[1:])
                # Remove extension
                timestamp_str = timestamp_str.split('.')[0]
                return self.parse_timestamp(timestamp_str)
            else:
                logger.warning(f"Could not extract timestamp from filename: {base}")
                return None
        except Exception as e:
            logger.error(f"Error extracting timestamp from {filename}: {e}")
            return None
    
    def load_mpc_predictions(self):
        """Load all MPC prediction data from JSON files"""
        prediction_files = sorted(glob.glob(os.path.join(self.predictions_dir, "pred_*.json")))
        logger.info(f"Found {len(prediction_files)} prediction files")
        
        for file in prediction_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                    # Ensure timestamp is a datetime object
                    if 'timestamp' in data:
                        data['timestamp'] = self.parse_timestamp(data['timestamp'])
                    else:
                        # Try to extract from filename
                        data['timestamp'] = self.extract_timestamp_from_filename(file)
                    
                    if data['timestamp'] is not None:
                        self.predictions.append(data)
            except Exception as e:
                logger.error(f"Error loading prediction file {file}: {e}")
        
        logger.info(f"Loaded {len(self.predictions)} prediction records")
        return len(self.predictions) > 0
    
    def load_actual_temperatures(self):
        """Load all actual temperature measurements from JSON files"""
        temp_files = sorted(glob.glob(os.path.join(self.temperatures_dir, "temps_*.json")))
        logger.info(f"Found {len(temp_files)} temperature files")
        
        for file in temp_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract timestamp
                    if 'timestamp' in data:
                        timestamp = self.parse_timestamp(data['timestamp'])
                    else:
                        timestamp = self.extract_timestamp_from_filename(file)
                    
                    if timestamp is None:
                        continue
                    
                    # Extract temperatures based on file format
                    temperatures = {}
                    if 'temperatures' in data:
                        temperatures = data['temperatures']
                    else:
                        # Try to extract directly from data
                        for zone in ZONE_NAMES:
                            if zone in data:
                                temperatures[zone] = data[zone]
                    
                    # Store the data
                    self.actual_temperatures.append({
                        'timestamp': timestamp,
                        'temperatures': temperatures
                    })
            except Exception as e:
                logger.error(f"Error loading temperature file {file}: {e}")
        
        logger.info(f"Loaded {len(self.actual_temperatures)} temperature records")
        return len(self.actual_temperatures) > 0
    
    def match_predictions_with_actuals(self, time_tolerance=timedelta(seconds=60)):
        """Match predictions with their corresponding actual measurements"""
        if not self.predictions or not self.actual_temperatures:
            logger.error("Cannot match data: missing predictions or actual temperatures")
            return False
        
        logger.info("Matching predictions with actual measurements...")
        self.matched_data = []
        
        # Sort both lists by timestamp
        self.predictions.sort(key=lambda x: x['timestamp'])
        self.actual_temperatures.sort(key=lambda x: x['timestamp'])
        
        # For each prediction
        for pred in self.predictions:
            pred_time = pred['timestamp']
            
            # Find actual temperature matches for each step in the prediction horizon
            matched_actuals = []
            
            # For each step in prediction horizon
            for step in range(PREDICTION_HORIZON):
                # Calculate the expected time for this prediction step
                target_time = pred_time + timedelta(seconds=TIME_STEP * step)
                
                # Find closest actual measurement
                closest_actual = None
                min_time_diff = timedelta(days=1)  # Start with large difference
                
                for actual in self.actual_temperatures:
                    time_diff = abs(actual['timestamp'] - target_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_actual = actual
                
                # Only use if within tolerance
                if closest_actual and min_time_diff <= time_tolerance:
                    matched_actuals.append({
                        'step': step,
                        'predicted_time': target_time,
                        'actual_time': closest_actual['timestamp'],
                        'time_diff_seconds': min_time_diff.total_seconds(),
                        'temperatures': closest_actual['temperatures']
                    })
                else:
                    matched_actuals.append(None)
            
            # Only include predictions with at least some matched actual data
            if any(matched_actuals):
                # Check if we have valid predicted trajectories
                if ('predicted_trajectories' in pred and 
                    pred['predicted_trajectories'] and 
                    len(pred['predicted_trajectories']) >= len(ZONE_NAMES)):
                    
                    self.matched_data.append({
                        'prediction_time': pred_time,
                        'current_state': pred.get('current_state'),
                        'predicted_trajectories': pred['predicted_trajectories'],
                        'ac_decision': pred.get('ac_decision'),
                        'damper_decisions': pred.get('damper_decisions'),
                        'ambient_temp': pred.get('ambient_temp'),
                        'matched_actuals': matched_actuals
                    })
        
        logger.info(f"Successfully matched {len(self.matched_data)} prediction-actual pairs")
        return len(self.matched_data) > 0
    
    def calculate_prediction_errors(self):
        """Calculate prediction errors at different horizon steps"""
        if not self.matched_data:
            logger.error("Cannot calculate errors: no matched data")
            return False
        
        logger.info("Calculating prediction errors...")
        
        # Create data structure for errors
        self.errors = {
            'step_errors': {
                'overall': {step: {'mae': [], 'rmse': [], 'max_error': []} for step in range(PREDICTION_HORIZON)},
                'zones': {zone: {step: {'mae': [], 'rmse': [], 'max_error': []} 
                                for step in range(PREDICTION_HORIZON)} 
                          for zone in ZONE_NAMES}
            },
            'zone_errors': {zone: {'mae': [], 'rmse': [], 'max_error': []} for zone in ZONE_NAMES},
            'overall': {'mae': [], 'rmse': [], 'max_error': []}
        }
        
        # Process each matched prediction
        for match in self.matched_data:
            predicted_trajectories = match['predicted_trajectories']
            matched_actuals = match['matched_actuals']
            
            # For each step with actual data
            for step_idx, actual_data in enumerate(matched_actuals):
                if actual_data is None:
                    continue
                
                step = actual_data['step']
                actual_temps = actual_data['temperatures']
                
                # Extract predicted temperatures for this step
                if step < len(predicted_trajectories[0]):
                    # Compare each zone
                    for zone_idx, zone in enumerate(ZONE_NAMES):
                        if zone_idx < len(predicted_trajectories) and zone in actual_temps:
                            pred_temp = predicted_trajectories[zone_idx][step]
                            actual_temp = actual_temps[zone]
                            
                            if actual_temp is not None:
                                # Calculate error
                                error = abs(pred_temp - actual_temp)
                                squared_error = error**2
                                
                                # Store error by step and zone
                                self.errors['step_errors']['overall'][step]['mae'].append(error)
                                self.errors['step_errors']['overall'][step]['rmse'].append(squared_error)
                                self.errors['step_errors']['overall'][step]['max_error'].append(error)
                                
                                # Store error by zone and step
                                self.errors['step_errors']['zones'][zone][step]['mae'].append(error)
                                self.errors['step_errors']['zones'][zone][step]['rmse'].append(squared_error)
                                self.errors['step_errors']['zones'][zone][step]['max_error'].append(error)
                                
                                # Store zone-specific errors
                                self.errors['zone_errors'][zone]['mae'].append(error)
                                self.errors['zone_errors'][zone]['rmse'].append(squared_error)
                                self.errors['zone_errors'][zone]['max_error'].append(error)
                                
                                # Store overall errors
                                self.errors['overall']['mae'].append(error)
                                self.errors['overall']['rmse'].append(squared_error)
                                self.errors['overall']['max_error'].append(error)
        
        # Calculate final statistics
        # Overall
        if self.errors['overall']['mae']:
            self.errors['overall']['mae'] = np.mean(self.errors['overall']['mae'])
            self.errors['overall']['rmse'] = np.sqrt(np.mean(self.errors['overall']['rmse']))
            self.errors['overall']['max_error'] = max(self.errors['overall']['max_error'])
        
        # By zone
        for zone in ZONE_NAMES:
            if self.errors['zone_errors'][zone]['mae']:
                self.errors['zone_errors'][zone]['mae'] = np.mean(self.errors['zone_errors'][zone]['mae'])
                self.errors['zone_errors'][zone]['rmse'] = np.sqrt(np.mean(self.errors['zone_errors'][zone]['rmse']))
                self.errors['zone_errors'][zone]['max_error'] = max(self.errors['zone_errors'][zone]['max_error'])
        
        # By step (overall)
        for step in range(PREDICTION_HORIZON):
            if self.errors['step_errors']['overall'][step]['mae']:
                self.errors['step_errors']['overall'][step]['mae'] = np.mean(self.errors['step_errors']['overall'][step]['mae'])
                self.errors['step_errors']['overall'][step]['rmse'] = np.sqrt(np.mean(self.errors['step_errors']['overall'][step]['rmse']))
                self.errors['step_errors']['overall'][step]['max_error'] = max(self.errors['step_errors']['overall'][step]['max_error'])
        
        # By step and zone
        for zone in ZONE_NAMES:
            for step in range(PREDICTION_HORIZON):
                if self.errors['step_errors']['zones'][zone][step]['mae']:
                    self.errors['step_errors']['zones'][zone][step]['mae'] = np.mean(self.errors['step_errors']['zones'][zone][step]['mae'])
                    self.errors['step_errors']['zones'][zone][step]['rmse'] = np.sqrt(np.mean(self.errors['step_errors']['zones'][zone][step]['rmse']))
                    self.errors['step_errors']['zones'][zone][step]['max_error'] = max(self.errors['step_errors']['zones'][zone][step]['max_error'])
        
        logger.info("Error calculation complete")
        return True
    
    def create_visualizations(self):
        """Generate visualizations of prediction accuracy"""
        if not self.matched_data or not self.errors:
            logger.error("Cannot create visualizations: missing data")
            return False
        
        logger.info("Creating validation visualizations...")
        
        # Create visualization folder
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Sample individual prediction-actual comparisons
        self._create_sample_comparisons(viz_dir)
        
        # 2. Error by prediction step
        self._create_step_error_plot(viz_dir)
        
        # 3. Error by zone
        self._create_zone_error_plot(viz_dir)
        
        # 4. Error heatmap by zone and step
        self._create_error_heatmap(viz_dir)
        
        logger.info(f"Visualizations created in {viz_dir}")
        return True
    
    def _create_sample_comparisons(self, viz_dir, num_samples=5):
        """Create sample comparison plots between predicted and actual temperatures"""
        # Select samples evenly spread across the dataset
        total_samples = len(self.matched_data)
        sample_indices = np.linspace(0, total_samples-1, min(num_samples, total_samples), dtype=int)
        
        for idx in sample_indices:
            match = self.matched_data[idx]
            pred_time = match['prediction_time']
            
            # Create figure
            fig, axes = plt.subplots(4, 2, figsize=(15, 20))
            fig.suptitle(f'Prediction vs Actual Temperatures (Prediction Time: {pred_time.strftime("%Y-%m-%d %H:%M")})', 
                         fontsize=16)
            
            axes = axes.flatten()
            
            for zone_idx, zone in enumerate(ZONE_NAMES):
                ax = axes[zone_idx]
                
                # Extract prediction data
                pred_times = [pred_time + timedelta(seconds=TIME_STEP * step) for step in range(PREDICTION_HORIZON)]
                pred_temps = [match['predicted_trajectories'][zone_idx][step] for step in range(PREDICTION_HORIZON)]
                
                # Extract actual data
                actual_times = []
                actual_temps = []
                
                for actual in match['matched_actuals']:
                    if actual is not None:
                        time = actual['predicted_time']
                        temp = actual['temperatures'].get(zone)
                        
                        if temp is not None:
                            actual_times.append(time)
                            actual_temps.append(temp)
                
                # Plot data
                ax.plot(pred_times, pred_temps, 'b-', label='Predicted')
                ax.plot(actual_times, actual_temps, 'r--o', label='Actual')
                
                # Calculate and display error metrics for this zone
                if actual_temps:
                    # Find matching indices between predictions and actuals
                    matching_indices = []
                    for i, actual_time in enumerate(actual_times):
                        for j, pred_time in enumerate(pred_times):
                            if abs((actual_time - pred_time).total_seconds()) < 60:  # Within 1 minute
                                matching_indices.append((i, j))
                    
                    if matching_indices:
                        errors = [abs(actual_temps[i] - pred_temps[j]) for i, j in matching_indices]
                        mae = np.mean(errors)
                        rmse = np.sqrt(np.mean([e**2 for e in errors]))
                        max_err = max(errors)
                        
                        ax.text(0.05, 0.95, f'MAE: {mae:.2f}°F\nRMSE: {rmse:.2f}°F\nMax Error: {max_err:.2f}°F',
                                transform=ax.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add comfort bounds
                ax.axhline(y=COMFORT_MIN, color='blue', linestyle='--', alpha=0.3, label='Min Comfort')
                ax.axhline(y=COMFORT_MAX, color='red', linestyle='--', alpha=0.3, label='Max Comfort')
                
                ax.set_title(f'Zone {zone}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Temperature (°F)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis to show only hours and minutes
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
            plt.savefig(os.path.join(viz_dir, f'prediction_comparison_{pred_time.strftime("%Y%m%d_%H%M")}.png'))
            plt.close()
    
    def _create_step_error_plot(self, viz_dir):
        """Create plot showing error by prediction step"""
        plt.figure(figsize=(10, 6))
        
        # Extract data
        steps = range(PREDICTION_HORIZON)
        mae_values = []
        rmse_values = []
        
        for step in steps:
            mae_values.append(self.errors['step_errors']['overall'][step].get('mae', 0))
            rmse_values.append(self.errors['step_errors']['overall'][step].get('rmse', 0))
        
        # Plot MAE and RMSE
        plt.bar([s - 0.2 for s in steps], mae_values, width=0.4, label='MAE', color='blue', alpha=0.7)
        plt.bar([s + 0.2 for s in steps], rmse_values, width=0.4, label='RMSE', color='red', alpha=0.7)
        
        plt.xlabel('Prediction Step (5 min intervals)')
        plt.ylabel('Error (°F)')
        plt.title('Prediction Error by Horizon Step')
        plt.xticks(steps)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'error_by_step.png'))
        plt.close()
    
    def _create_zone_error_plot(self, viz_dir):
        """Create plot showing error by zone"""
        plt.figure(figsize=(12, 6))
        
        # Extract data
        mae_values = []
        rmse_values = []
        
        for zone in ZONE_NAMES:
            mae_values.append(self.errors['zone_errors'][zone].get('mae', 0))
            rmse_values.append(self.errors['zone_errors'][zone].get('rmse', 0))
        
        # Plot
        x = np.arange(len(ZONE_NAMES))
        width = 0.35
        
        plt.bar(x - width/2, mae_values, width, label='MAE', color='blue', alpha=0.7)
        plt.bar(x + width/2, rmse_values, width, label='RMSE', color='red', alpha=0.7)
        
        plt.xlabel('Zone')
        plt.ylabel('Error (°F)')
        plt.title('Prediction Error by Zone')
        plt.xticks(x, ZONE_NAMES)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'error_by_zone.png'))
        plt.close()
    
    def _create_error_heatmap(self, viz_dir):
        """Create heatmap showing error by zone and prediction step"""
        plt.figure(figsize=(14, 8))
        
        # Create data matrix
        error_matrix = np.zeros((len(ZONE_NAMES), PREDICTION_HORIZON))
        
        for i, zone in enumerate(ZONE_NAMES):
            for step in range(PREDICTION_HORIZON):
                error_matrix[i, step] = self.errors['step_errors']['zones'][zone][step].get('mae', 0)
        
        # Create heatmap
        plt.imshow(error_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='MAE (°F)')
        
        plt.xlabel('Prediction Step (5 min intervals)')
        plt.ylabel('Zone')
        plt.title('Prediction Error by Zone and Horizon Step')
        plt.xticks(range(PREDICTION_HORIZON))
        plt.yticks(range(len(ZONE_NAMES)), ZONE_NAMES)
        
        # Add error values as text
        for i in range(len(ZONE_NAMES)):
            for j in range(PREDICTION_HORIZON):
                error = error_matrix[i, j]
                text_color = 'white' if error > 1.5 else 'black'
                plt.text(j, i, f'{error:.2f}', ha='center', va='center', color=text_color)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'error_heatmap.png'))
        plt.close()
    
    def export_to_excel(self):
        """Export validation results to Excel files"""
        if not self.matched_data or not self.errors:
            logger.error("Cannot export to Excel: missing data")
            return False
        
        logger.info("Exporting validation results to Excel...")
        
        # Create Excel directory
        excel_dir = os.path.join(self.output_dir, "excel")
        os.makedirs(excel_dir, exist_ok=True)
        
        # 1. Create summary Excel file
        self._create_summary_excel(excel_dir)
        
        # 2. Create detailed time series Excel file
        self._create_timeseries_excel(excel_dir)
        
        logger.info(f"Excel exports created in {excel_dir}")
        return True
    
    def _create_summary_excel(self, excel_dir):
        """Create Excel summary of validation results"""
        filename = os.path.join(excel_dir, 'validation_summary.xlsx')
        
        # Create a Pandas Excel writer
        with pd.ExcelWriter(filename) as writer:
            # 1. Overall metrics sheet
            overall_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'Max Error'],
                'Overall Value': [
                    self.errors['overall'].get('mae', 0),
                    self.errors['overall'].get('rmse', 0),
                    self.errors['overall'].get('max_error', 0)
                ]
            })
            overall_df.to_excel(writer, sheet_name='Overall Metrics', index=False)
            
            # 2. Zone metrics sheet
            zone_data = {
                'Zone': ZONE_NAMES,
                'MAE': [self.errors['zone_errors'][zone].get('mae', 0) for zone in ZONE_NAMES],
                'RMSE': [self.errors['zone_errors'][zone].get('rmse', 0) for zone in ZONE_NAMES],
                'Max Error': [self.errors['zone_errors'][zone].get('max_error', 0) for zone in ZONE_NAMES]
            }
            zone_df = pd.DataFrame(zone_data)
            zone_df.to_excel(writer, sheet_name='Zone Metrics', index=False)
            
            # 3. Step metrics sheet
            step_data = {
                'Step': list(range(PREDICTION_HORIZON)),
                'MAE': [self.errors['step_errors']['overall'][step].get('mae', 0) for step in range(PREDICTION_HORIZON)],
                'RMSE': [self.errors['step_errors']['overall'][step].get('rmse', 0) for step in range(PREDICTION_HORIZON)],
                'Max Error': [self.errors['step_errors']['overall'][step].get('max_error', 0) for step in range(PREDICTION_HORIZON)]
            }
            step_df = pd.DataFrame(step_data)
            step_df.to_excel(writer, sheet_name='Step Metrics', index=False)
            
            # 4. Zone-Step metrics sheet
            zone_step_data = []
            
            for zone in ZONE_NAMES:
                for step in range(PREDICTION_HORIZON):
                    zone_step_data.append({
                        'Zone': zone,
                        'Step': step,
                        'MAE': self.errors['step_errors']['zones'][zone][step].get('mae', 0),
                        'RMSE': self.errors['step_errors']['zones'][zone][step].get('rmse', 0),
                        'Max Error': self.errors['step_errors']['zones'][zone][step].get('max_error', 0)
                    })
            
            zone_step_df = pd.DataFrame(zone_step_data)
            zone_step_df.to_excel(writer, sheet_name='Zone-Step Metrics', index=False)
            
            # 5. Validation metadata
            metadata = pd.DataFrame({
                'Property': [
                    'Validation Date',
                    'Number of Predictions',
                    'Number of Temperature Records',
                    'Number of Matched Data Points',
                    'Time Step (seconds)',
                    'Prediction Horizon (steps)',
                    'Comfort Min (°F)',
                    'Comfort Max (°F)'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(self.predictions),
                    len(self.actual_temperatures),
                    len(self.matched_data),
                    TIME_STEP,
                    PREDICTION_HORIZON,
                    COMFORT_MIN,
                    COMFORT_MAX
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    def _create_timeseries_excel(self, excel_dir):
        """Create Excel file with full time series data"""
        filename = os.path.join(excel_dir, 'validation_timeseries.xlsx')
        
        # Create a Pandas Excel writer
        with pd.ExcelWriter(filename) as writer:
            # 1. Prepare time series data for each matched prediction
            for i, match in enumerate(self.matched_data):
                # Only include a sample of predictions to keep file size manageable
                if i % 5 != 0 and i != len(self.matched_data) - 1:  # Take every 5th plus the last one
                    continue
                
                pred_time = match['prediction_time']
                sheet_name = f'Pred_{pred_time.strftime("%m%d_%H%M")}'
                
                # Create data structure for this prediction
                pred_data = []
                
                # Time steps
                timestamps = [pred_time + timedelta(seconds=TIME_STEP * step) for step in range(PREDICTION_HORIZON)]
                
                # For each zone
                for zone_idx, zone in enumerate(ZONE_NAMES):
                    # Predicted temperatures
                    pred_temps = match['predicted_trajectories'][zone_idx]
                    
                    # Actual temperatures (match by closest timestamp)
                    actual_temps = [None] * PREDICTION_HORIZON
                    for step, actual in enumerate(match['matched_actuals']):
                        if actual is not None and zone in actual['temperatures']:
                            actual_temps[step] = actual['temperatures'][zone]
                    
                    # Calculate errors
                    errors = [abs(p - a) if a is not None else None for p, a in zip(pred_temps, actual_temps)]
                    
                    # Add rows for this zone
                    for step in range(PREDICTION_HORIZON):
                        if step < len(pred_temps):
                            pred_data.append({
                                'Zone': zone,
                                'Step': step,
                                'Timestamp': timestamps[step],
                                'Predicted (°F)': pred_temps[step],
                                'Actual (°F)': actual_temps[step],
                                'Error (°F)': errors[step]
                            })
                
                # Convert to DataFrame and write to sheet
                df = pd.DataFrame(pred_data)
                
                # Limit sheet name length (Excel limitation)
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 2. Create a predictions summary sheet
            pred_summary = []
            
            for match in self.matched_data:
                # Extract basic info
                pred_time = match['prediction_time']
                
                # Calculate average errors for this prediction
                zone_errors = {}
                for zone_idx, zone in enumerate(ZONE_NAMES):
                    errors = []
                    for step, actual in enumerate(match['matched_actuals']):
                        if actual is not None and zone in actual['temperatures']:
                            if step < len(match['predicted_trajectories'][zone_idx]):
                                pred = match['predicted_trajectories'][zone_idx][step]
                                act = actual['temperatures'][zone]
                                if act is not None:
                                    errors.append(abs(pred - act))
                    
                    zone_errors[zone] = np.mean(errors) if errors else None
                
                # Add to summary
                pred_summary.append({
                    'Prediction Time': pred_time,
                    'AC Decision': 'ON' if match['ac_decision'] else 'OFF',
                    'Average Error (°F)': np.mean([e for e in zone_errors.values() if e is not None]),
                    **{f"{zone} Error (°F)": zone_errors[zone] for zone in ZONE_NAMES}
                })
            
            # Convert to DataFrame and write to sheet
            pred_summary_df = pd.DataFrame(pred_summary)
            pred_summary_df.to_excel(writer, sheet_name='Predictions Summary', index=False)
            
            # 3. Create an overall time series sheet
            # This combines all actual measurements with their matched predictions
            overall_series = []
            
            for match in self.matched_data:
                pred_time = match['prediction_time']
                
                for step, actual in enumerate(match['matched_actuals']):
                    if actual is not None:
                        for zone_idx, zone in enumerate(ZONE_NAMES):
                            if zone in actual['temperatures'] and actual['temperatures'][zone] is not None:
                                if step < len(match['predicted_trajectories'][zone_idx]):
                                    overall_series.append({
                                        'Timestamp': actual['actual_time'],
                                        'Zone': zone,
                                        'Prediction Time': pred_time,
                                        'Step': step,
                                        'Predicted (°F)': match['predicted_trajectories'][zone_idx][step],
                                        'Actual (°F)': actual['temperatures'][zone],
                                        'Error (°F)': abs(match['predicted_trajectories'][zone_idx][step] - actual['temperatures'][zone])
                                    })
            
            # Sort by timestamp
            overall_series_df = pd.DataFrame(overall_series)
            if not overall_series_df.empty:
                overall_series_df = overall_series_df.sort_values('Timestamp')
                overall_series_df.to_excel(writer, sheet_name='Complete Time Series', index=False)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive validation report with text summary"""
        if not self.errors:
            logger.error("Cannot generate report: missing error data")
            return False
        
        logger.info("Generating comprehensive validation report...")
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create text report
        report_path = os.path.join(report_dir, "validation_report.md")
        
        with open(report_path, "w") as f:
            f.write("# MPC Thermal Model Validation Report\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report analyzes the prediction accuracy of the MPC thermal model used in the HVAC control system.\n")
            f.write(f"The validation is based on {len(self.matched_data)} matched prediction-actual pairs.\n\n")
            
            # Overall metrics
            f.write("### Overall Prediction Performance\n\n")
            f.write(f"- **Mean Absolute Error (MAE)**: {self.errors['overall'].get('mae', 0):.2f}°F\n")
            f.write(f"- **Root Mean Square Error (RMSE)**: {self.errors['overall'].get('rmse', 0):.2f}°F\n")
            f.write(f"- **Maximum Error**: {self.errors['overall'].get('max_error', 0):.2f}°F\n\n")
            
            # Performance interpretation
            mae = self.errors['overall'].get('mae', 0)
            if mae < 1.0:
                assessment = "excellent"
            elif mae < 2.0:
                assessment = "good"
            elif mae < 3.0:
                assessment = "acceptable"
            else:
                assessment = "needs improvement"
            
            f.write(f"The overall model accuracy is considered **{assessment}** for MPC applications. ")
            f.write(f"The average prediction error of {mae:.2f}°F ")
            
            if mae < 2.0:
                f.write("is within the acceptable range for effective control decisions.\n\n")
            else:
                f.write("exceeds the ideal threshold for optimal control decisions.\n\n")
            
            # Zone-specific performance
            f.write("## Zone-Specific Prediction Performance\n\n")
            f.write("| Zone | MAE (°F) | RMSE (°F) | Max Error (°F) | Assessment |\n")
            f.write("|------|----------|-----------|----------------|------------|\n")
            
            for zone in ZONE_NAMES:
                zone_mae = self.errors['zone_errors'][zone].get('mae', 0)
                zone_rmse = self.errors['zone_errors'][zone].get('rmse', 0)
                zone_max = self.errors['zone_errors'][zone].get('max_error', 0)
                
                if zone_mae < 1.0:
                    zone_assessment = "Excellent"
                elif zone_mae < 2.0:
                    zone_assessment = "Good"
                elif zone_mae < 3.0:
                    zone_assessment = "Acceptable"
                else:
                    zone_assessment = "Needs Improvement"
                
                f.write(f"| {zone} | {zone_mae:.2f} | {zone_rmse:.2f} | {zone_max:.2f} | {zone_assessment} |\n")
            
            f.write("\n")
            
            # Prediction horizon performance
            f.write("## Prediction Horizon Performance\n\n")
            f.write("This analysis shows how prediction accuracy changes over the prediction horizon.\n\n")
            f.write("| Step | Time (min) | MAE (°F) | RMSE (°F) |\n")
            f.write("|------|------------|----------|----------|\n")
            
            for step in range(PREDICTION_HORIZON):
                step_mae = self.errors['step_errors']['overall'][step].get('mae', 0)
                step_rmse = self.errors['step_errors']['overall'][step].get('rmse', 0)
                time_min = (step * TIME_STEP) / 60
                
                f.write(f"| {step} | {time_min:.0f} | {step_mae:.2f} | {step_rmse:.2f} |\n")
            
            f.write("\n")
            
            # Key findings and observations
            f.write("## Key Findings and Observations\n\n")
            
            # Find worst and best performing zones
            zone_maes = [(zone, self.errors['zone_errors'][zone].get('mae', 0)) for zone in ZONE_NAMES]
            zone_maes.sort(key=lambda x: x[1])
            best_zone = zone_maes[0]
            worst_zone = zone_maes[-1]
            
            # Analyze how error changes with prediction horizon
            step_maes = [self.errors['step_errors']['overall'][step].get('mae', 0) for step in range(PREDICTION_HORIZON)]
            error_increase = (step_maes[-1] - step_maes[0]) if step_maes[0] > 0 else 0
            
            f.write(f"1. **Zone-Specific Performance**: Zone {best_zone[0]} shows the best prediction accuracy (MAE: {best_zone[1]:.2f}°F), while Zone {worst_zone[0]} shows the worst (MAE: {worst_zone[1]:.2f}°F).\n\n")
            
            if error_increase > 0:
                f.write(f"2. **Horizon Impact**: Prediction error increases by {error_increase:.2f}°F over the {PREDICTION_HORIZON} step horizon. ")
                
                if error_increase < 1.0:
                    f.write("This indicates good model stability over the prediction horizon.\n\n")
                elif error_increase < 2.0:
                    f.write("This increase is acceptable for MPC applications.\n\n")
                else:
                    f.write("This substantial increase suggests limitations in long-horizon predictions.\n\n")
            else:
                f.write(f"2. **Horizon Impact**: Prediction error remains stable over the prediction horizon, indicating excellent model stability.\n\n")
            
            # Overall assessment
            f.write("## Conclusion and Recommendations\n\n")
            
            if mae < 2.0:
                f.write("The thermal model demonstrates **sufficient accuracy** for effective MPC control. ")
                if worst_zone[1] > 3.0:
                    f.write(f"However, Zone {worst_zone[0]} shows higher prediction errors and may benefit from model refinement.\n\n")
                else:
                    f.write("All zones are predicted with reasonable accuracy.\n\n")
                    
                if error_increase > 2.0:
                    f.write("While short-term predictions are accurate, the increasing error over the prediction horizon suggests")
                    f.write(" that the MPC algorithm may benefit from a shorter horizon or increased weight on near-term predictions.\n\n")
            else:
                f.write("The thermal model shows **moderate accuracy** with room for improvement. ")
                f.write("Consider the following refinements:\n\n")
                f.write("1. Review thermal coupling parameters, especially for zones with higher errors\n")
                f.write("2. Check for potential external disturbances not captured in the model\n")
                f.write("3. Consider adding adaptive parameters to improve model accuracy over time\n\n")
            
            # Reference to supporting files
            f.write("## Supporting Documentation\n\n")
            f.write("Detailed validation data, visualizations, and Excel exports are available in the following locations:\n\n")
            f.write(f"- Visualizations: `{os.path.abspath(os.path.join(self.output_dir, 'visualizations'))}`\n")
            f.write(f"- Excel Reports: `{os.path.abspath(os.path.join(self.output_dir, 'excel'))}`\n\n")
        
        logger.info(f"Comprehensive report generated at {report_path}")
        return True
    
    def run_validation(self):
        """Run the complete validation process"""
        logger.info("Starting MPC thermal model validation...")
        
        # 1. Load data
        if not self.load_mpc_predictions():
            logger.error("Failed to load MPC predictions")
            return False
        
        if not self.load_actual_temperatures():
            logger.error("Failed to load actual temperature measurements")
            return False
        
        # 2. Match predictions with actuals
        if not self.match_predictions_with_actuals():
            logger.error("Failed to match predictions with actual measurements")
            return False
        
        # 3. Calculate errors
        if not self.calculate_prediction_errors():
            logger.error("Failed to calculate prediction errors")
            return False
        
        # 4. Create visualizations
        if not self.create_visualizations():
            logger.warning("Failed to create visualizations")
        
        # 5. Export to Excel
        if not self.export_to_excel():
            logger.warning("Failed to export to Excel")
        
        # 6. Generate comprehensive report
        if not self.generate_comprehensive_report():
            logger.warning("Failed to generate comprehensive report")
        
        logger.info("MPC thermal model validation completed successfully")
        return True


def add_logging_to_thermal_mpc(output_dir="data/mpc_predictions"):
    """
    Add logging code to thermal_mpc.py to record predictions
    
    This function generates the code that needs to be added to the
    optimize_ac_damper_control function in thermal_mpc.py
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the code to add
    code = """
    # After solving optimization, before returning results
    if T_pred is not None:
        # Log the prediction and current state for validation
        import os
        import json
        from datetime import datetime
        
        timestamp = datetime.now().isoformat()
        prediction_log = {
            "timestamp": timestamp,
            "current_state": x_current.tolist() if x_current is not None else None,
            "predicted_trajectories": T_pred.tolist() if T_pred is not None else None,
            "ac_decision": ac_first,
            "damper_decisions": damper_first,
            "ambient_temp": future_amb[0] if future_amb else None
        }
        
        # Save to file (ensure directory exists)
        os.makedirs("data/mpc_predictions", exist_ok=True)
        with open(f"data/mpc_predictions/pred_{timestamp.replace(':', '-')}.json", "w") as f:
            json.dump(prediction_log, f, indent=2)
    """
    
    return code


def main():
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='MPC Thermal Model Validation Tool')
    parser.add_argument('--data-dir', default='data', help='Directory containing data files')
    parser.add_argument('--output-dir', default='validation_results', help='Directory for output files')
    parser.add_argument('--generate-code', action='store_true', help='Generate logging code to add to thermal_mpc.py')
    
    args = parser.parse_args()
    
    # If generating code is requested, print it and exit
    if args.generate_code:
        code = add_logging_to_thermal_mpc()
        print("\nAdd the following code to the optimize_ac_damper_control function in thermal_mpc.py:")
        print("Place it right before the 'return ac_first, damper_first, T_pred' line\n")
        print(code)
        return
    
    # Create validator and run validation
    validator = MPCValidator(data_dir=args.data_dir, output_dir=args.output_dir)
    success = validator.run_validation()
    
    if success:
        print("\nValidation completed successfully!")
        print(f"Results available in: {os.path.abspath(args.output_dir)}")
    else:
        print("\nValidation failed. See log for details.")


if __name__ == "__main__":
    main()
