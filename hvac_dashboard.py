import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime, timedelta
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HVAC_Dashboard")

class HVACDashboard:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.temp_dir = os.path.join(data_dir, "temperatures")
        self.control_dir = os.path.join(data_dir, "control_actions")
        self.update_interval = 10  # seconds
        
        # Define comfort bounds
        self.comfort_min_temp = 70.0  # °F
        self.comfort_max_temp = 80.0  # °F
        
        # Define the correct zone-to-damper mapping
        self.zone_to_damper = {
            "TR1": 6,  # Damper 7
            "TR2": 4,  # Damper 5
            "TR3": 2,  # Damper 3
            "TR4": 0,  # Damper 1
            "BR1": 7,  # Damper 8
            "BR2": 5,  # Damper 6
            "BR3": 3,  # Damper 4
            "BR4": 1,  # Damper 2
        }
        # Reverse mapping for display purposes
        self.damper_to_zone = {v: k for k, v in self.zone_to_damper.items()}
        
        # Data storage
        self.temperatures = {
            "TR1": [], "TR2": [], "TR3": [], "TR4": [],
            "BR1": [], "BR2": [], "BR3": [], "BR4": [],
            "Ambient": []
        }
        self.timestamps = []
        self.ac_status = []
        self.ac_timestamps = []
        # Store dampers by actual physical number (0-7) for clarity
        self.damper_states = {i: [] for i in range(8)}
        self.damper_timestamps = []
        
        # Energy usage data
        self.energy_data = {
            "power": [],
            "voltage": [],
            "current": [],
            "energy_timestamps": []
        }
        
        # Comfort violation tracking
        self.comfort_violations = {
            zone: {"too_cold": [], "too_hot": [], "total": 0} 
            for zone in ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
        }
        
        # Initialize data
        self.load_historical_data()
        
        # Create the Dash app
        self.app = dash.Dash(__name__, title="HVAC Monitoring System")
        self.setup_layout()
        self.setup_callbacks()
    
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string from various formats"""
        try:
            # First try the standard format
            return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            try:
                # Try ISO format
                return datetime.fromisoformat(timestamp_str)
            except ValueError:
                try:
                    # Try date only format
                    return datetime.strptime(timestamp_str, '%Y-%m-%d')
                except ValueError:
                    # If we can't parse it directly, try to extract with regex
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', timestamp_str)
                    time_match = re.search(r'(\d{2})[:-](\d{2})[:-](\d{2})', timestamp_str)
                    
                    if date_match:
                        date_str = date_match.group(1)
                        
                        if time_match:
                            # We have both date and time
                            hour, minute, second = time_match.groups()
                            timestamp = f"{date_str}T{hour}:{minute}:{second}"
                            return datetime.fromisoformat(timestamp)
                        else:
                            # Date only
                            return datetime.strptime(date_str, '%Y-%m-%d')
                    
                    # If all else fails, raise the exception
                    raise ValueError(f"Cannot parse timestamp: {timestamp_str}")
    
    def extract_timestamp_from_filename(self, filename):
        """Extract timestamp from filename using robust pattern matching"""
        try:
            # First, try standard pattern - temps_YYYY-MM-DD_HH-MM-SS.json
            base = os.path.basename(filename)
            name_parts = base.split('_')
            
            # Check if we have a typical format with underscores
            if len(name_parts) >= 3:
                # Extract date and time parts
                date_part = name_parts[1]
                time_part = name_parts[2].split('.')[0]  # Remove extension
                
                # Try to parse as YYYY-MM-DD_HH-MM-SS
                return self.parse_timestamp(f"{date_part}_{time_part}")
            
            # If that fails, try to extract any date and time using regex
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', base)
            time_match = re.search(r'(\d{2})[-:](\d{2})[-:](\d{2})', base)
            
            if date_match:
                date_str = date_match.group(1)
                
                if time_match:
                    # We have both date and time
                    hour, minute, second = time_match.groups()
                    return datetime.strptime(f"{date_str} {hour}:{minute}:{second}", '%Y-%m-%d %H:%M:%S')
                else:
                    # Date only - set time to noon
                    return datetime.strptime(f"{date_str} 12:00:00", '%Y-%m-%d %H:%M:%S')
            
            # If all else fails
            logger.warning(f"Could not parse timestamp from filename: {base}")
            # Return a fallback timestamp
            return datetime.now() - timedelta(days=1)
                
        except Exception as e:
            logger.error(f"Error extracting timestamp from {filename}: {e}")
            # Return a fallback timestamp
            return datetime.now() - timedelta(days=1)
    
    def parse_timestamp_from_data(self, data):
        """Try to extract timestamp from data dict"""
        try:
            if 'timestamp' in data:
                return self.parse_timestamp(data['timestamp'])
            else:
                # Look for any field that might contain a timestamp
                for key in ['time', 'date', 'created', 'recorded']:
                    if key in data:
                        return self.parse_timestamp(data[key])
            
            # If we can't find a timestamp, return None
            return None
        except Exception as e:
            logger.error(f"Error parsing timestamp from data: {e}")
            return None
    
    def check_comfort_violations(self, zone, temp):
        """Check if a temperature reading violates comfort bounds"""
        if temp is None:
            return
            
        if temp < self.comfort_min_temp:
            self.comfort_violations[zone]["too_cold"].append(1)
            self.comfort_violations[zone]["total"] += 1
        else:
            self.comfort_violations[zone]["too_cold"].append(0)
            
        if temp > self.comfort_max_temp:
            self.comfort_violations[zone]["too_hot"].append(1)
            self.comfort_violations[zone]["total"] += 1
        else:
            self.comfort_violations[zone]["too_hot"].append(0)
    
    def load_historical_data(self):
        """Load historical data from files"""
        try:
            # Load temperature data
            temp_files = sorted(glob.glob(os.path.join(self.temp_dir, "temps_*.json")))
            for file in temp_files[-100:]:  # Load last 100 files for initial display
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        
                        # Try to get timestamp from data, or fallback to filename
                        timestamp = self.parse_timestamp_from_data(data)
                        if timestamp is None:
                            timestamp = self.extract_timestamp_from_filename(file)
                        
                        self.timestamps.append(timestamp)
                        
                        for zone in self.temperatures.keys():
                            # Navigate through nested dictionaries if needed
                            if 'temperatures' in data and zone in data['temperatures']:
                                temp = data['temperatures'].get(zone)
                            else:
                                temp = data.get(zone)
                            
                            self.temperatures[zone].append(temp)
                            
                            # Check for comfort violations
                            if zone != "Ambient":
                                self.check_comfort_violations(zone, temp)
                except Exception as e:
                    logger.error(f"Error processing temperature file {file}: {e}")
            
            # Load AC control data
            ac_files = sorted(glob.glob(os.path.join(self.control_dir, "ac_*.json")))
            for file in ac_files[-100:]:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        
                        # Try to get timestamp from data, or fallback to filename
                        timestamp = self.parse_timestamp_from_data(data)
                        if timestamp is None:
                            timestamp = self.extract_timestamp_from_filename(file)
                        
                        self.ac_timestamps.append(timestamp)
                        
                        # Try different fields that might contain the AC status
                        if 'action' in data:
                            self.ac_status.append(1 if data['action'] == "on" else 0)
                        elif 'ac_on' in data:
                            self.ac_status.append(1 if data['ac_on'] else 0)
                        elif 'status' in data:
                            self.ac_status.append(1 if data['status'] == "on" else 0)
                        else:
                            # Default to off if we can't determine
                            self.ac_status.append(0)
                except Exception as e:
                    logger.error(f"Error processing AC file {file}: {e}")
            
            # Load damper control data
            damper_files = sorted(glob.glob(os.path.join(self.control_dir, "dampers_*.json")))
            for file in damper_files[-100:]:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        
                        # Try to get timestamp from data, or fallback to filename
                        timestamp = self.parse_timestamp_from_data(data)
                        if timestamp is None:
                            timestamp = self.extract_timestamp_from_filename(file)
                        
                        self.damper_timestamps.append(timestamp)
                        
                        # Extract damper states - handle both zone-ordered and damper-ordered data
                        if 'damper_states' in data:
                            # This is already in damper order
                            damper_states = data['damper_states']
                            for i in range(8):
                                if i < len(damper_states):
                                    self.damper_states[i].append(damper_states[i])
                                else:
                                    self.damper_states[i].append(None)
                        elif 'zone_states' in data:
                            # This is in zone order, need to map to damper order
                            zone_states = data['zone_states']
                            # Initialize all dampers as closed
                            damper_states = [0] * 8
                            
                            # Map zone states to damper states
                            zone_names = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
                            for i, zone in enumerate(zone_names):
                                if i < len(zone_states):
                                    damper_idx = self.zone_to_damper[zone]
                                    damper_states[damper_idx] = zone_states[i]
                            
                            # Store the mapped states
                            for i in range(8):
                                self.damper_states[i].append(damper_states[i])
                        else:
                            # Try to find individual damper states
                            for i in range(8):
                                if f"damper_{i}" in data:
                                    self.damper_states[i].append(1 if data[f"damper_{i}"] else 0)
                                else:
                                    self.damper_states[i].append(None)
                except Exception as e:
                    logger.error(f"Error processing damper file {file}: {e}")
            
            # Load energy data
            energy_files = sorted(glob.glob(os.path.join(self.control_dir, "energy_*.json")))
            for file in energy_files[-100:]:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        
                        # Try to get timestamp from data, or fallback to filename
                        timestamp = self.parse_timestamp_from_data(data)
                        if timestamp is None:
                            timestamp = self.extract_timestamp_from_filename(file)
                        
                        self.energy_data["energy_timestamps"].append(timestamp)
                        
                        # Extract energy data
                        if 'energy_data' in data:
                            energy_data = data['energy_data']
                            self.energy_data["power"].append(energy_data.get("power", 0))
                            self.energy_data["voltage"].append(energy_data.get("voltage", 0))
                            self.energy_data["current"].append(energy_data.get("current", 0))
                        else:
                            # Try to find energy data directly in the file
                            self.energy_data["power"].append(data.get("power", 0))
                            self.energy_data["voltage"].append(data.get("voltage", 0))
                            self.energy_data["current"].append(data.get("current", 0))
                except Exception as e:
                    logger.error(f"Error processing energy file {file}: {e}")
            
            logger.info(f"Loaded {len(self.timestamps)} temperature records, "
                        f"{len(self.ac_timestamps)} AC records, "
                        f"{len(self.damper_timestamps)} damper records, and "
                        f"{len(self.energy_data['energy_timestamps'])} energy records")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def update_data(self):
        """Check for new data files and update the stored data"""
        try:
            # Update temperature data
            if self.timestamps:
                latest_temp_time = self.timestamps[-1]
                new_temp_files = sorted(glob.glob(os.path.join(self.temp_dir, "temps_*.json")))
                for file in new_temp_files:
                    try:
                        # Extract timestamp from filename
                        file_timestamp = self.extract_timestamp_from_filename(file)
                        
                        if file_timestamp > latest_temp_time:
                            with open(file, 'r') as f:
                                data = json.load(f)
                                self.timestamps.append(file_timestamp)
                                
                                for zone in self.temperatures.keys():
                                    # Navigate through nested dictionaries if needed
                                    if 'temperatures' in data and zone in data['temperatures']:
                                        temp = data['temperatures'].get(zone)
                                    else:
                                        temp = data.get(zone)
                                    
                                    self.temperatures[zone].append(temp)
                                    
                                    # Check for comfort violations
                                    if zone != "Ambient":
                                        self.check_comfort_violations(zone, temp)
                    except Exception as e:
                        logger.error(f"Error processing new temperature file {file}: {e}")
            
            # Update AC control data - similar robust pattern
            if self.ac_timestamps:
                latest_ac_time = self.ac_timestamps[-1]
                new_ac_files = sorted(glob.glob(os.path.join(self.control_dir, "ac_*.json")))
                for file in new_ac_files:
                    try:
                        # Extract timestamp from filename
                        file_timestamp = self.extract_timestamp_from_filename(file)
                        
                        if file_timestamp > latest_ac_time:
                            with open(file, 'r') as f:
                                data = json.load(f)
                                self.ac_timestamps.append(file_timestamp)
                                
                                # Try different fields that might contain the AC status
                                if 'action' in data:
                                    self.ac_status.append(1 if data['action'] == "on" else 0)
                                elif 'ac_on' in data:
                                    self.ac_status.append(1 if data['ac_on'] else 0)
                                elif 'status' in data:
                                    self.ac_status.append(1 if data['status'] == "on" else 0)
                                else:
                                    # Default to off if we can't determine
                                    self.ac_status.append(0)
                    except Exception as e:
                        logger.error(f"Error processing new AC file {file}: {e}")
            
            # Update damper control data - similar robust pattern
            if self.damper_timestamps:
                latest_damper_time = self.damper_timestamps[-1]
                new_damper_files = sorted(glob.glob(os.path.join(self.control_dir, "dampers_*.json")))
                for file in new_damper_files:
                    try:
                        # Extract timestamp from filename
                        file_timestamp = self.extract_timestamp_from_filename(file)
                        
                        if file_timestamp > latest_damper_time:
                            with open(file, 'r') as f:
                                data = json.load(f)
                                self.damper_timestamps.append(file_timestamp)
                                
                                # Extract damper states - handle both zone-ordered and damper-ordered data
                                if 'damper_states' in data:
                                    # This is already in damper order
                                    damper_states = data['damper_states']
                                    for i in range(8):
                                        if i < len(damper_states):
                                            self.damper_states[i].append(damper_states[i])
                                        else:
                                            self.damper_states[i].append(None)
                                elif 'zone_states' in data:
                                    # This is in zone order, need to map to damper order
                                    zone_states = data['zone_states']
                                    # Initialize all dampers as closed
                                    damper_states = [0] * 8
                                    
                                    # Map zone states to damper states
                                    zone_names = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
                                    for i, zone in enumerate(zone_names):
                                        if i < len(zone_states):
                                            damper_idx = self.zone_to_damper[zone]
                                            damper_states[damper_idx] = zone_states[i]
                                    
                                    # Store the mapped states
                                    for i in range(8):
                                        self.damper_states[i].append(damper_states[i])
                                else:
                                    # Try to find individual damper states
                                    for i in range(8):
                                        if f"damper_{i}" in data:
                                            self.damper_states[i].append(1 if data[f"damper_{i}"] else 0)
                                        else:
                                            self.damper_states[i].append(None)
                    except Exception as e:
                        logger.error(f"Error processing new damper file {file}: {e}")
            
            # Update energy data
            if self.energy_data["energy_timestamps"]:
                latest_energy_time = self.energy_data["energy_timestamps"][-1]
                new_energy_files = sorted(glob.glob(os.path.join(self.control_dir, "energy_*.json")))
                for file in new_energy_files:
                    try:
                        # Extract timestamp from filename
                        file_timestamp = self.extract_timestamp_from_filename(file)
                        
                        if file_timestamp > latest_energy_time:
                            with open(file, 'r') as f:
                                data = json.load(f)
                                self.energy_data["energy_timestamps"].append(file_timestamp)
                                
                                # Extract energy data
                                if 'energy_data' in data:
                                    energy_data = data['energy_data']
                                    self.energy_data["power"].append(energy_data.get("power", 0))
                                    self.energy_data["voltage"].append(energy_data.get("voltage", 0))
                                    self.energy_data["current"].append(energy_data.get("current", 0))
                                else:
                                    # Try to find energy data directly in the file
                                    self.energy_data["power"].append(data.get("power", 0))
                                    self.energy_data["voltage"].append(data.get("voltage", 0))
                                    self.energy_data["current"].append(data.get("current", 0))
                    except Exception as e:
                        logger.error(f"Error processing new energy file {file}: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return False
    
    def setup_layout(self):
        """Define the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("HVAC Monitoring System", style={'textAlign': 'center'}),
            
            # Time range selector
            html.Div([
                html.Label("Time Range:"),
                dcc.RadioItems(
                    id='time-range',
                    options=[
                        {'label': 'Last Hour', 'value': '1H'},
                        {'label': 'Last 6 Hours', 'value': '6H'},
                        {'label': 'Last 24 Hours', 'value': '24H'},
                        {'label': 'All', 'value': 'ALL'}
                    ],
                    value='6H',
                    labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                )
            ], style={'margin': '20px'}),
            
            # Current status cards
            html.Div([
                html.H2("Current Status", style={'textAlign': 'center'}),
                html.Div([
                    # AC Status Card
                    html.Div([
                        html.H3("AC Status", style={'textAlign': 'center'}),
                        html.Div(id='ac-status-card')
                    ], className='status-card', style={'width': '30%', 'margin': '10px'}),
                    
                    # Current Power Card
                    html.Div([
                        html.H3("Power Consumption", style={'textAlign': 'center'}),
                        html.Div(id='power-status-card')
                    ], className='status-card', style={'width': '30%', 'margin': '10px'}),
                    
                    # Comfort Status Card
                    html.Div([
                        html.H3("Comfort Status", style={'textAlign': 'center'}),
                        html.Div(id='comfort-status-card')
                    ], className='status-card', style={'width': '30%', 'margin': '10px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={'margin': '20px'}),
            
            # Temperature graph
            html.Div([
                html.H2("Zone Temperatures", style={'textAlign': 'center'}),
                dcc.Graph(id='temperature-graph'),
            ], style={'margin': '20px'}),
            
            # AC Status and Power Usage
            html.Div([
                html.Div([
                    html.H2("AC Status", style={'textAlign': 'center'}),
                    dcc.Graph(id='ac-graph'),
                ], style={'width': '48%'}),
                html.Div([
                    html.H2("Power Consumption", style={'textAlign': 'center'}),
                    dcc.Graph(id='power-graph'),
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px'}),
            
            # Damper Status graph
            html.Div([
                html.H2("Damper Status", style={'textAlign': 'center'}),
                dcc.Graph(id='damper-graph'),
            ], style={'margin': '20px'}),
            
            # Comfort Violations
            html.Div([
                html.H2("Comfort Violations", style={'textAlign': 'center'}),
                dcc.Graph(id='comfort-graph'),
            ], style={'margin': '20px'}),
            
            # Hidden div for triggering data updates
            html.Div(id='trigger-update', style={'display': 'none'}),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # in milliseconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Set up Dash callbacks for interactive elements"""
        
        # Callback for updating data periodically
        @self.app.callback(
            Output('trigger-update', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_data_periodically(n):
            self.update_data()
            return f"Data updated at {datetime.now()}"
        
        # Callback for AC status card
        @self.app.callback(
            Output('ac-status-card', 'children'),
            Input('trigger-update', 'children')
        )
        def update_ac_status_card(update_time):
            if not self.ac_status or not self.ac_timestamps:
                return html.Div("No AC data available")
            
            current_status = self.ac_status[-1]
            status_text = "ON" if current_status else "OFF"
            status_color = "green" if current_status else "red"
            last_change_time = self.ac_timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")
            
            return html.Div([
                html.Div(status_text, style={
                    'fontSize': '24px', 
                    'fontWeight': 'bold',
                    'color': status_color,
                    'textAlign': 'center',
                    'margin': '10px'
                }),
                html.Div(f"Last updated: {last_change_time}", style={
                    'fontSize': '12px',
                    'textAlign': 'center'
                })
            ])
        
        # Callback for power status card
        @self.app.callback(
            Output('power-status-card', 'children'),
            Input('trigger-update', 'children')
        )
        def update_power_status_card(update_time):
            if not self.energy_data["power"] or not self.energy_data["energy_timestamps"]:
                return html.Div("No power data available")
            
            current_power = self.energy_data["power"][-1]
            last_update_time = self.energy_data["energy_timestamps"][-1].strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate estimated hourly cost (assuming $0.12 per kWh)
            hourly_cost = current_power * 0.12 / 1000
            
            return html.Div([
                html.Div(f"{current_power:.1f} W", style={
                    'fontSize': '24px', 
                    'fontWeight': 'bold',
                    'color': 'blue',
                    'textAlign': 'center',
                    'margin': '10px'
                }),
                html.Div(f"Est. cost: ${hourly_cost:.2f}/hr", style={
                    'fontSize': '16px',
                    'textAlign': 'center',
                    'margin': '5px'
                }),
                html.Div(f"Last updated: {last_update_time}", style={
                    'fontSize': '12px',
                    'textAlign': 'center'
                })
            ])
        
        # Callback for comfort status card
        @self.app.callback(
            Output('comfort-status-card', 'children'),
            Input('trigger-update', 'children')
        )
        def update_comfort_status_card(update_time):
            if not self.timestamps or not self.temperatures["TR1"]:
                return html.Div("No temperature data available")
            
            # Check current comfort status
            current_temps = {zone: temps[-1] if temps else None 
                             for zone, temps in self.temperatures.items() 
                             if zone != "Ambient"}
            
            # Count violations
            hot_zones = []
            cold_zones = []
            comfort_zones = []
            
            for zone, temp in current_temps.items():
                if temp is None:
                    continue
                if temp > self.comfort_max_temp:
                    hot_zones.append(zone)
                elif temp < self.comfort_min_temp:
                    cold_zones.append(zone)
                else:
                    comfort_zones.append(zone)
            
            # Determine overall status
            if hot_zones:
                status_text = "TOO HOT"
                status_color = "red"
            elif cold_zones:
                status_text = "TOO COLD"
                status_color = "blue"
            else:
                status_text = "COMFORTABLE"
                status_color = "green"
            
            # Count total zones with data
            total_zones = len([z for z in current_temps.values() if z is not None])
            comfort_percent = (len(comfort_zones) / total_zones * 100) if total_zones > 0 else 0
            
            return html.Div([
                html.Div(status_text, style={
                    'fontSize': '24px', 
                    'fontWeight': 'bold',
                    'color': status_color,
                    'textAlign': 'center',
                    'margin': '10px'
                }),
                html.Div(f"{len(comfort_zones)} of {total_zones} zones comfortable ({comfort_percent:.1f}%)", style={
                    'fontSize': '14px',
                    'textAlign': 'center',
                    'margin': '5px'
                }),
                html.Div(f"Range: {self.comfort_min_temp}°F - {self.comfort_max_temp}°F", style={
                    'fontSize': '12px',
                    'textAlign': 'center'
                })
            ])
        
        # Callback for temperature graph
        @self.app.callback(
            Output('temperature-graph', 'figure'),
            [Input('trigger-update', 'children'),
             Input('time-range', 'value')]
        )
        def update_temperature_graph(update_time, time_range):
            filtered_data = self.filter_data_by_time_range(self.timestamps, time_range)
            if not filtered_data:
                return go.Figure()
            
            start_idx = filtered_data['start_idx']
            end_idx = filtered_data['end_idx']
            
            # Safety check for empty datasets
            if start_idx >= end_idx or start_idx >= len(self.timestamps) or end_idx > len(self.timestamps):
                return go.Figure()
            
            fig = go.Figure()
            
            for zone, temps in self.temperatures.items():
                if temps and len(temps) >= end_idx:  # Only add zones that have data
                    # Filter out None values for better visualization
                    x_values = []
                    y_values = []
                    for i in range(start_idx, end_idx):
                        if temps[i] is not None:
                            x_values.append(self.timestamps[i])
                            y_values.append(temps[i])
                    
                    if x_values:  # Only add trace if we have valid data points
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='lines+markers',
                            name=zone
                        ))
            
            if start_idx < len(self.timestamps) and end_idx <= len(self.timestamps) and start_idx < end_idx:
                # Add comfort threshold lines
                fig.add_shape(
                    type="line",
                    x0=self.timestamps[start_idx],
                    y0=self.comfort_min_temp,
                    x1=self.timestamps[end_idx-1],
                    y1=self.comfort_min_temp,
                    line=dict(color="blue", width=2, dash="dash"),
                    name="Min Comfortable Temp"
                )
                
                fig.add_shape(
                    type="line",
                    x0=self.timestamps[start_idx],
                    y0=self.comfort_max_temp,
                    x1=self.timestamps[end_idx-1],
                    y1=self.comfort_max_temp,
                    line=dict(color="red", width=2, dash="dash"),
                    name="Max Comfortable Temp"
                )
                
                # Add damper threshold line
                fig.add_shape(
                    type="line",
                    x0=self.timestamps[start_idx],
                    y0=75,
                    x1=self.timestamps[end_idx-1],
                    y1=75,
                    line=dict(color="green", width=1, dash="dot"),
                    name="Damper Threshold (75°F)"
                )
            
            fig.update_layout(
                title="Zone Temperatures Over Time",
                xaxis_title="Time",
                yaxis_title="Temperature (°F)",
                legend_title="Zones",
                hovermode="closest"
            )
            
            return fig
        
        # Callback for AC status graph
        @self.app.callback(
            Output('ac-graph', 'figure'),
            [Input('trigger-update', 'children'),
             Input('time-range', 'value')]
        )
        def update_ac_graph(update_time, time_range):
            filtered_data = self.filter_data_by_time_range(self.ac_timestamps, time_range)
            if not filtered_data:
                return go.Figure()
            
            start_idx = filtered_data['start_idx']
            end_idx = filtered_data['end_idx']
            
            # Safety check for empty datasets
            if start_idx >= end_idx or start_idx >= len(self.ac_timestamps) or end_idx > len(self.ac_timestamps):
                return go.Figure()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.ac_timestamps[start_idx:end_idx],
                y=self.ac_status[start_idx:end_idx],
                mode='lines+markers',
                name='AC Status',
                line=dict(color='red', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="AC Status Over Time",
                xaxis_title="Time",
                yaxis_title="Status (1=ON, 0=OFF)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['OFF', 'ON']
                ),
                hovermode="closest"
            )
            
            # Set y-axis range with a bit of padding
            fig.update_yaxes(range=[-0.1, 1.1])
            
            return fig
        
        # Callback for power graph
        # Callback for power graph
        @self.app.callback(
            Output('power-graph', 'figure'),
            [Input('trigger-update', 'children'),
            Input('time-range', 'value')]
        )
        def update_power_graph(update_time, time_range):
            filtered_data = self.filter_data_by_time_range(self.energy_data["energy_timestamps"], time_range)
            if not filtered_data:
                return go.Figure()
            
            start_idx = filtered_data['start_idx']
            end_idx = filtered_data['end_idx']
            
            # Safety check for empty datasets
            if start_idx >= end_idx or start_idx >= len(self.energy_data["energy_timestamps"]) or end_idx > len(self.energy_data["energy_timestamps"]):
                return go.Figure()
            
            fig = go.Figure()
            
            # Add power trace
            fig.add_trace(go.Scatter(
                x=self.energy_data["energy_timestamps"][start_idx:end_idx],
                y=self.energy_data["power"][start_idx:end_idx],
                mode='lines',
                name='Power (W)',
                line=dict(color='blue', width=2)
            ))
            
            # Calculate cumulative energy usage
            if len(self.energy_data["power"]) > 1:
                # Convert W to kWh
                # For each interval, calculate energy = power * time_interval / 3600000
                energy_usage = []
                timestamps = self.energy_data["energy_timestamps"][start_idx:end_idx]
                power_values = self.energy_data["power"][start_idx:end_idx]
                
                # Initialize with the first point
                cumulative_energy = 0
                energy_usage.append(cumulative_energy)
                
                # Calculate cumulative energy
                for i in range(1, len(timestamps)):
                    # Time difference in hours
                    time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                    # Energy in kWh for this interval
                    energy = power_values[i-1] * time_diff / 1000
                    cumulative_energy += energy
                    energy_usage.append(cumulative_energy)
                
                # Add energy trace with secondary y-axis
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=energy_usage,
                    mode='lines',
                    name='Energy (kWh)',
                    yaxis='y2',
                    line=dict(color='green', width=2, dash='dot')
                ))
                
                # Update layout for dual y-axis
                fig.update_layout(
                    title="Power Consumption Over Time",
                    xaxis_title="Time",
                    yaxis=dict(
                        title=dict(
                            text="Power (W)",
                            font=dict(color="blue")
                        ),
                        tickfont=dict(color="blue")
                    ),
                    yaxis2=dict(
                        title=dict(
                            text="Energy (kWh)",
                            font=dict(color="green")
                        ),
                        tickfont=dict(color="green"),
                        anchor="x",
                        overlaying="y",
                        side="right"
                    ),
                    hovermode="x unified"
                )
            else:
                fig.update_layout(
                    title="Power Consumption Over Time",
                    xaxis_title="Time",
                    yaxis_title="Power (W)",
                    hovermode="closest"
                )
            
            return fig
        
        # Callback for damper status graph
        @self.app.callback(
            Output('damper-graph', 'figure'),
            [Input('trigger-update', 'children'),
             Input('time-range', 'value')]
        )
        def update_damper_graph(update_time, time_range):
            filtered_data = self.filter_data_by_time_range(self.damper_timestamps, time_range)
            if not filtered_data:
                return go.Figure()
            
            start_idx = filtered_data['start_idx']
            end_idx = filtered_data['end_idx']
            
            # Safety check for empty datasets
            if start_idx >= end_idx or start_idx >= len(self.damper_timestamps) or end_idx > len(self.damper_timestamps):
                return go.Figure()
            
            fig = go.Figure()
            
            # Colors for different dampers
            colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
            
            for i in range(8):
                states = self.damper_states[i]
                if states and len(states) >= end_idx:  # Only add dampers that have data
                    # Get the zone name for this damper for better labeling
                    zone_name = self.damper_to_zone.get(i, f"Damper {i+1}")
                    
                    # Add small offset to each damper for better visibility
                    offset = i * 0.05
                    
                    # Filter out None values for better visualization
                    x_values = []
                    y_values = []
                    for j in range(start_idx, end_idx):
                        if j < len(states) and states[j] is not None:
                            x_values.append(self.damper_timestamps[j])
                            y_values.append(states[j] + offset)
                    
                    if x_values:  # Only add trace if we have valid data points
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='lines+markers',
                            name=f"{zone_name} (Damper {i+1})",
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=8)
                        ))
            
            fig.update_layout(
                title="Damper Status Over Time (Correctly Labeled by Room)",
                xaxis_title="Time",
                yaxis_title="Status (1=OPEN, 0=CLOSED)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['CLOSED', 'OPEN']
                ),
                legend_title="Dampers",
                hovermode="closest"
            )
            
            # Set y-axis range with padding for offsets
            fig.update_yaxes(range=[-0.1, 1.5])
            
            return fig
        
        # Callback for comfort violations graph
        @self.app.callback(
            Output('comfort-graph', 'figure'),
            [Input('trigger-update', 'children'),
             Input('time-range', 'value')]
        )
        def update_comfort_graph(update_time, time_range):
            filtered_data = self.filter_data_by_time_range(self.timestamps, time_range)
            if not filtered_data:
                return go.Figure()
            
            start_idx = filtered_data['start_idx']
            end_idx = filtered_data['end_idx']
            
            # Safety check for empty datasets
            if start_idx >= end_idx or start_idx >= len(self.timestamps) or end_idx > len(self.timestamps):
                return go.Figure()
            
            fig = go.Figure()
            
            # Calculate percentage of time each zone is in comfort violation
            zones = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
            time_period = f"{self.timestamps[start_idx].strftime('%m/%d %H:%M')} - {self.timestamps[end_idx-1].strftime('%m/%d %H:%M')}"
            
            # Calculate comfort violation percentages
            hot_violations = []
            cold_violations = []
            
            for zone in zones:
                # Calculate percentage of time too hot
                hot_count = sum(self.comfort_violations[zone]["too_hot"][start_idx:end_idx])
                total_readings = end_idx - start_idx
                hot_percent = (hot_count / total_readings * 100) if total_readings > 0 else 0
                hot_violations.append(hot_percent)
                
                # Calculate percentage of time too cold
                cold_count = sum(self.comfort_violations[zone]["too_cold"][start_idx:end_idx])
                cold_percent = (cold_count / total_readings * 100) if total_readings > 0 else 0
                cold_violations.append(cold_percent)
            
            # Create stacked bar chart
            fig.add_trace(go.Bar(
                name='Too Hot',
                x=zones,
                y=hot_violations,
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                name='Too Cold',
                x=zones,
                y=cold_violations,
                marker_color='blue'
            ))
            
            # Configure layout
            fig.update_layout(
                title=f"Comfort Violations by Zone ({time_period})",
                xaxis_title="Zone",
                yaxis_title="% of Time Outside Comfort Range",
                barmode='stack',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            return fig
    
    def filter_data_by_time_range(self, timestamps, time_range):
        """Filter data based on selected time range"""
        if not timestamps:
            return None
        
        now = datetime.now()
        
        if time_range == '1H':
            cutoff = now - timedelta(hours=1)
        elif time_range == '6H':
            cutoff = now - timedelta(hours=6)
        elif time_range == '24H':
            cutoff = now - timedelta(hours=24)
        else:  # 'ALL'
            return {'start_idx': 0, 'end_idx': len(timestamps)}
        
        # Find index of first timestamp after cutoff
        start_idx = 0
        for i, ts in enumerate(timestamps):
            if ts >= cutoff:
                start_idx = i
                break
        
        return {'start_idx': start_idx, 'end_idx': len(timestamps)}
    
    def run(self, debug=False, host='0.0.0.0', port=8050):
        """Run the Dash app"""
        # Use app.run instead of app.run_server for newer Dash versions
        self.app.run(debug=debug, host=host, port=port)

# Run the dashboard if the script is executed directly
if __name__ == '__main__':
    dashboard = HVACDashboard()
    dashboard.run(debug=True)