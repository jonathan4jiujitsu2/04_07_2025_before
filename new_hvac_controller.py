"""
Enhanced HVAC Controller with Model Predictive Control (MPC)
Fixed to handle 3-value return from apply_control_actions
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import thermal MPC module
import thermal_mpc

# Import hardware interface
from hardware_interface import HardwareInterface
from hardware_interface_patch import patch_hardware_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hvac_controller.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HVAC_Controller")

class HVACController:
    """Enhanced HVAC controller with MPC integration and AC verification"""
    
    def __init__(self, data_dir="data"):
        """Initialize HVAC controller"""
        self.data_dir = data_dir
        self.control_interval = 300  # 5 minutes (same as dt in MPC)
        self.hardware = HardwareInterface(data_dir)
        patch_hardware_interface(self.hardware)
        self.thermal_system = None
        self.last_temperatures = None
        self.last_control_action = None
        self.last_energy_data = None
        self.ac_state_history = []  # Track AC state changes
        self.last_verified_ac_state = None  # Last verified state (from energy monitor)
        self.verification_attempts = 0  # Counter for verification attempts
        
        # Create data directories if they don't exist
        os.makedirs(os.path.join(data_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "control_actions"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "ac_verification"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "excel_data"), exist_ok=True)
        
        logger.info("HVAC Controller initialized")
    
    async def initialize_thermal_system(self):
        """Initialize the thermal system model with current temperature readings"""
        # First, get temperature readings from hardware
        temp_readings, _ = await self.hardware.read_temperatures()
        if not temp_readings:
            logger.error("Failed to get temperature readings for system initialization")
            # Create a default system
            self.thermal_system = thermal_mpc.create_real_system()
            return False
        
        # Store the temperature readings
        self.last_temperatures = temp_readings
        
        # Create thermal system with current temperatures
        self.thermal_system = thermal_mpc.create_real_system(temp_readings)
        
        logger.info(f"Thermal system initialized with temperatures: {temp_readings}")
        return True
    
    async def update_thermal_system(self):
        """Update the thermal system with current temperature readings"""
        # Get current temperature readings
        temp_readings, _ = await self.hardware.read_temperatures()
        if not temp_readings:
            logger.error("Failed to get temperature readings for system update")
            return False
        
        # Store the temperature readings
        self.last_temperatures = temp_readings
        
        # Update the thermal system
        self.thermal_system = thermal_mpc.update_thermal_system(self.thermal_system, temp_readings)
        
        logger.info("Thermal system updated with current temperatures")
        return True
    
    async def run_mpc(self):
        """Run MPC optimization using current thermal system state"""
        logger.info("Running MPC optimization")
        
        try:
            # Extract current temperatures from thermal system
            x_current = np.array([zone.initial_temperature for zone in self.thermal_system.zones])
            
            # Create future ambient temperature predictions
            # For simplicity, assuming constant ambient temperature
            ambient_temp = self.thermal_system.ambient_temperature(0)  # Current time
            future_amb = [ambient_temp] * 12  # 12 time steps (horizon)
            
            # Run MPC optimization
            dt = 300.0  # 5 minutes time step
            horizon = 12  # 1 hour prediction horizon
            bigM = 50.0  # Big-M value for logical constraints
            
            ac_on, damper_states, T_pred = thermal_mpc.optimize_ac_damper_control(
                self.thermal_system, x_current, future_amb, horizon, dt, bigM=bigM
            )
            
            # Store MPC results
            mpc_result = {
                "timestamp": datetime.now().isoformat(),
                "ac_on": ac_on,
                "damper_states": damper_states,
                "ambient_temperature": ambient_temp,
                "zone_temperatures": {
                    self.thermal_system.zones[i].name: x_current[i]
                    for i in range(len(self.thermal_system.zones))
                }
            }
            
            # Save MPC results to file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.data_dir, "control_actions", f"mpc_{timestamp}.json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(mpc_result, f, indent=2)
            
            logger.info(f"MPC result: AC={ac_on}, dampers={damper_states}")
            return ac_on, damper_states
            
        except Exception as e:
            logger.error(f"Error running MPC: {e}")
            return None, None
    
    async def apply_control_actions(self, ac_on, damper_states):
        """Apply control actions from MPC to hardware with verification"""
        if ac_on is None or damper_states is None:
            logger.warning("No valid control actions to apply")
            return False, None
        
        # Track AC state change
        ac_state_changed = (len(self.ac_state_history) == 0 or 
                           self.ac_state_history[-1] != ac_on)
        
        if ac_state_changed:
            self.ac_state_history.append(ac_on)
            logger.info(f"AC state change requested: {'ON' if ac_on else 'OFF'}")
        
        # Apply control actions to hardware - handling 3 return values
        success, energy_data, verification = await self.hardware.apply_control_actions(ac_on, damper_states)
        
        # Store control actions and energy data
        self.last_control_action = {
            "timestamp": datetime.now().isoformat(),
            "ac_on": ac_on,
            "damper_states": damper_states,
            "success": success
        }
        
        self.last_energy_data = energy_data
        
        # If AC state was changed and verification failed, try to retry
        if ac_state_changed and success and verification and not verification.get("verified", False):
            logger.warning("AC state verification failed, attempting to retry control")
            await self.retry_ac_control(ac_on)
        
        return success, energy_data
    
    async def retry_ac_control(self, desired_state, max_attempts=3):
        """
        Retry AC control if verification fails
        
        Args:
            desired_state: Desired AC state (True=on, False=off)
            max_attempts: Maximum number of retry attempts
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Retrying AC control to set state to {'ON' if desired_state else 'OFF'}")
        
        for attempt in range(max_attempts):
            logger.info(f"AC control retry attempt {attempt+1}/{max_attempts}")
            
            # Apply AC control
            action = "on" if desired_state else "off"
            await self.hardware.control_ac(action)
            
            # Wait for AC to respond (longer wait for retry)
            await asyncio.sleep(5)
            
            # Verify the state
            verification = await self.hardware.verify_ac_power(expected_on=desired_state)
            
            if verification.get("verified", False):
                logger.info(f"AC control retry successful on attempt {attempt+1}")
                return True
        
        logger.error(f"AC control retry failed after {max_attempts} attempts")
        
        # If we've had persistent verification failures, trigger an alert
        self.verification_attempts += 1
        if self.verification_attempts >= 3:
            await self.trigger_ac_control_alert()
            
        return False
    
    async def trigger_ac_control_alert(self):
        """Trigger an alert for persistent AC control issues"""
        logger.critical("ALERT: Persistent AC control issues detected!")
        
        # Create alert file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        alert_dir = os.path.join(self.data_dir, "alerts")
        os.makedirs(alert_dir, exist_ok=True)
        alert_file = os.path.join(alert_dir, f"ac_control_alert_{timestamp}.json")
        
        with open(alert_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "type": "AC_CONTROL_FAILURE",
                "message": "Persistent AC control verification failures detected",
                "verification_attempts": self.verification_attempts,
                "last_control_action": self.last_control_action,
                "last_energy_data": self.last_energy_data
            }, f, indent=2)
        
        logger.info(f"AC control alert created: {alert_file}")
    
    async def fallback_control_strategy(self):
        """Fallback control strategy when MPC fails"""
        logger.info("Using fallback control strategy")
        
        if not self.last_temperatures:
            logger.error("No temperature data available for fallback strategy")
            return False
        
        # Temperature thresholds
        critical_temp = 81.0
        high_temp = 80.0
        
        # Check temperatures and determine control actions
        zone_names = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
        damper_states = [0] * 8  # Default all closed
        
        # Check if any zone is above critical or high temperature
        critical_zones = []
        high_temp_zones = []
        
        for i, name in enumerate(zone_names):
            temp = self.last_temperatures.get(name)
            if temp is not None:
                if temp >= critical_temp:
                    critical_zones.append(name)
                    damper_states[i] = 1  # Open damper
                elif temp >= high_temp:
                    high_temp_zones.append(name)
                    damper_states[i] = 1  # Open damper
        
        # Determine AC state based on zone temperatures
        ac_on = len(critical_zones) > 0 or len(high_temp_zones) > 0
        
        logger.info(f"Fallback strategy: AC={'ON' if ac_on else 'OFF'}, " +
                  f"critical zones={critical_zones}, high temp zones={high_temp_zones}")
        
        # Apply control actions using the simplified method
        try:
            # Use the apply_fallback_actions method which returns only two values
            success, energy_data = await self.hardware.apply_fallback_actions(ac_on, damper_states)
            
            # Store control actions and energy data
            self.last_control_action = {
                "timestamp": datetime.now().isoformat(),
                "ac_on": ac_on,
                "damper_states": damper_states,
                "success": success,
                "mode": "fallback"
            }
            
            self.last_energy_data = energy_data
            
            return success
        except Exception as e:
            logger.error(f"Error in fallback strategy: {e}")
            return False
    
    async def generate_report(self):
        """Generate performance report, visualizations, and Excel data export with complete time series data"""
        logger.info("Generating system performance report with complete time series data")
        
        try:
            # Create directories for reports and Excel files
            reports_dir = os.path.join(self.data_dir, "reports")
            excel_dir = os.path.join(self.data_dir, "excel_data")
            os.makedirs(reports_dir, exist_ok=True)
            os.makedirs(excel_dir, exist_ok=True)
            
            # Current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Generate Excel report with pandas
            if self.last_temperatures:
                try:
                    import pandas as pd
                    
                    # Create or load existing dataframes
                    temp_excel = os.path.join(excel_dir, "temperature_data.xlsx")
                    control_excel = os.path.join(excel_dir, "control_actions_data.xlsx")
                    energy_excel = os.path.join(excel_dir, "energy_consumption_data.xlsx")
                    
                    # -- TEMPERATURE DATA --
                    # Current temperature readings
                    temp_data = {
                        'timestamp': timestamp,
                        'datetime': datetime.now().isoformat()
                    }
                    # Add temperature readings
                    for zone, temp in self.last_temperatures.items():
                        if temp is not None:
                            temp_data[f"{zone}_temp"] = temp
                    
                    # Create DataFrame for this data point
                    temp_df = pd.DataFrame([temp_data])
                    
                    # -- CONTROL ACTIONS DATA --
                    # Current control actions
                    control_data = {
                        'timestamp': timestamp,
                        'datetime': datetime.now().isoformat()
                    }
                    
                    if self.last_control_action:
                        control_data['ac_on'] = self.last_control_action.get('ac_on', False)
                        control_data['control_mode'] = self.last_control_action.get('mode', 'MPC')
                        
                        # Add damper states
                        damper_states = self.last_control_action.get('damper_states', [])
                        zone_names = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
                        for i, state in enumerate(damper_states):
                            if i < len(zone_names):
                                control_data[f"{zone_names[i]}_damper"] = state
                    
                    # Create DataFrame for this data point
                    control_df = pd.DataFrame([control_data])
                    
                    # -- ENERGY DATA --
                    # Current energy data
                    energy_data = {
                        'timestamp': timestamp,
                        'datetime': datetime.now().isoformat()
                    }
                    
                    if self.last_energy_data:
                        # Extract common energy metrics
                        if isinstance(self.last_energy_data, dict):
                            energy_data['power'] = self.last_energy_data.get('power', 0)
                            energy_data['voltage'] = self.last_energy_data.get('voltage', 0)
                            energy_data['current'] = self.last_energy_data.get('current', 0)
                            energy_data['ac_status'] = self.last_energy_data.get('ac_status', 'unknown')
                    
                    # Create DataFrame for this data point
                    energy_df = pd.DataFrame([energy_data])
                    
                    # -- DATA STORAGE STRATEGY --
                    # Function to append data to existing Excel or create new file if needed
                    def append_to_excel(new_df, file_path):
                        if os.path.exists(file_path):
                            try:
                                # Load existing data
                                existing_df = pd.read_excel(file_path)
                                
                                # Check if the new timestamp already exists in the data
                                if 'timestamp' in existing_df.columns and 'timestamp' in new_df.columns:
                                    new_timestamps = set(new_df['timestamp'].values)
                                    existing_timestamps = set(existing_df['timestamp'].values)
                                    
                                    # Only append if this is truly new data
                                    if not new_timestamps.issubset(existing_timestamps):
                                        # Append new data
                                        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                                        
                                        # Sort by timestamp to maintain chronological order
                                        if 'datetime' in updated_df.columns:
                                            updated_df['datetime'] = pd.to_datetime(updated_df['datetime'])
                                            updated_df = updated_df.sort_values('datetime')
                                        
                                        # Write back to Excel
                                        updated_df.to_excel(file_path, index=False)
                                        logger.info(f"Appended new data to {file_path}")
                                    else:
                                        logger.info(f"Data with timestamp {timestamp} already exists in {file_path}, skipping append")
                                else:
                                    # If there's no timestamp column, just append
                                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                                    updated_df.to_excel(file_path, index=False)
                                    logger.info(f"Appended new data to {file_path}")
                            except Exception as e:
                                logger.error(f"Error appending to existing file {file_path}: {e}")
                                # If there's any error with the existing file, create a new one
                                new_df.to_excel(file_path, index=False)
                                logger.info(f"Created new file {file_path} due to error with existing file")
                        else:
                            # Create new file
                            new_df.to_excel(file_path, index=False)
                            logger.info(f"Created new file {file_path}")
                    
                    # Append all data to the time series Excel files
                    append_to_excel(temp_df, temp_excel)
                    append_to_excel(control_df, control_excel)
                    append_to_excel(energy_df, energy_excel)
                    
                    # Save a combined snapshot just for this timestamp
                    combined_excel = os.path.join(excel_dir, f"hvac_data_snapshot_{timestamp}.xlsx")
                    with pd.ExcelWriter(combined_excel) as writer:
                        temp_df.to_excel(writer, sheet_name='Temperature', index=False)
                        control_df.to_excel(writer, sheet_name='Control Actions', index=False)
                        energy_df.to_excel(writer, sheet_name='Energy Consumption', index=False)
                    
                    # Additionally, create a consolidated historical file with all data in different sheets
                    try:
                        # Load existing temperature, control, and energy data
                        temp_df_all = pd.read_excel(temp_excel) if os.path.exists(temp_excel) else pd.DataFrame()
                        control_df_all = pd.read_excel(control_excel) if os.path.exists(control_excel) else pd.DataFrame()
                        energy_df_all = pd.read_excel(energy_excel) if os.path.exists(energy_excel) else pd.DataFrame()
                        
                        # Save to a combined historical data file
                        combined_historical = os.path.join(excel_dir, "hvac_historical_data.xlsx")
                        with pd.ExcelWriter(combined_historical) as writer:
                            temp_df_all.to_excel(writer, sheet_name='Temperature History', index=False)
                            control_df_all.to_excel(writer, sheet_name='Control History', index=False)
                            energy_df_all.to_excel(writer, sheet_name='Energy History', index=False)
                        
                        logger.info(f"Updated combined historical data file: {combined_historical}")
                    except Exception as e:
                        logger.error(f"Error creating combined historical file: {e}")
                    
                    logger.info(f"Excel time series data export complete. Files saved to {excel_dir}")
                    
                except ImportError:
                    logger.warning("Pandas not installed. Excel export skipped. Install with: pip install pandas openpyxl")
                except Exception as excel_error:
                    logger.error(f"Error exporting to Excel: {excel_error}")
            
            # If we have a thermal system, create a simple visualization
            if self.thermal_system and self.last_temperatures:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot current temperatures
                zone_names = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
                zone_temps = [
                    self.last_temperatures.get(name, None) for name in zone_names
                ]
                
                # Filter out None values
                valid_data = [(name, temp) for name, temp in zip(zone_names, zone_temps) if temp is not None]
                if valid_data:
                    names, temps = zip(*valid_data)
                    
                    # Create bar chart
                    bars = ax.bar(names, temps)
                    
                    # Add threshold lines
                    ax.axhline(y=75.0, linestyle='--', color='blue', alpha=0.7, label='Damper Threshold (75°F)')
                    ax.axhline(y=81.0, linestyle='--', color='red', alpha=0.7, label='Critical Temp (81°F)')
                    
                    # Add labels and legend
                    ax.set_xlabel('Zone')
                    ax.set_ylabel('Temperature (°F)')
                    ax.set_title(f'Current Zone Temperatures - {timestamp}')
                    ax.legend()
                    
                    # Save figure
                    plt.tight_layout()
                    fig_path = os.path.join(reports_dir, f"temperatures_{timestamp}.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    logger.info(f"Temperature visualization saved to {fig_path}")
            
            # Create a text report
            report_path = os.path.join(reports_dir, f"report_{timestamp}.txt")
            with open(report_path, "w") as f:
                f.write(f"HVAC System Report - {timestamp}\n")
                f.write("=" * 50 + "\n\n")
                
                # Write current temperatures
                f.write("Current Temperatures:\n")
                if self.last_temperatures:
                    for name, temp in self.last_temperatures.items():
                        if temp is not None:
                            f.write(f"  {name}: {temp:.1f}°F\n")
                f.write("\n")
                
                # Write control actions
                f.write("Current Control Actions:\n")
                if self.last_control_action:
                    f.write(f"  AC: {'ON' if self.last_control_action.get('ac_on') else 'OFF'}\n")
                    f.write("  Dampers:\n")
                    damper_states = self.last_control_action.get('damper_states', [])
                    for i, state in enumerate(damper_states):
                        if i < len(zone_names):
                            f.write(f"    {zone_names[i]}: {'OPEN' if state else 'CLOSED'}\n")
                    
                    # Write control mode
                    mode = self.last_control_action.get('mode', 'MPC')
                    f.write(f"  Control Mode: {mode}\n")
                f.write("\n")
                
                # Write energy data
                f.write("Energy Consumption:\n")
                if self.last_energy_data:
                    for key, value in self.last_energy_data.items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # Write AC verification status
                f.write("AC Control Verification:\n")
                f.write(f"  Last Verified State: {'ON' if self.last_verified_ac_state else 'OFF'}" 
                       f" (or None if not verified)\n")
                f.write(f"  Verification Failures: {self.verification_attempts}\n")
                f.write("\n")
                
                # Add information about Excel data
                f.write("Excel Data Export:\n")
                f.write(f"  Complete time series data has been exported to Excel files in: {excel_dir}\n")
                f.write("  Files exported:\n")
                f.write("    - temperature_data.xlsx (complete temperature time series)\n")
                f.write("    - control_actions_data.xlsx (complete control actions time series)\n")
                f.write("    - energy_consumption_data.xlsx (complete energy data time series)\n")
                f.write("    - hvac_historical_data.xlsx (consolidated file with all historical data)\n")
                f.write(f"    - hvac_data_snapshot_{timestamp}.xlsx (current snapshot)\n")
                f.write("\n")
            
            logger.info(f"Report saved to {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False
    
    async def control_loop(self):
        """Main control loop"""
        logger.info("Starting HVAC control loop")
        
        # Initialize thermal system
        await self.initialize_thermal_system()
        
        while True:
            try:
                # 1. Update thermal system with current temperatures
                system_updated = await self.update_thermal_system()
                
                # 2. Run MPC optimization
                if system_updated:
                    ac_on, damper_states = await self.run_mpc()
                    
                    # 3. Apply control actions
                    if ac_on is not None and damper_states is not None:
                        success, _ = await self.apply_control_actions(ac_on, damper_states)
                        if not success:
                            logger.warning("Failed to apply MPC control actions, using fallback")
                            await self.fallback_control_strategy()
                    else:
                        logger.warning("MPC failed to produce valid control actions, using fallback")
                        await self.fallback_control_strategy()
                else:
                    logger.warning("Failed to update thermal system, using fallback")
                    await self.fallback_control_strategy()
                
                # 4. Generate periodic reports (every hour)
                if datetime.now().minute < 5:  # Generate report in the first 5 minutes of each hour
                    await self.generate_report()
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                # Try fallback strategy in case of errors
                try:
                    await self.fallback_control_strategy()
                except Exception as inner_e:
                    logger.error(f"Fallback strategy also failed: {inner_e}")
            
            # Wait for next control interval
            logger.info(f"Waiting {self.control_interval} seconds until next control cycle")
            await asyncio.sleep(self.control_interval)

async def main():
    """Main entry point"""
    controller = HVACController()
    
    try:
        await controller.control_loop()
    finally:
        # Ensure proper shutdown of hardware
        controller.hardware.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("HVAC Controller stopped by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
