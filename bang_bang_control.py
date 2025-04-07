"""
Bang-Bang Control Implementation for HVAC System
For experimental comparison with MPC approach
"""

import os
import sys
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta

# Import the existing hardware interface
from hardware_interface import HardwareInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bang_bang_control.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Bang_Bang_Controller")

class BangBangController:
    """Simple bang-bang controller for HVAC system"""
    
    def __init__(self, data_dir="data/bang_bang"):
        """Initialize bang-bang controller"""
        self.data_dir = data_dir
        self.control_interval = 300  # 5 minutes - same as MPC
        self.hardware = HardwareInterface(data_dir)
        
        # Bang-bang control parameters
        self.upper_threshold = 80.0  # Turn AC ON if any zone exceeds this temperature (°F)
        self.lower_threshold = 75.0  # Turn AC OFF if all zones below this temperature (°F)
        self.hysteresis = 2.0  # Hysteresis to prevent rapid cycling (°F)
        
        # Current state tracking
        self.last_temperatures = None
        self.ac_status = False  # Current AC state (True=ON, False=OFF)
        self.last_control_action = None
        self.last_energy_data = None
        
        # Analysis metrics
        self.comfort_violations = 0
        self.control_actions = 0  # Count of AC state changes
        self.energy_usage = 0.0  # Estimated total energy usage
        
        # Create data directories if they don't exist
        os.makedirs(os.path.join(data_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "control_actions"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "temperatures"), exist_ok=True)
        
        logger.info("Bang-Bang Controller initialized")
    
    async def read_temperatures(self):
        """Read temperatures from hardware"""
        temp_readings, _ = await self.hardware.read_temperatures()
        if not temp_readings:
            logger.error("Failed to get temperature readings")
            return False
        
        # Store the temperature readings
        self.last_temperatures = temp_readings
        
        # Count comfort violations
        for zone, temp in temp_readings.items():
            if zone != "Ambient" and temp is not None:
                if temp > self.upper_threshold + self.hysteresis:
                    self.comfort_violations += 1
                    logger.warning(f"Comfort violation in {zone}: {temp}°F > {self.upper_threshold}°F")
        
        return True
    
    async def determine_control_action(self):
        """Determine control action using bang-bang logic"""
        if not self.last_temperatures:
            logger.error("No temperature data available")
            return None, None
        
        # Extract zone temperatures (excluding ambient)
        zone_temps = {k: v for k, v in self.last_temperatures.items() if k != "Ambient" and v is not None}
        
        if not zone_temps:
            logger.error("No valid zone temperatures available")
            return None, None
        
        # Get max temperature across all zones
        max_temp = max(zone_temps.values())
        # Determine if any zones are too hot
        any_too_hot = any(temp > self.upper_threshold for temp in zone_temps.values())
        # Determine if all zones are cool enough
        all_cool_enough = all(temp < self.lower_threshold for temp in zone_temps.values())
        
        # Bang-bang control logic with hysteresis
        new_ac_status = self.ac_status  # Start with current state
        
        # If AC is OFF and any zone is too hot, turn AC ON
        if not self.ac_status and any_too_hot:
            new_ac_status = True
            logger.info(f"Bang-bang control: Turning AC ON (max temp: {max_temp}°F > {self.upper_threshold}°F)")
        
        # If AC is ON and all zones are cool enough, turn AC OFF
        elif self.ac_status and all_cool_enough:
            new_ac_status = False
            logger.info(f"Bang-bang control: Turning AC OFF (all zones < {self.lower_threshold}°F)")
        
        # Create damper states - in bang-bang control, all dampers are open when AC is ON
        # More sophisticated version: only open dampers for zones above the threshold
        damper_states = []
        for zone in ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]:
            temp = self.last_temperatures.get(zone)
            if temp is not None:
                # Open damper if temperature is above the open threshold
                damper_states.append(1 if temp > self.lower_threshold else 0)
            else:
                damper_states.append(0)  # Default to closed if no temperature reading
        
        # Log damper states
        logger.info(f"Damper states: {damper_states}")
        
        # Count control action if AC state changed
        if new_ac_status != self.ac_status:
            self.control_actions += 1
        
        # Update AC status
        self.ac_status = new_ac_status
        
        return new_ac_status, damper_states
    
    async def apply_control_actions(self, ac_on, damper_states):
        """Apply control actions to hardware"""
        if ac_on is None or damper_states is None:
            logger.warning("No valid control actions to apply")
            return False, None
        
        # Apply control actions to hardware
        success, energy_data, verification = await self.hardware.apply_control_actions(ac_on, damper_states)
        
        # Store control actions and energy data
        self.last_control_action = {
            "timestamp": datetime.now().isoformat(),
            "ac_on": ac_on,
            "damper_states": damper_states,
            "success": success
        }
        
        self.last_energy_data = energy_data
        
        # Update energy usage
        if energy_data and "power" in energy_data:
            power = energy_data.get("power", 0)
            # Convert W to kWh for this interval
            energy_used = power * (self.control_interval / 3600) / 1000
            self.energy_usage += energy_used
            logger.info(f"Energy used in this interval: {energy_used:.4f} kWh, Total: {self.energy_usage:.4f} kWh")
        
        return success, energy_data
    
    async def generate_report(self, experiment_duration):
        """Generate performance report"""
        logger.info("Generating performance report")
        
        try:
            # Create report directory
            reports_dir = os.path.join(self.data_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Create a text report
            report_path = os.path.join(reports_dir, f"bang_bang_report_{timestamp}.txt")
            with open(report_path, "w") as f:
                f.write(f"Bang-Bang Control Experiment Report - {timestamp}\n")
                f.write("=" * 60 + "\n\n")
                
                # Experiment parameters
                f.write("Experiment Parameters:\n")
                f.write(f"  Duration: {experiment_duration:.2f} hours\n")
                f.write(f"  Control Interval: {self.control_interval} seconds\n")
                f.write(f"  Upper Threshold: {self.upper_threshold}°F\n")
                f.write(f"  Lower Threshold: {self.lower_threshold}°F\n")
                f.write(f"  Hysteresis: {self.hysteresis}°F\n\n")
                
                # Performance metrics
                f.write("Performance Metrics:\n")
                f.write(f"  Control Actions: {self.control_actions}\n")
                f.write(f"  Comfort Violations: {self.comfort_violations}\n")
                f.write(f"  Total Energy Usage: {self.energy_usage:.4f} kWh\n")
                
                if experiment_duration > 0:
                    f.write(f"  Control Frequency: {self.control_actions / experiment_duration:.2f} actions/hour\n")
                    f.write(f"  Comfort Violation Rate: {self.comfort_violations / experiment_duration:.2f} violations/hour\n")
                    f.write(f"  Average Power: {(self.energy_usage / experiment_duration) * 1000:.2f} W\n\n")
                
                # Current state
                f.write("Current State:\n")
                f.write(f"  AC Status: {'ON' if self.ac_status else 'OFF'}\n")
                
                if self.last_temperatures:
                    f.write("  Zone Temperatures:\n")
                    for zone, temp in self.last_temperatures.items():
                        if temp is not None:
                            f.write(f"    {zone}: {temp:.1f}°F\n")
                
                if self.last_energy_data:
                    f.write("\nEnergy Data:\n")
                    for key, value in self.last_energy_data.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {key}: {value}\n")
            
            logger.info(f"Report saved to {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    async def run_experiment(self, duration_hours=2):
        """Run a timed experiment with bang-bang control"""
        logger.info(f"Starting bang-bang control experiment for {duration_hours} hours")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Store experiment parameters
        experiment_info = {
            "start_time": start_time.isoformat(),
            "planned_end_time": end_time.isoformat(),
            "duration_hours": duration_hours,
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "hysteresis": self.hysteresis
        }
        
        # Save experiment parameters
        experiment_file = os.path.join(self.data_dir, "experiment_info.json")
        with open(experiment_file, "w") as f:
            json.dump(experiment_info, f, indent=2)
        
        iterations = 0
        
        try:
            while datetime.now() < end_time:
                iterations += 1
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds() / 3600  # hours
                remaining = (end_time - current_time).total_seconds() / 60  # minutes
                
                logger.info(f"Iteration {iterations} - Elapsed: {elapsed:.2f}h, Remaining: {remaining:.1f}min")
                
                try:
                    # 1. Read temperatures
                    await self.read_temperatures()
                    
                    # 2. Determine control action
                    ac_on, damper_states = await self.determine_control_action()
                    
                    # 3. Apply control actions
                    if ac_on is not None and damper_states is not None:
                        success, _ = await self.apply_control_actions(ac_on, damper_states)
                        if not success:
                            logger.warning("Failed to apply control actions")
                    
                    # 4. Generate interim report every hour
                    if iterations % 12 == 0:  # Every hour (12 * 5min intervals)
                        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                        await self.generate_report(elapsed_hours)
                    
                except Exception as e:
                    logger.error(f"Error in control loop: {e}")
                
                # Wait for next control interval
                logger.info(f"Waiting {self.control_interval} seconds until next control cycle")
                await asyncio.sleep(self.control_interval)
        
        except KeyboardInterrupt:
            logger.info("Experiment interrupted by user")
        
        finally:
            # Calculate actual experiment duration
            actual_duration = (datetime.now() - start_time).total_seconds() / 3600
            
            # Generate final report
            final_report = await self.generate_report(actual_duration)
            
            # Update experiment information
            experiment_info["actual_end_time"] = datetime.now().isoformat()
            experiment_info["actual_duration_hours"] = actual_duration
            experiment_info["iterations"] = iterations
            experiment_info["control_actions"] = self.control_actions
            experiment_info["comfort_violations"] = self.comfort_violations
            experiment_info["energy_usage_kwh"] = self.energy_usage
            experiment_info["final_report"] = final_report
            
            # Save updated experiment information
            with open(experiment_file, "w") as f:
                json.dump(experiment_info, f, indent=2)
            
            logger.info(f"Experiment completed - Duration: {actual_duration:.2f} hours")
            logger.info(f"Control actions: {self.control_actions}, Comfort violations: {self.comfort_violations}")
            logger.info(f"Energy usage: {self.energy_usage:.4f} kWh")
            
            # Ensure proper shutdown of hardware
            self.hardware.shutdown()

async def main():
    """Main entry point"""
    controller = BangBangController()
    
    # Default duration 2 hours
    duration = 2.0
    
    # Allow specifying duration from command line
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            print(f"Invalid duration: {sys.argv[1]}. Using default: {duration} hours")
    
    try:
        await controller.run_experiment(duration)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
    finally:
        # Ensure proper shutdown even on error
        if hasattr(controller, 'hardware'):
            controller.hardware.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bang-Bang Controller stopped by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
