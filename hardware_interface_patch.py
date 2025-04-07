"""
Hardware Interface Patch for Kalman Filtering Integration
"""
import logging
import os
from datetime import datetime
from temperature_kalman_filter import TemperatureKalmanFilterManager

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HardwarePatch")

def patch_hardware_interface(hardware_interface):
    """
    Patch the HardwareInterface to use Kalman filtering for temperature readings.
    
    Args:
        hardware_interface: The HardwareInterface instance to patch
    """
    # Store original method reference
    original_read_temperatures = hardware_interface.read_temperatures
    
    # Variable to track if this is the first reading
    first_reading = True
    
    # Define the patched method
    async def patched_read_temperatures(self):
        """Read temperatures from DAQ hardware with Kalman filtering"""
        nonlocal first_reading
        
        # Call original method to get raw temperatures
        raw_temps, raw_file = await original_read_temperatures()
        
        if not raw_temps:
            logger.error("Failed to get raw temperature readings")
            return None, None
        
        try:
            # If this is the first reading, initialize the filter manager
            # with the actual initial temperatures from DAQs
            if first_reading:
                self.filter_manager = TemperatureKalmanFilterManager(
                    self.data_dir, 
                    initial_temps=raw_temps
                )
                first_reading = False
                logger.info("Initialized Kalman filters with initial DAQ readings")
            # If filter_manager doesn't exist yet (shouldn't happen with the flag)
            elif not hasattr(self, 'filter_manager'):
                self.filter_manager = TemperatureKalmanFilterManager(
                    self.data_dir,
                    initial_temps=raw_temps
                )
                logger.info("Created Kalman filter manager with current readings")
            
            # Get HVAC status for adaptive filtering
            hvac_status = {
                'ac_on': False,
                'damper_states': []
            }
            
            # If we have last_control_action, extract status
            if hasattr(self, 'last_ac_command') and self.last_ac_command:
                hvac_status['ac_on'] = self.last_ac_command == 'on'
            
            if hasattr(self, 'last_damper_command') and self.last_damper_command:
                hvac_status['damper_states'] = self.last_damper_command
            
            # Apply filtering
            filtered_temps = self.filter_manager.filter_temperatures(raw_temps, hvac_status)
            
            # Log the readings
            logger.info(f"Applied Kalman filtering to temperature readings")
            
            # Save temperature data to file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.data_dir, "temperatures", f"filtered_temps_{timestamp}.json")
            
            import json
            with open(filename, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "temperatures": filtered_temps,
                    "raw_temperatures": raw_temps,
                }, f, indent=2)
            
            return filtered_temps, filename
            
        except Exception as e:
            logger.error(f"Error applying Kalman filtering: {e}")
            # Fall back to raw temperature readings
            return raw_temps, raw_file
    
    # Monkey-patch the read_temperatures method
    import types
    hardware_interface.read_temperatures = types.MethodType(patched_read_temperatures, hardware_interface)
    
    logger.info("HardwareInterface patched to use Kalman filtering")
    
    return hardware_interface