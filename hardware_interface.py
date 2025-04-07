"""
Hardware interface module with corrected AC power verification threshold
"""

import os
import sys
import json
import asyncio
import logging
import time
from datetime import datetime

# Import hardware interface modules
import DAQs  # Temperature sensor module
import RM4mini  # AC control module
import dampers  # Damper control module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HVAC_Hardware")

class HardwareInterface:
    """Enhanced interface to HVAC hardware components with correct damper mapping"""
    
    def __init__(self, data_dir="data"):
        """Initialize hardware interface"""
        self.data_dir = data_dir
        self.last_ac_command = None
        self.last_damper_command = None
        
        # Set proper power threshold for AC verification with correct interpretation of power values
        self.ac_power_threshold = 100.0  # Watts - threshold to consider AC in active cooling (between 120W standby and 800W active)
        self.ac_standby_threshold = 5.0   # Watts - threshold to detect if AC has any power (above ~0.8W off state)
        
        # Define the correct zone-to-damper mapping
        # Each key is a zone name, value is the corresponding damper index (0-7)
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
        
        # Create data directories if they don't exist
        os.makedirs(os.path.join(data_dir, "temperatures"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "control_actions"), exist_ok=True)
        
        # Initialize hardware components
        self._init_hardware()
        
        logger.info("Hardware interface initialized with corrected damper mapping")
    
    def _init_hardware(self):
        """Initialize hardware components"""
        try:
            # Initialize damper control
            dampers.initialize_serial()
            logger.info("Damper control initialized")
        except Exception as e:
            logger.error(f"Failed to initialize damper control: {e}")
    
    async def read_temperatures(self):
        """Read temperatures from DAQ hardware
        
        Returns:
            dict: Dictionary with temperature readings for all zones and ambient
        """
        logger.info("Reading temperatures from sensors")
        
        try:
            temperatures = {}
            # Read temperatures from all configured DAQ boards
            for serial_number, board_num in DAQs.daq_boards.items():
                for channel, labels in DAQs.daq_mapping[serial_number].items():
                    temp = DAQs.read_temperature(board_num, channel)
                    if isinstance(temp, (int, float)):  # Only store valid readings
                        for label in labels:
                            temperatures[label] = temp
            
            # Extract the specific zone temperatures needed for MPC
            zone_temps = {
                "TR1": temperatures.get("TR1  (C4H)", None),
                "TR2": temperatures.get("TR2  (C6H)", None),
                "TR3": temperatures.get("TR3 (C1H)", None),
                "TR4": temperatures.get("TR4 (C2H)", None),
                "BR1": temperatures.get("BR1 (C4H)", None),
                "BR2": temperatures.get("BR2 (C7H)", None),
                "BR3": temperatures.get("BR3 (C0H)", None),
                "BR4": temperatures.get("BR4 (C2H)", None),
                "Ambient": temperatures.get("AMBIENT (C6H)", None)
            }
            
            # Log the readings
            logger.info(f"Zone temperatures: {zone_temps}")
            
            # Save temperature data to file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.data_dir, "temperatures", f"temps_{timestamp}.json")
            
            with open(filename, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "temperatures": zone_temps,
                    "raw_readings": temperatures
                }, f, indent=2)
            
            return zone_temps, filename
        
        except Exception as e:
            logger.error(f"Error reading temperatures: {e}")
            return None, None
    
    async def control_ac(self, on):
        """Control AC unit via RM4mini based on MPC decision
        
        Args:
            on (bool): True to turn on AC, False to turn off
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            action = "on" if on else "off"
            logger.info(f"Sending {action.upper()} command to AC unit")
            
            # Find the RM4mini device
            device = RM4mini.get_device_by_name()
            if not device:
                logger.error("Failed to find RM4mini device")
                return False
            
            # Send toggle command
            success = RM4mini.send_toggle_command(device)
            
            # Log the action
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.data_dir, "control_actions", f"ac_{timestamp}.json")
            
            with open(filename, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "action": action,
                    "success": success
                }, f, indent=2)
            
            # Update last command
            self.last_ac_command = action
            
            return success
                
        except Exception as e:
            logger.error(f"Error controlling AC: {e}")
            return False
    
    async def verify_ac_power(self, expected_on=True, max_attempts=3, delay=2):
        """
        Verify AC power state by checking energy consumption with updated thresholds
        
        Args:
            expected_on (bool): Expected state (True=on, False=off)
            max_attempts (int): Maximum verification attempts
            delay (int): Seconds between attempts
            
        Returns:
            dict: Verification result with status and power data
        """
        logger.info(f"Verifying AC power state (expected: {'ON' if expected_on else 'OFF'})")
        
        for attempt in range(max_attempts):
            try:
                # Import the energy monitor module
                from energy_monitor import get_energy_data
                
                # Get energy data
                energy_data = await get_energy_data()
                
                if not energy_data or "error" in energy_data:
                    logger.warning(f"Failed to get energy data on attempt {attempt+1}/{max_attempts}")
                    await asyncio.sleep(delay)
                    continue
                
                # Get power value
                power = energy_data.get("power", 0)
                
                # Check if power matches expected state using updated thresholds
                # - For "ON" verification: Power must be above active cooling threshold (200W)
                # - For "OFF" verification: Power must be below standby threshold (5W)
                if expected_on:
                    power_on = power > self.ac_power_threshold
                    power_state = "ACTIVE" if power_on else "STANDBY" if power > self.ac_standby_threshold else "OFF"
                else:
                    # For OFF verification, we only care if it's in active cooling mode
                    power_on = power > self.ac_power_threshold
                    power_state = "ACTIVE" if power_on else "STANDBY/OFF"
                
                state_matches = power_on == expected_on
                
                logger.info(f"Power reading: {power:.2f}W, Power state: {power_state}, Expected: {'ACTIVE' if expected_on else 'STANDBY/OFF'}")
                
                if state_matches:
                    logger.info(f"AC power state verified: {power_state}")
                    return {
                        "verified": True,
                        "power": power,
                        "power_state": power_state,
                        "is_active": power_on,
                        "expected_on": expected_on
                    }
                else:
                    logger.warning(f"AC power state mismatch: Expected {'ACTIVE' if expected_on else 'STANDBY/OFF'}, "
                                 f"Got {power_state} ({power:.2f}W)")
            
            except Exception as e:
                logger.error(f"Error verifying AC power on attempt {attempt+1}/{max_attempts}: {e}")
            
            # Wait before trying again
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
        
        # If we get here, verification failed
        logger.error(f"AC power verification failed after {max_attempts} attempts")
        return {
            "verified": False,
            "expected_on": expected_on
        }
    
    def _map_zone_states_to_dampers(self, zone_states):
        """
        Map zone-ordered states to damper-ordered states using the zone_to_damper mapping
        
        Args:
            zone_states: List of states ordered by zone [TR1, TR2, TR3, TR4, BR1, BR2, BR3, BR4]
            
        Returns:
            List of states ordered by damper number [1, 2, 3, 4, 5, 6, 7, 8]
        """
        # Create a list of 8 damper states, all initialized to 0 (closed)
        damper_states = [0] * 8
        
        # Map each zone state to its corresponding damper position
        zone_names = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
        
        for i, zone in enumerate(zone_names):
            if i < len(zone_states):
                damper_idx = self.zone_to_damper[zone]
                damper_states[damper_idx] = zone_states[i]
        
        return damper_states
    
    async def control_dampers(self, zone_states):
        """Control dampers via Arduino with zone-to-damper mapping
        
        Args:
            zone_states (list): List of binary states ordered by zones [TR1, TR2, TR3, TR4, BR1, BR2, BR3, BR8]
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Map zone states to damper states using the correct mapping
            damper_states = self._map_zone_states_to_dampers(zone_states)
            
            # Log the mapping for debugging
            zone_names = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
            logger.info("Zone to damper mapping:")
            for i, zone in enumerate(zone_names):
                if i < len(zone_states):
                    damper_idx = self.zone_to_damper[zone]
                    logger.info(f"  {zone} (state: {zone_states[i]}) -> Damper {damper_idx+1} (state: {damper_states[damper_idx]})")
            
            # Check if this is a redundant command
            if self.last_damper_command == damper_states:
                logger.info("Dampers already in requested positions, skipping command")
                return True
            
            # Convert binary states to servo angles (0 or 90)
            damper_positions = dampers.convert_binary_to_positions(damper_states)
            
            logger.info(f"Setting damper positions to: {damper_positions}")
            
            # Call dampers module to control servos
            success = dampers.send_servo_positions(damper_positions)
            
            if not success:
                logger.warning("Failed to set damper positions, retrying...")
                # Brief pause before retry
                await asyncio.sleep(1)
                success = dampers.send_servo_positions(damper_positions)
                
                if not success:
                    logger.error("Damper control failed after retry")
                    return False
            
            # Log the action
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.data_dir, "control_actions", f"dampers_{timestamp}.json")
            
            with open(filename, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "zone_states": zone_states,
                    "damper_states": damper_states,
                    "damper_positions": damper_positions,
                    "success": True
                }, f, indent=2)
            
            # Update last command
            self.last_damper_command = damper_states.copy()
            
            return True
            
        except Exception as e:
            logger.error(f"Error controlling dampers: {e}")
            return False
    
    async def monitor_energy(self):
        """Monitor energy consumption via KP115 smart plug
        
        Returns:
            dict: Energy data if successful, None otherwise
        """
        try:
            # Import energy_monitor module
            from energy_monitor import get_energy_data
            
            logger.info("Monitoring energy consumption")
            energy_data = await get_energy_data()
            
            if energy_data and "error" not in energy_data:
                # Log the energy data
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(self.data_dir, "control_actions", f"energy_{timestamp}.json")
                
                with open(filename, "w") as f:
                    json.dump({
                        "timestamp": timestamp,
                        "energy_data": energy_data
                    }, f, indent=2)
                
                return energy_data
            else:
                logger.warning(f"Energy monitoring returned error: {energy_data.get('error', 'Unknown error')}")
                return None
            
        except Exception as e:
            logger.error(f"Error monitoring energy: {e}")
            return None
    
    async def apply_control_actions(self, ac_on, zone_states):
        """Apply control actions to hardware with continuous verification
        
        Args:
            ac_on (bool): AC on/off state from MPC
            zone_states (list): Damper states ordered by zones [TR1, TR2, TR3, TR4, BR1, BR2, BR3, BR8]
        
        Returns:
            tuple: (success, energy_data, verification_result)
        """
        # Control dampers first (to prepare for AC)
        dampers_success = await self.control_dampers(zone_states)
        
        # Brief pause between commands
        await asyncio.sleep(1)
        
        # First check current state before taking action
        from energy_monitor import get_energy_data
        
        logger.info(f"MPC decision: AC {'ON' if ac_on else 'OFF'}")
        logger.info("Measuring current AC state before taking action")
        
        # Get current energy data
        energy_data = await get_energy_data()
        if energy_data and "error" not in energy_data:
            power = energy_data.get("power", 0)
            current_is_on = power > self.ac_power_threshold
            logger.info(f"Current AC state: {'ON' if current_is_on else 'OFF/STANDBY'} ({power:.2f}W)")
            
            # Only take action if current state doesn't match desired state
            if current_is_on != ac_on:
                logger.info(f"Changing AC state to match MPC decision: {'ON' if ac_on else 'OFF'}")
                # Control AC
                ac_success = await self.control_ac(ac_on)
                
                # Verify state change with multiple checks over 30 seconds
                verification_success = False
                verification_result = None
                
                logger.info("Beginning 30-second verification period...")
                start_time = time.time()
                check_interval = 5  # Check every 5 seconds
                
                while time.time() - start_time < 30:
                    # Wait for a bit before checking
                    await asyncio.sleep(check_interval)
                    
                    # Check current state
                    energy_data = await get_energy_data()
                    if energy_data and "error" not in energy_data:
                        power = energy_data.get("power", 0)
                        actual_is_on = power > self.ac_power_threshold
                        seconds_elapsed = time.time() - start_time
                        
                        logger.info(f"Verification check at {seconds_elapsed:.1f}s: " 
                                   f"State {'ON' if actual_is_on else 'OFF/STANDBY'} ({power:.2f}W)")
                        
                        if actual_is_on == ac_on:
                            verification_success = True
                            verification_result = {
                                "verified": True,
                                "power": power,
                                "power_state": "ACTIVE" if actual_is_on else "STANDBY/OFF",
                                "is_active": actual_is_on,
                                "expected_on": ac_on
                            }
                            logger.info(f"AC state verified as {'ON' if ac_on else 'OFF'}")
                            break
                
                if not verification_success:
                    logger.error(f"AC state verification failed after 30 seconds")
                    verification_result = {
                        "verified": False,
                        "expected_on": ac_on
                    }
                    
                    # Try one more toggle if verification failed
                    logger.info("Trying one more toggle command")
                    device = RM4mini.get_device_by_name()
                    if device:
                        RM4mini.send_toggle_command(device)
            else:
                logger.info(f"AC is already in state requested by MPC: {'ON' if ac_on else 'OFF'}")
                ac_success = True
                verification_result = {
                    "verified": True,
                    "power": power,
                    "power_state": "ACTIVE" if current_is_on else "STANDBY/OFF",
                    "is_active": current_is_on,
                    "expected_on": ac_on
                }
        else:
            # Fallback if we can't get energy data
            logger.warning("Could not verify current AC state, proceeding with MPC command anyway")
            ac_success = await self.control_ac(ac_on)
            verification_result = None
        
        # Final energy data monitoring
        energy_data = await self.monitor_energy()
        
        # Return overall results
        return (dampers_success and (ac_success if 'ac_success' in locals() else False)), energy_data, verification_result
    
    # Add a simpler version for fallback control that returns only 2 values
    async def apply_fallback_actions(self, ac_on, zone_states):
        """Simplified version of apply_control_actions for fallback control
        
        Args:
            ac_on (bool): AC on/off state
            zone_states (list): Damper states ordered by zones [TR1, TR2, TR3, TR4, BR1, BR2, BR3, BR8]
        
        Returns:
            tuple: (success, energy_data)
        """
        success, energy_data, _ = await self.apply_control_actions(ac_on, zone_states)
        return success, energy_data
    
    def shutdown(self):
        """Shutdown hardware interface and release resources"""
        try:
            # Close damper serial connection
            dampers.close_serial()
            logger.info("Hardware interface shutdown complete")
        except Exception as e:
            logger.error(f"Error during hardware shutdown: {e}")