import asyncio
from kasa import SmartPlug  # Updated import
import logging
import json
import os
from datetime import datetime
import socket
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("energy_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Energy_Monitor")

# Smart plug IP address - always check hotspot for this info tho
PLUG_IP = "192.168.1.3"  # actual static
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def ping_device(ip, timeout=1):
    """Check if device is reachable with ping"""
    try:
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Try to connect to the device
        result = sock.connect_ex((ip, 9999))  # KP115 uses port 9999
        sock.close()
        
        return result == 0
    
    except Exception as e:
        logger.warning(f"Network error pinging device: {e}")
        return False

async def get_energy_data(retry_count=0):
    """
    Reads energy consumption data from the KP115 Smart Plug with retry logic.
    
    Args:
        retry_count: Current retry attempt
        
    Returns:
        dict: Energy data including power, voltage, current and total consumption
    """
    # Check if device is reachable first
    if not ping_device(PLUG_IP):
        if retry_count < MAX_RETRIES:
            logger.warning(f"Device at {PLUG_IP} not reachable, retrying in {RETRY_DELAY}s (attempt {retry_count+1}/{MAX_RETRIES})")
            await asyncio.sleep(RETRY_DELAY)
            return await get_energy_data(retry_count + 1)
        else:
            logger.error(f"Device at {PLUG_IP} not reachable after {MAX_RETRIES} attempts")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": f"Device at {PLUG_IP} not reachable"
            }
    
    # Initialize the plug using the updated API
    plug = SmartPlug(PLUG_IP)
    
    try:
        # Connect to the plug with timeout
        try:
            await asyncio.wait_for(plug.update(), timeout=5.0)
        except asyncio.TimeoutError:
            if retry_count < MAX_RETRIES:
                logger.warning(f"Connection timeout, retrying in {RETRY_DELAY}s (attempt {retry_count+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                return await get_energy_data(retry_count + 1)
            else:
                raise TimeoutError(f"Connection timeout after {MAX_RETRIES} attempts")
        
        # Fetch energy data using the updated API
        if hasattr(plug, "emeter_realtime"):
            # For older versions of the kasa library
            emeter = plug.emeter_realtime
            logger.debug("Using legacy emeter_realtime property")
        elif hasattr(plug, "modules") and "Energy" in plug.modules:
            # For newer versions of the kasa library
            emeter = await plug.modules["Energy"].get_realtime_stats()
            logger.debug("Using new modules['Energy'] API")
        else:
            # Fallback to try accessing the Energy module directly
            try:
                from kasa.modules import Energy
                energy_module = Energy(plug)
                emeter = await energy_module.get_realtime()
                logger.debug("Using direct Energy module access")
            except Exception as e:
                logger.error(f"Cannot access energy data using any method: {e}")
                emeter = {}
        
        status = "ON" if plug.is_on else "OFF"
        
        # Log all available fields for debugging
        logger.debug(f"All emeter fields: {emeter}")
        
        # Handle different ways to access emeter values
        if hasattr(emeter, 'power'):
            # Access as an object attribute
            power = emeter.power
            voltage = emeter.voltage
            current = emeter.current
            total = getattr(emeter, 'total_wh', getattr(emeter, 'total', 0))
        else:
            # Access as a dictionary
            power = emeter.get('power', 0)
            voltage = emeter.get('voltage', 0)
            current = emeter.get('current', 0)
            total = emeter.get('total_wh', emeter.get('total', 0))
        
        logger.info(f"AC is {status}, Power: {power}W, Voltage: {voltage}V, Current: {current}A")
        
        # If we get valid readings with zeros across the board, it could indicate a problem
        if power == 0 and current == 0 and plug.is_on:
            logger.warning(f"Suspicious readings: AC is reported ON but power and current are 0")
            logger.warning(f"This could indicate: 1) The AC is in standby mode, 2) The KP115 is not correctly measuring power, or 3) The AC isn't actually plugged into the KP115")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "ac_status": status.lower(),
            "power": power,
            "voltage": voltage,
            "current": current,
            "total_energy": total,
            "raw_emeter_data": str(emeter)  # Convert to string for JSON serialization
        }
            
    except Exception as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Error reading energy data: {e}, retrying in {RETRY_DELAY}s (attempt {retry_count+1}/{MAX_RETRIES})")
            await asyncio.sleep(RETRY_DELAY)
            return await get_energy_data(retry_count + 1)
        else:
            logger.error(f"Error reading energy data after {MAX_RETRIES} attempts: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

async def toggle_plug(turn_on=True):
    """
    Toggle the smart plug on or off for testing purposes
    
    Args:
        turn_on: True to turn on, False to turn off
    """
    try:
        plug = SmartPlug(PLUG_IP)
        await plug.update()
        current_state = "ON" if plug.is_on else "OFF"
        
        if turn_on and not plug.is_on:
            logger.info(f"Turning plug ON (current state: {current_state})")
            await plug.turn_on()
        elif not turn_on and plug.is_on:
            logger.info(f"Turning plug OFF (current state: {current_state})")
            await plug.turn_off()
        else:
            logger.info(f"Plug is already {'ON' if turn_on else 'OFF'}")
        
        # Verify the state change
        await asyncio.sleep(1)
        await plug.update()
        new_state = "ON" if plug.is_on else "OFF"
        logger.info(f"Plug is now {new_state}")
        
        return new_state == ("ON" if turn_on else "OFF")
    
    except Exception as e:
        logger.error(f"Error toggling plug: {e}")
        return False

async def verify_connection_to_ac():
    """
    Test sequence to verify if the plug is actually connected to the AC unit
    
    This cycles the plug and prompts the user to verify if the AC responds
    """
    print("\n==== VERIFYING KP115 CONNECTION TO AC UNIT ====")
    print("This test will cycle the smart plug to check if it's controlling the AC.")
    print("IMPORTANT: Make sure it's safe to temporarily turn off your AC.")
    proceed = input("Continue with this test? (y/n): ").lower()
    
    if proceed != 'y':
        print("Test cancelled.")
        return
    
    try:
        # First make sure the plug is ON
        plug = SmartPlug(PLUG_IP)
        await plug.update()
        
        if not plug.is_on:
            print("Turning plug ON first...")
            await plug.turn_on()
            await asyncio.sleep(2)  # Give it time to turn on
        
        # Now get a power reading
        print("Getting current power reading...")
        energy_data = await get_energy_data()
        current_power = energy_data.get('power', 0)
        print(f"Current power consumption: {current_power}W")
        
        # Now turn OFF the plug
        print("\nTurning the plug OFF for 5 seconds...")
        await plug.turn_off()
        
        # Wait and ask user to verify
        for i in range(5, 0, -1):
            print(f"Waiting... {i} seconds", end="\r")
            await asyncio.sleep(1)
        print("\n")
        
        ac_turned_off = input("Did the AC unit turn OFF? (y/n/not sure): ").lower()
        
        # Turn the plug back ON
        print("\nTurning the plug back ON...")
        await plug.turn_on()
        await asyncio.sleep(3)  # Give it time to turn back on
        
        # Get another power reading
        print("Getting updated power reading...")
        energy_data = await get_energy_data()
        new_power = energy_data.get('power', 0)
        print(f"New power consumption: {new_power}W")
        
        ac_turned_on = input("Did the AC unit turn back ON? (y/n/not sure): ").lower()
        
        # Analyze results
        print("\n==== TEST RESULTS ====")
        if ac_turned_off == 'y' and ac_turned_on == 'y':
            print("PASS: The KP115 appears to be correctly controlling the AC unit.")
        elif ac_turned_off == 'y' or ac_turned_on == 'y':
            print("PARTIAL PASS: The KP115 appears to have some control over the AC unit.")
        else:
            print("FAIL: The KP115 does not appear to be controlling the AC unit.")
            print("Possible issues:")
            print("1. The AC unit may be plugged into a different outlet")
            print("2. The AC unit may have battery backup or separate power")
            print("3. The KP115 may not be working properly")
        
        print(f"\nPower readings: Before={current_power}W, After={new_power}W")
        power_changed = abs(current_power - new_power) > 0.05  # More than 0.05W difference (adjusted for low values)
        
        if power_changed:
            print("Power consumption changed during the test.")
        else:
            print("Power consumption did NOT change significantly during the test.")
            if current_power < 0.1 and new_power < 0.1:
                print("Both readings are very low, suggesting the AC might not be drawing power through this plug.")
        
        return ac_turned_off == 'y' and ac_turned_on == 'y'
        
    except Exception as e:
        print(f"Error during connection test: {e}")
        return False

async def continuous_monitoring(interval=60, data_dir="data/energy"):
    """
    Continuously monitor and log energy data at specified intervals
    
    Args:
        interval: Seconds between readings
        data_dir: Directory to store energy logs
    """
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f"Starting continuous energy monitoring (interval: {interval}s)")
    logger.info(f"Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Get energy data
            data = await get_energy_data()
            
            # Save to a daily energy log file
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(data_dir, f"energy_log_{date_str}.jsonl")
            
            with open(log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
            
            # Also save latest reading to a separate file for easy access
            latest_file = os.path.join(data_dir, "latest_reading.json")
            with open(latest_file, "w") as f:
                json.dump(data, f, indent=2)
            
            # Wait for next interval
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Continuous monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in continuous monitoring: {e}")

async def discover_plugs():
    """
    Discover TP-Link smart plugs on the network
    """
    from kasa import Discover
    
    logger.info("Searching for TP-Link devices on the network...")
    found_devices = await Discover.discover()
    
    if not found_devices:
        logger.warning("No TP-Link devices found on the network.")
        return {}
    
    logger.info(f"Found {len(found_devices)} TP-Link devices:")
    for ip, device in found_devices.items():
        await device.update()
        logger.info(f"- {device.alias} ({device.model}) at {ip}")
    
    return found_devices

async def diagnostics_mode():
    """
    Run comprehensive diagnostics on the energy monitoring setup
    """
    print("\n==== ENERGY MONITORING DIAGNOSTICS ====")
    
    # 1. Network connectivity check
    print("\n1. Checking network connectivity to KP115...")
    if ping_device(PLUG_IP):
        print(f"✅ KP115 is reachable at {PLUG_IP}")
    else:
        print(f"❌ KP115 is NOT reachable at {PLUG_IP}")
        print(f"   Check if the device is powered on and connected to your network.")
        return
    
    # 2. Device discovery
    print("\n2. Running device discovery...")
    devices = await discover_plugs()
    if PLUG_IP in devices:
        print(f"✅ KP115 was found in device discovery at {PLUG_IP}")
    else:
        print(f"⚠️ KP115 at {PLUG_IP} was not found during discovery")
        print(f"   Device may be online but not responding to discovery packets")
    
    # 3. Connection test
    print("\n3. Testing direct connection to KP115...")
    try:
        plug = SmartPlug(PLUG_IP)
        await plug.update()
        print(f"✅ Connected to {plug.alias} ({plug.model}) successfully")
        print(f"   Hardware: {plug.hw_info}")
        print(f"   Software: {plug.sw_info}")
        print(f"   Current state: {'ON' if plug.is_on else 'OFF'}")
    except Exception as e:
        print(f"❌ Failed to connect to KP115: {e}")
        return
    
    # 4. Energy data test
    print("\n4. Testing energy data retrieval...")
    energy_data = await get_energy_data()
    if "error" in energy_data:
        print(f"❌ Failed to retrieve energy data: {energy_data['error']}")
    else:
        print(f"✅ Energy data retrieved successfully:")
        print(f"   Status: {energy_data['ac_status'].upper()}")
        print(f"   Power: {energy_data['power']}W")
        print(f"   Voltage: {energy_data['voltage']}V")
        print(f"   Current: {energy_data['current']}A")
        
        if energy_data['power'] < 0.1 and energy_data['current'] < 0.01:
            print(f"\n⚠️ WARNING: Power and current readings are very low")
            print(f"   This could indicate the AC is in standby mode or not connected to this plug")
    
    # 5. Device capabilities
    print("\n5. Checking device capabilities...")
    if hasattr(plug, "modules"):
        print(f"✅ Device supports the modules API")
        print(f"   Available modules: {', '.join(plug.modules.keys())}")
        
        if "Energy" in plug.modules:
            print(f"✅ Energy module is available")
        else:
            print(f"❌ Energy module is NOT available")
    else:
        print(f"⚠️ Device does not support the modules API")
        
    # 6. Control test
    print("\n6. Testing device control...")
    control_test = input("Would you like to test turning the plug ON and OFF? (y/n): ").lower()
    if control_test == 'y':
        original_state = plug.is_on
        
        # Toggle to the opposite state
        if original_state:
            print(f"Turning plug OFF...")
            await plug.turn_off()
        else:
            print(f"Turning plug ON...")
            await plug.turn_on()
        
        await asyncio.sleep(2)
        await plug.update()
        
        if plug.is_on != original_state:
            print(f"✅ Successfully changed plug state to {'ON' if plug.is_on else 'OFF'}")
        else:
            print(f"❌ Failed to change plug state")
        
        # Return to original state
        print(f"Returning plug to original state...")
        if original_state:
            await plug.turn_on()
        else:
            await plug.turn_off()
    
    # 7. AC connection verification
    print("\n7. Verifying KP115 connection to AC...")
    verify_test = input("Would you like to verify if this plug is controlling your AC? (y/n): ").lower()
    if verify_test == 'y':
        await verify_connection_to_ac()
    
    print("\n==== DIAGNOSTICS COMPLETE ====")
    print("If you're still experiencing issues, please check the logs and contact support.")

if __name__ == "__main__":
    async def main():
        # Set a higher debug level temporarily
        logger.setLevel(logging.DEBUG)
        
        # Show menu
        print("\nTP-Link KP115 Energy Monitor")
        print("=" * 30)
        print("1. Get a single energy reading")
        print("2. Start continuous monitoring")
        print("3. Run diagnostics")
        print("4. Verify connection to AC")
        print("5. Discover TP-Link devices")
        choice = input("Enter choice (1-5): ")
        
        if choice == "1":
            data = await get_energy_data()
            print("\nEnergy Data:")
            print(json.dumps(data, indent=2))
        elif choice == "2":
            interval = int(input("Enter monitoring interval in seconds (default 60): ") or "60")
            await continuous_monitoring(interval)
        elif choice == "3":
            await diagnostics_mode()
        elif choice == "4":
            await verify_connection_to_ac()
        elif choice == "5":
            await discover_plugs()
        else:
            print("Invalid choice")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")