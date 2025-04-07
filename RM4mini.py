import time
import broadlink
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RM4mini")

DEVICE_NAME = "VERGA1"

# Single IR code that acts as a toggle
toggle_code = b'&\x00P\x00\x00\x01 \x8f\x10\x13\x10\x13\x10\x13\x104\x104\x11\x12\x11\x12\x113\x104\x105\x104\x103\x11\x12\x114\x113\x10\x13\x104\x10\x13\x10\x13\x104\x113\x11\x12\x11\x12\x11\x12\x10\x13\x104\x105\x0f\x14\x0f\x13\x104\x113\x114\x0f\x00\x04Y\x00\x01 E\x0f\x00\r\x05'

# Track last command time to prevent super rapid toggling
last_command_time = 0
MIN_COMMAND_INTERVAL = 5  # 5 seconds between commands

def discover_devices(timeout=5):
    """Discover all Broadlink devices on the network"""
    logger.info(f"Searching for Broadlink devices...")
    devices = broadlink.discover(timeout=timeout)
    
    if not devices:
        logger.error("No Broadlink devices found!")
        return None
    
    logger.info(f"Found {len(devices)} Broadlink device(s):")
    for i, dev in enumerate(devices):
        if hasattr(dev, "name"):
            logger.info(f"  Device {i+1}: {dev.name} ({dev.type})")
        else:
            logger.info(f"  Device {i+1}: unnamed ({dev.type})")
    
    return devices

def get_device_by_name(name=DEVICE_NAME, timeout=5):
    """Find a specific Broadlink device by name"""
    devices = discover_devices(timeout)
    if not devices:
        return None
    
    for dev in devices:
        if hasattr(dev, "name") and dev.name == name:
            dev.auth()
            logger.info(f"✅ Connected to {name}")
            return dev
    
    logger.error(f"❌ ERROR: Device '{name}' not found!")
    return None

def send_toggle_command(device):
    """Send the IR toggle command to the device"""
    logger.info(f"Sending IR toggle command")
    try:
        device.send_data(toggle_code)
        logger.info(f"📡 Sent IR toggle command")
        return True
    except Exception as e:
        logger.error(f"Failed to send IR command: {e}")
        return False

def set_ac_state(desired_state, current_state):
    """
    Set AC to desired state based on current state
    
    Args:
        desired_state (bool): True=ON, False=OFF
        current_state (bool): Current state determined from power readings
                              True=ON (>400W), False=OFF/Standby (<400W)
    
    Returns:
        bool: True if action taken, False if error or no action needed
    """
    global last_command_time
    
    # If already in desired state, do nothing
    if desired_state == current_state:
        logger.info(f"AC is already {'ON' if desired_state else 'OFF/STANDBY'}, no action needed")
        return True
    
    # Check for minimum interval between commands
    current_time = time.time()
    if (current_time - last_command_time) < MIN_COMMAND_INTERVAL:
        wait_time = MIN_COMMAND_INTERVAL - (current_time - last_command_time)
        logger.info(f"Waiting {wait_time:.1f} seconds before sending next command")
        time.sleep(wait_time)
    
    # Find the RM4mini device
    device = get_device_by_name()
    if not device:
        logger.error("Failed to find RM4mini device")
        return False
    
    # Send toggle command to change state
    logger.info(f"Sending toggle command to change state from {'ON' if current_state else 'OFF/STANDBY'} to {'ON' if desired_state else 'OFF/STANDBY'}")
    success = send_toggle_command(device)
    
    if success:
        # Update command time
        last_command_time = time.time()
    
    return success

def learn_ir_command():
    """Learn and print an IR command from the remote"""
    device = get_device_by_name()
    if not device:
        return
    
    print("Point your remote at the RM4mini and press the button...")
    print("Waiting for IR command...")
    
    device.enter_learning()
    time.sleep(10)  # Wait for 10 seconds to receive the IR command
    
    ir_packet = device.check_data()
    if ir_packet:
        print(f"IR command received: {ir_packet}")
        return ir_packet
    else:
        print("No IR command received.")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 2:
        action = sys.argv[1].lower()
        if action == "discover":
            discover_devices()
        elif action == "learn":
            learn_ir_command()
        elif action == "toggle":
            device = get_device_by_name()
            if device:
                send_toggle_command(device)
        else:
            print("❌ Invalid action. Use 'toggle', 'discover', or 'learn'.")
    else:
        print("❌ No action provided. Use 'toggle', 'discover', or 'learn'.")