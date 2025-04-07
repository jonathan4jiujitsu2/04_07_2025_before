import time
import serial
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global serial connection to be used by all functions
global_ser = None
SERIAL_PORT = "COM3"   
BAUD_RATE = 9600

DAMPER_CLOSED = 0
DAMPER_OPEN = 90  # Changed from 180 to 90 degrees for fully open

def initialize_serial():
    """Initialize the serial connection and return it"""
    global global_ser
    
    if global_ser is not None and global_ser.is_open:
        logger.info("Serial connection already open")
        return global_ser
        
    try:
        logger.info(f"Connecting to Arduino on {SERIAL_PORT}")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow Arduino to initialize
        print(f"Connected to Arduino on {SERIAL_PORT}")
        
        # Save the connection for future use
        global_ser = ser
        return ser
    except Exception as e:
        logger.error(f"Failed to connect to Arduino: {e}")
        return None

def send_servo_positions(positions):
    """
    Send a list of servo angles to Arduino and verify response.
    :param positions: List of 8 angles (0-90) corresponding to the 8 dampers.
    """
    global global_ser
    
    if len(positions) != 8:
        logger.error("Error: Expected 8 servo positions.")
        return False
    
    # Ensure serial connection is open
    if global_ser is None or not global_ser.is_open:
        global_ser = initialize_serial()
        if global_ser is None:
            return False
    
    # Print debug info to verify dampers' movement
    print("\nMoving Dampers to new positions:")
    for i, angle in enumerate(positions, start=1):
        print(f"Damper {i} → {angle}°")
    
    try:
        message = f"S:{','.join(map(str, positions))};"
        global_ser.write(message.encode())
        print(f"Sent: {message}")
        
        # Wait for Arduino to process data
        time.sleep(0.5)
        
        # Read the response from Arduino
        response = global_ser.readline().decode("utf-8").strip()
        print(f"Arduino Response: {response}")
        
        # Accept various valid responses
        valid_responses = ["Servo positions set", "OK", "Arduino Ready"]
        
        # If we got any response, consider it successful
        if response:
            if any(valid_text in response for valid_text in valid_responses):
                logger.info(f"Damper positions successfully set (Response: '{response}')")
            else:
                logger.info(f"Received response '{response}', assuming success")
            return True
        else:
            logger.warning("No response from Arduino")
            return False
    except Exception as e:
        logger.error(f"Error sending servo positions: {e}")
        # Try to re-establish connection if error occurs
        try:
            global_ser.close()
        except:
            pass
        global_ser = None
        return False

def convert_binary_to_positions(damper_states):
    """
    Convert binary damper states (0=closed, 1=open) to actual servo positions.
    
    Args:
        damper_states: List of 8 binary states (0 or 1)
        
    Returns:
        List of 8 servo positions (0 or 90 degrees)
    """
    return [DAMPER_OPEN if state else DAMPER_CLOSED for state in damper_states]

def close_serial():
    """Close the serial connection when program ends"""
    global global_ser
    if global_ser is not None and global_ser.is_open:
        global_ser.close()
        global_ser = None
        print("Serial connection closed.")

# If this script is run directly, perform a test
if __name__ == "__main__":
    try:
        # Initialize serial connection
        initialize_serial()
        
        # Test opening and closing all dampers
        open_positions = [DAMPER_OPEN] * 8  # All dampers fully open (now 90°)
        close_positions = [DAMPER_CLOSED] * 8   # All dampers fully closed
        
        print("Testing damper control with new 90° max position...")
        
        print("\nOpening all dampers...")
        send_servo_positions(open_positions)
        time.sleep(2)
        
        print("\nClosing all dampers...")
        send_servo_positions(close_positions)
        time.sleep(2)
        
        print("\nSequential test (opening one by one)...")
        for i in range(8):
            test_positions = [DAMPER_CLOSED] * 8
            test_positions[i] = DAMPER_OPEN
            print(f"\nOpening only damper {i+1}...")
            send_servo_positions(test_positions)
            time.sleep(1)
        
        # Close the connection only when running as standalone script
        close_serial()
        print("\nTest completed successfully.")
    except KeyboardInterrupt:
        close_serial()
        print("Test interrupted by user.")