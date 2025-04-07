import time
import serial

# Define the serial port (COM3 in your case)
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

# Try to open the serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow Arduino to initialize
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Failed to connect to Arduino: {e}")
    exit()

# Function to send servo positions to Arduino
def send_servo_positions(positions):
    """
    Send a list of servo angles to Arduino.
    :param positions: List of 8 angles (0-180) corresponding to the 8 servos.
    """
    if len(positions) != 8:
        print("Error: Expected 8 servo positions.")
        return

    # Create the message with servo positions
    message = f"S:{','.join(map(str, positions))};"
    
    # Send message via serial
    ser.write(message.encode())
    print(f"Sent: {message}")
    
    # Wait for Arduino to process data
    time.sleep(0.5)
    
    # Read the response from Arduino (if any)
    response = ser.readline().decode("utf-8").strip()
    print(f"Arduino Response: {response}")

# Define servo positions (Open and Close)
open_positions = [180] * 8  # All servos at 180 degrees
close_positions = [0] * 8  # All servos at 0 degrees

# Send servo commands to Arduino
send_servo_positions(open_positions)
time.sleep(2)

send_servo_positions(close_positions)
time.sleep(2)

# Close serial connection after sending
ser.close()
print("Serial connection closed.")
