from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from mcculw.enums import TempScale

# Correct DAQ-to-Channel Mapping (Each channel represents CXH & CXL)
daq_mapping = {
    "1DE4413": {  # DAQ #1
        0: ["BR3 (C0H)", "BR3 (C0L)"],
        1: ["BR3 DUCT (C1H)", "BR3 DUCT (C1L)"],
        2: ["BR4 (C2H)", "BR4 (C2L)"],
        3: ["BR4 DUCT (C3H)", "BR4 DUCT (C3L)"],
        4: ["BR1 (C4H)", "BR1 (C4L)"],
        6: ["BR2 duct (C6H)", "BR2 duct (C6L)"],
        7: ["BR2 (C7H)", "BR2 (C7L)"]
    },
    "19327AA": {  # DAQ #2
        0: ["TR3 DUCT (C0H)", "TR3 DUCT (C0L)"],
        1: ["TR3 (C1H)", "TR3 (C1L)"],
        2: ["TR4 (C2H)", "TR4 (C2L)"],
        3: ["TR4 DUCT (C3H)", "TR4 DUCT (C3L)"],
        4: ["TR1  (C4H)", "TR1  (C4L)"],
        5: ["TR1d(C5H)", "TR1d(C5L)"],
        6: ["TR2  (C6H)", "TR2  (C6L)"],
        7: ["TR2 DUCT (C7H)", "TR2 DUCT (C7L)"]
    },
    "2097739": {  # DAQ #5
        4: ["AC DUCT (C4H)", "AC DUCT (C4L)"],
        5: ["AC DUCT (C5H)", "AC DUCT (C5L)"],
        6: ["AMBIENT (C6H)", "AMBIENT (C6L)"],
        7: ["AMBIENT (C7H)", "AMBIENT (C7L)"]
    },
    "1932796": {  # DAQ #3
        7: ["TR4 (C7H)", "TR4 (C7L)"],
        6: ["BR4 (C6H)", "BR4 (C6L)"],
        5: ["TR3 (C5H)", "TR3 (C5L)"],
        4: ["BR3 (C4H)", "BR3 (C4L)"]
    },
    "2097737": {  # DAQ #4
        4: ["BR2 (C4H)", "BR2 (C4L)"],
        5: ["TR2 (C5H)", "TR2 (C5L)"],
        3: ["BR1 (C3H)", "BR1 (C3L)"],
        2: ["TR1 (C2H)", "TR1 (C2L)"]
    }
}

# Function to Read Temperature
def read_temperature(board_num, channel):
    """
    Reads temperature from a DAQ board and channel.
    Returns temperature (rounded) or an error message.
    """
    try:
        temp = ul.t_in(board_num, channel, TempScale.FAHRENHEIT)
        return round(temp, 2)
    except Exception as e:
        return f"Error reading from board {board_num}, channel {channel}: {e}"

# Detect Connected DAQ Devices
print("=== Detecting USB-TC Devices ===")
daq_boards = {}

for board_num in range(6):  # Check up to 6 DAQ boards
    try:
        daq_dev_info = DaqDeviceInfo(board_num)
        serial_number = daq_dev_info.unique_id

        if serial_number in daq_mapping:
            daq_boards[serial_number] = board_num
            print(f"Board #{board_num}: {daq_dev_info.product_name} (ID: {serial_number})")
        else:
            print(f"Board #{board_num}: {daq_dev_info.product_name} (Skipped)")
    except Exception as e:
        print(f"Board #{board_num}: Not Found - {e}")

# Read and Display Temperatures with Labels
print("\n=== Reading Temperatures from USB-TC ===")
for serial_number, board_num in daq_boards.items():
    print(f"\n--- DAQ {serial_number} ---")
    for channel, labels in daq_mapping[serial_number].items():
        temp = read_temperature(board_num, channel)
        if "Error 145" in str(temp):
            for label in labels:
                print(f"{label}: Open Connection (Check Thermocouple)")
        elif "Error" in str(temp):
            for label in labels:
                print(f"{label}: {temp}")
        else:
            for label in labels:
                print(f"{label}: {temp}°F")