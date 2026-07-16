import time
import numpy as np

from rustypot import Scs0009PyController

# Change this to the IDs you want to test
IDS = [1, 2, 3, 4, 5, 6, 7, 8]

# Middle position (degrees)
MiddlePos = 0

c = Scs0009PyController(
    serial_port="COM14",
    baudrate=1000000,
    timeout=0.6,
)


def main():
    # Enable torque
    for ID in IDS:
        c.write_torque_enable(ID, 1)  # 1 = On

    while True:
        print(f"Testing Servos {IDS} - Close")
        close_servo()
        time.sleep(2)

        print(f"Testing Servos {IDS} - Open")
        open_servo()
        time.sleep(2)

        # Uncomment if you want to read the current position
        for ID in IDS:
             pos = c.read_present_position(ID)
             print(f"ID {ID} Position: {np.rad2deg(pos):.2f}°")
        time.sleep(0.1)


def close_servo():
    pos = np.deg2rad(MiddlePos + 90)
    for ID in IDS:
        c.write_goal_speed(ID, 1)  # Max speed
        c.write_goal_position(ID, pos)


def open_servo():
    pos = np.deg2rad(MiddlePos - 30)
    for ID in IDS:
        c.write_goal_speed(ID, 1)  # Max speed
        c.write_goal_position(ID, pos)


if __name__ == "__main__":
    main()