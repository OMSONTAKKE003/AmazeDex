import time
import numpy as np

from rustypot import Scs0009PyController


ID_1 = 3 # Change to servo ID you want to calibrate 
ID_2 = 4 # Change to servo ID you want to calibrate 
MiddlePos_1 = 0 # Middle position for servo ID_1 
MiddlePos_2 = 0 # Middle position for servo ID_2


c = Scs0009PyController(
    serial_port="/dev/ttyACM0",
    baudrate=1000000,
    timeout=0.5,
)


def reset_to_home():
    """Slowly moves servos back to center position 511 (0 rad)."""
    slow_speed = 0.5  # Slow speed for safe return
    c.write_goal_speed(ID_1, slow_speed)
    c.write_goal_speed(ID_2, slow_speed)

    try:
        # Write raw position 511 (Feetech center position) directly
        c.write_raw_goal_position(ID_1, -511)
        c.write_raw_goal_position(ID_2, 511)
    except AttributeError:
        # Fallback if rustypot controller uses radians (0.0 rad = 511 raw)
        c.write_goal_position(ID_1, 0.0)
        c.write_goal_position(ID_2, 0.0)

    time.sleep(2.0)  # Allow time for servos to complete movement

    # Disable torque safely after stopping
    c.write_torque_enable(ID_1, 0)
    c.write_torque_enable(ID_2, 0)
def CloseFinger():
    # Use integer speeds (e.g., 30 for controlled motion, 100 for faster)
    c.write_goal_speed(ID_1, 0.5) 
    c.write_goal_speed(ID_2, 0.5)
    
    Pos_1 = np.deg2rad(MiddlePos_1 + 90)
    Pos_2 = np.deg2rad(MiddlePos_2 - 90)
    
    c.write_goal_position(ID_1, Pos_1)
    c.write_goal_position(ID_2, Pos_2)


def OpenFinger():
    c.write_goal_speed(ID_1, 0.5)
    c.write_goal_speed(ID_2, 0.5)
    
    Pos_1 = np.deg2rad(MiddlePos_1 - 90)
    Pos_2 = np.deg2rad(MiddlePos_2 + 90)
    
    c.write_goal_position(ID_1, Pos_1)
    c.write_goal_position(ID_2, Pos_2)

def main():
    # Enable torque for configured IDs
    c.write_torque_enable(ID_1, 1)
    c.write_torque_enable(ID_2, 1)

    try:
        while True:
            CloseFinger()
            time.sleep(3)

            OpenFinger()
            time.sleep(1)

            # Read and cast to scalar floats to avoid numpy format errors
            raw_a = c.read_present_position(ID_1)
            raw_b = c.read_present_position(ID_2)

            a = float(np.rad2deg(raw_a))
            b = float(np.rad2deg(raw_b))

            print(f"{a:.2f}° {b:.2f}°")
            time.sleep(0.001)

    except KeyboardInterrupt:
        print(
            "\n[INFO] KeyboardInterrupt detected. Slowly returning to position"
            " 511..."
        )
        reset_to_home()
        print("[INFO] Safe shutdown complete.")


if __name__ == '__main__':
    main()