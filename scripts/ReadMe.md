
# AmazeDex Project


## Overview

This repository contains the simulation environments, configuration files, and hardware control scripts for the AmazeDex robotic system.



## File Structure

Below is a brief overview of the files included in this project:

* **`amazedex_cube_env.py`**: Implements a custom Gymnasium style environment.


* **`config.json`**: Specifies configuration parameters and joint properties for the MuJoCo model.


* **`mj_mink_right.py`**: Runs a MuJoCo simulation to move the fingers using trignometric functions


* **`motorflash.py`**: Script to test and move the physical servo motors.


* **`mujoco_env.py`**: Defines the generic base MuJoCo environment class used for handling simulation physics and rendering.


* **`register_amazedex_env.py`**: Registers the custom AmazeDex environment with Gymnasium to enable standard instantiation.

* **`realpose.py`**:Contains the code  for 6D pose estimation in real.

* **`cameracaliberation.py`**:Contains the script  for  camera  caliberatiion of  for april tags.

* **`detector.py`**:Contains the code  for 6D pose estimation in mujoco.





---