## **AmazeDex**

This project aims to train a multi fingered robotic hand to autonomously rotate a cube to any desired target orientation using deep reinforcement learning. The platform utilizes the Amazing Hand by Pollen Robotics.


The robot receives information about the object’s current orientation and target orientation and learns finger movements that gradually align the object with the target pose. The training is performed entirely in a simulated environment, where the agent learns through trial and error by maximizing rewards based on orientation accuracy and grasp stability. The project demonstrates the application of deep reinforcement learning for dexterous manipulation and highlights the challenges of controlling multi-finger robotic systems for complex in hand manipulation and deploying it on hardware

## Folder Structure

* **`amazedex_cube_env`**: environment for amazedex with cube (2)  
* **`amazedex_rps_env`**: environment for rock paper scissor task   
* **`detector.py`**: detect tags  
* **`cube_face_cnn_detector.py`**: detect numbers using CNN   
* **`generate_face_dataset.py`**: generate dataset for cube for CNN training  
* **`mediapipe_rps_gesture.py`**: track rps movements  
* **`mj_mink_right.py`**: operate fingers manually for testing   
* **`motorflash.py`**: operate servos using rustypot  
* **`mujoco_env.py`**: environment for amazedex with cube (1)   
* **`register_amazedex_rps_env`**: register and test environment for rock paper scissor task   
* **`register_amazedex_cube_env`**: register and test environment for amazedex with cube    
* **`sim2real_rps.py`**: deployment for sim2real  
* **`train_face_cnn.py`**:train cnn  
* **`train_ppo_rps`**: train for rps   

