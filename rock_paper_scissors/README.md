# RPS TASK

We also tried a sim2real Rock,Paper and scissor task.Firstly we trained it using ppo and then deployed it on hardware.

## Folder Structure

```
rock_paper_scissors/
├── amazedex_rps_env.py         #Has the env for the task
├── mediapipe_rps_gesture.py    #Detection file for RPS
├── README.md
├── register_amazedex_rps_env.py  #Register for the RPS env
├── sim2real_rps.py               #Deployment script
└── train_ppo_rps.py              #Training script
```