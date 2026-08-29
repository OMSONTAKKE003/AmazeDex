##  Sim2Real Pose Estimation

In simulation, the cube's orientation comes directly from MuJoCo's ground truth — this isn't available in the real world, so a way to estimate the cube's 6D pose from a camera was needed.

Options considered: NVIDIA **DOPE**, **MegaPose6D**, and classical **OpenCV + PnP solver**. All of these were expected to struggle with occlusion and inconsistent lighting, so **AprilTags** fixed to each cube face were used instead for a simpler, more robust solution.

<p align="center">
  <img src="../../assets/images/hand_pose_detection.png" alt="AprilTag pose estimation setup" width="450"/>
</p>

---