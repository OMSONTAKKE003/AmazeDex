from __future__ import annotations

import os
import torch
import psutil
import imageio
import wandb
import gymnasium as gym
import register_amazedex_env  # noqa: F401  -- registers AmazeDex/CubeRotate-v0 in the main process

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization
)
from wandb.integration.sb3 import WandbCallback

# ============================================================
# Configuration
# ============================================================
ENV_ID =  "AmazeDex/CubeGrasp-v0"
MAX_ENVS = 32
N_ENVS = min(os.cpu_count() or 16, int(os.environ.get("N_ENVS_OVERRIDE", MAX_ENVS)))
TOTAL_TIMESTEPS = 40_000_000
MODEL_DIR = "models"
LOG_DIR = "logs"
WANDB_PROJECT = "amazedex-cube"
WANDB_ENTITY = None
WANDB_RUN_NAME = None
LEARNING_RATE = 5e-5
N_STEPS = 1024
TARGET_BATCH_SIZE = 4096

# Only relevant for envs that expose a `set_curriculum_step(step)` method
# (e.g. AmazeDexCubeGraspEnv in amzedex_cube_env2.py). Should generally match
# the total_training_steps the env's RewardCurriculumManager was built with.
CURRICULUM_SYNC_EVERY_N_STEPS = 2048

# ============================================================
# Env Factory (subprocess-safe)
# ============================================================
def make_env(rank: int, seed: int = 0, env_kwargs: dict | None = None):
    """
    Returns a thunk that builds one instance of ENV_ID, safe to run inside
    a SubprocVecEnv worker.

    SB3's SubprocVecEnv defaults to the 'forkserver' start method (see its
    __init__: it prefers forkserver over fork whenever available, since fork
    isn't thread/CUDA-safe). A forkserver worker starts from a *fresh*
    interpreter -- it does NOT inherit whatever the main process already
    imported, so the `import register_amazedex_env` at the top of this file
    only registers "AmazeDex/CubeRotate-v0" in the main process. Each worker
    must re-run that import itself, or gym.make() raises
    `gymnasium.error.NameNotFound: Environment 'CubeRotate' doesn't exist in
    namespace AmazeDex.` the moment it's created -- which is what was
    happening. Doing the import inside this closure means it gets re-run
    every time a worker unpickles and calls this thunk.
    """
    def _init():
        import register_amazedex_env  # noqa: F401  -- re-registers inside the worker process
        kwargs = env_kwargs or {}
        env = gym.make(ENV_ID, **kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ============================================================
# Utility
# ============================================================
def largest_clean_divisor(buffer_size: int, target: int) -> int:
    """Return the largest divisor of buffer_size that is <= target."""
    for candidate in range(min(target, buffer_size), 0, -1):
        if buffer_size % candidate == 0:
            return candidate
    return buffer_size

# ============================================================
# Memory Guard
# ============================================================
class MemoryGuardCallback(BaseCallback):
    """Stop training gracefully before the Linux OOM killer terminates the process."""
    def __init__(self, save_path: str, check_every_n_steps: int = 2000, threshold_pct: float = 90.0, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_every_n_steps = check_every_n_steps
        self.threshold_pct = threshold_pct

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every_n_steps != 0:
            return True

        mem = psutil.virtual_memory()
        if self.verbose:
            print(f"[mem-guard] system RAM: {mem.percent:.1f}% used ({mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB)")

        if mem.percent < self.threshold_pct:
            return True

        os.makedirs(self.save_path, exist_ok=True)
        emergency_path = os.path.join(self.save_path, f"emergency_step_{self.num_timesteps}")
        
        print(f"\n[mem-guard] RAM at {mem.percent:.1f}% (>= {self.threshold_pct}%)")
        print(f"[mem-guard] saving emergency checkpoint to {emergency_path}")
        
        self.model.save(emergency_path)
        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(emergency_path + "_vecnormalize.pkl")
            
        print("[mem-guard] stopping training safely.")
        return False

# ============================================================
# Curriculum Sync
# ============================================================
class CurriculumSyncCallback(BaseCallback):
    """
    Keeps a per-env RewardCurriculumManager in sync with *global* training
    progress instead of each worker's local step count.

    With SubprocVecEnv + N_ENVS workers, an env that advances its own
    curriculum by 1 step per env.step() call only sees total_timesteps/N_ENVS
    "local" steps over the whole run -- so its curriculum would finish
    N_ENVS times slower than intended. This callback instead pushes
    model.num_timesteps (the true global count) into every worker via
    env_method(), each time it fires.

    Safe no-op for envs that don't define set_curriculum_step (e.g. the
    rotation env) -- the AttributeError is swallowed and sync is skipped.
    """
    def __init__(self, sync_every_n_steps: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.sync_every_n_steps = sync_every_n_steps
        self._warned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.sync_every_n_steps != 0:
            return True
        try:
            self.training_env.env_method("set_curriculum_step", self.num_timesteps)
            if self.verbose:
                print(f"[curriculum-sync] step {self.num_timesteps}: synced across all workers")
        except AttributeError:
            if not self._warned:
                print("[curriculum-sync] env has no set_curriculum_step(); skipping sync.")
                self._warned = True
        return True


# ============================================================
# Video Evaluation
# ============================================================
class VideoEvalCallback(BaseCallback):
    """Record one deterministic evaluation episode as an MP4."""
    def __init__(self, video_env, save_dir: str, record_every_n_steps: int, fps: int | None = None, max_steps: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.video_env = video_env
        self.raw_env = video_env.envs[0]
        self.save_dir = save_dir
        self.record_every_n_steps = record_every_n_steps
        self.fps = fps or self.raw_env.metadata.get("render_fps", 15)
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.record_every_n_steps != 0:
            return True
        self._record_video()
        return True

    def _record_video(self) -> None:
        if isinstance(self.training_env, VecNormalize) and isinstance(self.video_env, VecNormalize):
            sync_envs_normalization(self.training_env, self.video_env)

        frames = []
        obs = self.video_env.reset()
        done = False
        steps = 0

        while not done and steps < self.max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _reward, done_arr, _infos = self.video_env.step(action)
            done = bool(done_arr[0])
            steps += 1
            frame = self.raw_env.render()
            if frame is not None:
                frames.append(frame)

        if not frames:
            print(f"[video-eval] step {self.num_timesteps}: no frames captured (render() returned None).")
            return

        os.makedirs(self.save_dir, exist_ok=True)
        mp4_path = os.path.join(self.save_dir, f"eval_step_{self.num_timesteps}.mp4")

        try:
            imageio.mimsave(mp4_path, frames, fps=self.fps)
        except Exception as e:
            print(f"[video-eval] step {self.num_timesteps}: failed to write {mp4_path}: {e}")
            return

        print(f"[video-eval] step {self.num_timesteps}: saved {mp4_path} ({len(frames)} frames @ {self.fps} fps)")

        if wandb.run is not None:
            wandb.log({"eval/video": wandb.Video(mp4_path, fps=self.fps, format="mp4")}, step=self.num_timesteps)

# ============================================================
# Main Training
# ============================================================
def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(MODEL_DIR, "checkpoints")
    video_dir = os.path.join(checkpoint_dir, "videos")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    buffer_size = N_STEPS * N_ENVS
    batch_size = largest_clean_divisor(buffer_size, TARGET_BATCH_SIZE)

    print("\n========== TRAINING CONFIG ==========")
    print(f"Environment:       {ENV_ID}")
    print(f"N_ENVS:            {N_ENVS}")
    print(f"N_STEPS:           {N_STEPS}")
    print(f"Rollout buffer:    {buffer_size}")
    print(f"Target batch:      {TARGET_BATCH_SIZE}")
    print(f"Actual batch:      {batch_size}")
    print(f"Minibatches/epoch: {buffer_size // batch_size}")
    print("PPO epochs:        5")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Total timesteps:   {TOTAL_TIMESTEPS}")
    print("Device:            CUDA")
    print("=====================================\n")

    config = {
        "env_id": ENV_ID,
        "n_envs": N_ENVS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_steps": N_STEPS,
        "batch_size": batch_size,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "target_kl": 0.02,
        "learning_rate": LEARNING_RATE,
        "policy": "MlpPolicy",
        "device": "cuda",
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, gamma=config["gamma"])

    eval_env = DummyVecEnv([make_env(0, seed=1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    video_env = DummyVecEnv([make_env(0, seed=2000, env_kwargs={"render_mode": "rgb_array"})])
    video_env = VecNormalize(video_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        policy=config["policy"],
        env=train_env,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        target_kl=config["target_kl"],
        learning_rate=config["learning_rate"],
        device=config["device"],
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print("\n========== CUDA CHECK ==========")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("SB3 device:", model.device)
    print("Policy device:", next(model.policy.parameters()).device)
    print("================================\n")

    EVAL_EVERY_N_TOTAL_STEPS = 200_000
    eval_freq = max(EVAL_EVERY_N_TOTAL_STEPS // N_ENVS, 1)
    N_EVAL_EPISODES = 10

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=eval_freq,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(MODEL_DIR, "wandb"),
        model_save_freq=eval_freq * N_ENVS,
        verbose=2,
    )

    checkpoint_freq = max(1_000_000 // N_ENVS, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_rps",
        save_vecnormalize=True,
    )

    memory_guard_callback = MemoryGuardCallback(
        save_path=checkpoint_dir,
        check_every_n_steps=2000,
        threshold_pct=90.0,
        verbose=1,
    )

    curriculum_sync_callback = CurriculumSyncCallback(
        sync_every_n_steps=CURRICULUM_SYNC_EVERY_N_STEPS,
        verbose=1,
    )

    video_eval_callback = VideoEvalCallback(
        video_env=video_env,
        save_dir=video_dir,
        record_every_n_steps=checkpoint_freq,
        max_steps=1000,
        verbose=1,
    )

    callback = CallbackList([
        eval_callback,
        wandb_callback,
        checkpoint_callback,
        memory_guard_callback,
        curriculum_sync_callback,
        video_eval_callback,
    ])

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

        final_sb3_path = os.path.join(MODEL_DIR, "ppo_rps_final.zip")
        model.save(final_sb3_path)

        final_vecnormalize_path = os.path.join(MODEL_DIR, "vecnormalize_final.pkl")
        train_env.save(final_vecnormalize_path)

        pth_path = os.path.join(MODEL_DIR, "model.pth")
        torch.save(model.policy.state_dict(), pth_path)

        print("\n========== TRAINING COMPLETE ==========")
        print(f"SB3 Model:          {final_sb3_path}")
        print(f"VecNormalize stats: {final_vecnormalize_path}")
        print(f"PyTorch weights:    {pth_path}")
        print("========================================\n")

    finally:
        train_env.close()
        eval_env.close()
        video_env.close()
        run.finish()

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()