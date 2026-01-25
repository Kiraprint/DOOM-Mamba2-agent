import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import vizdoom.gymnasium_wrapper  # register Vizdoom envs

from mamba_model import SSDMamba2Combatant


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DOOM agent")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="ssd_mamba2_vizdoom_ppo.pth",
        help="Path to model checkpoint (default: ssd_mamba2_vizdoom_ppo.pth)"
    )
    parser.add_argument(
        "--env", "-e",
        type=str,
        default="VizdoomCorridor-v1",
        help="Environment to evaluate on (default: VizdoomCorridor-v1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.03,
        help="Delay between frames in seconds (default: 0.03)"
    )
    return parser.parse_args()


GYM_ENV_IDS = {
    "VizdoomBasic-v1": "VizdoomBasic-MultiBinary-v1",
    "VizdoomCorridor-v1": "VizdoomCorridor-MultiBinary-v1",
    "VizdoomDefendCenter-v1": "VizdoomDefendCenter-MultiBinary-v1",
    "VizdoomDefendLine-v1": "VizdoomDefendLine-MultiBinary-v1",
    "VizdoomHealthGathering-v1": "VizdoomHealthGathering-MultiBinary-v1",
}

SCENARIO_BUTTON_INDICES = {
    "VizdoomBasic-v1": [2, 3, 6],
    "VizdoomCorridor-v1": list(range(7)),
    "VizdoomDefendCenter-v1": [4, 5, 6],
    "VizdoomDefendLine-v1": [4, 5, 6],
    "VizdoomHealthGathering-v1": [0, 4, 5],
}


class ExtractScreenWrapper(gym.ObservationWrapper):
    """Custom wrapper to extract only the screen image buffer from ViZDoom's Dict."""

    def __init__(self, env):
        super().__init__(env)
        # ViZDoom images are typically (H, W, C) or (C, H, W)
        self.observation_space = env.observation_space.spaces["screen"]

    def observation(self, obs):
        return obs["screen"]


class CombatActionWrapper(gym.ActionWrapper):
    """Maps [3,3,3,2] to 7-button binary, then subset per scenario."""

    def __init__(self, env, button_indices=None):
        super().__init__(env)
        self.button_indices = button_indices if button_indices is not None else list(range(7))

    def action(self, action):
        new_action = np.zeros(7, dtype=np.int32)
        if action[0] == 1:
            new_action[0] = 1
        elif action[0] == 2:
            new_action[1] = 1
        if action[1] == 1:
            new_action[2] = 1
        elif action[1] == 2:
            new_action[3] = 1
        if action[2] == 1:
            new_action[4] = 1
        elif action[2] == 2:
            new_action[5] = 1
        if action[3] == 1:
            new_action[6] = 1
        return new_action[self.button_indices].copy()


# 1. Re-initialize Environment with Rendering
def make_eval_env(env_id):
    """Create eval environment. Uses -MultiBinary-v1 and scenario button subset."""
    gym_id = GYM_ENV_IDS.get(env_id, "VizdoomCorridor-MultiBinary-v1")
    indices = SCENARIO_BUTTON_INDICES.get(env_id, list(range(7)))
    env = gym.make(gym_id, render_mode="human")
    env = ExtractScreenWrapper(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = CombatActionWrapper(env, button_indices=indices)
    return env


args = parse_args()

print(f"Loading model: {args.model}")
print(f"Environment: {args.env}")
print("-" * 50)

env = make_eval_env(args.env)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Model
# Ensure you use the exact same architecture class from training
model = SSDMamba2Combatant().to(device)

# Handle torch.compile() prefix in saved weights
state_dict = torch.load(args.model, weights_only=True)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()  # Set to evaluation mode
print(f"Model loaded successfully!")

# 3. Inference Loop
obs, info = env.reset()
total_reward = 0.0
episode_rewards = []  # Track all episode rewards for histogram

print("Watching the agent... Press Ctrl+C to stop and see statistics.")

try:
    while True:
        # Convert observation to tensor and add batch dimension
        # Shape: (1, 4, 84, 84, 1)
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)

        with torch.inference_mode():
            # Use get_action_and_value for proper multi-discrete action handling
            action, _, _, _ = model.get_action_and_value(obs_tensor)
            action_np = action.cpu().numpy()[0]  # Convert to numpy for env.step()

        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward

        # Window is already visible via game config, just delay for viewing
        time.sleep(args.delay)

        if terminated or truncated:
            episode_rewards.append(total_reward)
            print(f"Episode {len(episode_rewards)} Finished. Score: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    env.close()

# Show statistics and histogram if we have data
if episode_rewards:
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Episodes played: {len(episode_rewards)}")
    print(f"Mean score: {np.mean(episode_rewards):.2f}")
    print(f"Std: {np.std(episode_rewards):.2f}")
    print(f"Min: {np.min(episode_rewards):.2f}")
    print(f"Max: {np.max(episode_rewards):.2f}")
    print(f"Median: {np.median(episode_rewards):.2f}")
    
    # Plot reward histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(episode_rewards, bins=min(30, len(episode_rewards)), 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax.axvline(np.median(episode_rewards), color='orange', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(episode_rewards):.1f}')
    ax.set_xlabel('Episode Reward', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Agent Evaluation - Reward Distribution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save with model name in filename
    model_name = Path(args.model).stem
    hist_path = f"eval_{model_name}_histogram.png"
    plt.savefig(hist_path, dpi=150)
    print(f"\nHistogram saved to '{hist_path}'")
    plt.show()
else:
    print("No episodes completed.")
