import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import vizdoom as vzd
from vizdoom import gymnasium_wrapper

from mamba_model import SSDMamba2Combatant


class ExtractScreenWrapper(gym.ObservationWrapper):
    """Custom wrapper to extract only the screen image buffer from ViZDoom's Dict."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["screen"]

    def observation(self, obs):
        return obs["screen"]


class CombatActionWrapper(gym.ActionWrapper):
    """Maps model's [[0,1,2], [0,1,2], [0,1,2], [0,1]] to ViZDoom's flat binary list."""

    def action(self, action):
        # action is [Fwd/Back, Left/Right, TurnL/R, Shoot]
        # Map them to the 7 buttons configured in make_env
        new_action = np.zeros(7, dtype=np.int32)

        if action[0] == 1:
            new_action[0] = 1  # Fwd
        elif action[0] == 2:
            new_action[1] = 1  # Back

        if action[1] == 1:
            new_action[2] = 1  # Left
        elif action[1] == 2:
            new_action[3] = 1  # Right

        if action[2] == 1:
            new_action[4] = 1  # TurnL
        elif action[2] == 2:
            new_action[5] = 1  # TurnR

        if action[3] == 1:
            new_action[6] = 1  # Shoot

        return new_action


def make_env(env_id, seed: int):
    def thunk():
        # Setup simultaneous button capability
        env = gym.make(env_id, render_mode="rgb_array", max_buttons_pressed=0)
        game = env.unwrapped.game
        game.set_available_buttons(
            [
                vzd.Button.MOVE_FORWARD,
                vzd.Button.MOVE_BACKWARD,
                vzd.Button.MOVE_LEFT,
                vzd.Button.MOVE_RIGHT,
                vzd.Button.TURN_LEFT,
                vzd.Button.TURN_RIGHT,
                vzd.Button.ATTACK,
            ]
        )

        env = gym.wrappers.ResizeObservation(ExtractScreenWrapper(env), (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = CombatActionWrapper(env)
        return env

    return thunk


# Training Parameters
ENV_ID = "VizdoomCorridor-v1"
LEARNING_RATE = 2e-4
NUM_STEPS = 128          # Steps per rollout per environment
NUM_ENVS = 56            # Parallel VizDoom instances
TOTAL_TIMESTEPS = 2000000
PPO_EPOCHS = 8
BATCH_SIZE = 512         # Minibatch for optimization
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENTROPY_COEF = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use AsyncVectorEnv for parallel environment stepping (separate processes)
envs = gym.vector.AsyncVectorEnv([make_env(ENV_ID, i) for i in range(NUM_ENVS)])

agent = SSDMamba2Combatant().to(device)

# Compile model for faster execution (PyTorch 2.0+)
agent = torch.compile(agent, mode="reduce-overhead")

optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

# Storage Setup (Rollout Buffer)
num_heads = len(agent.action_dims)
batch_size = NUM_ENVS * NUM_STEPS

obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((NUM_STEPS, NUM_ENVS, num_heads)).to(device)
logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

# Pre-allocate CPU tensors with pinned memory for faster GPU transfer
next_obs_cpu = torch.zeros((NUM_ENVS,) + envs.single_observation_space.shape, pin_memory=True)
next_done_cpu = torch.zeros(NUM_ENVS, pin_memory=True)
reward_cpu = torch.zeros(NUM_ENVS, pin_memory=True)

# Initialize
obs_np, _ = envs.reset()
next_obs_cpu.copy_(torch.from_numpy(obs_np))
next_obs = next_obs_cpu.to(device, non_blocking=True)
next_done = torch.zeros(NUM_ENVS, device=device)

global_step = 0
episodic_rewards = np.zeros(NUM_ENVS)
reward_history = []

num_updates = TOTAL_TIMESTEPS // batch_size

for update in range(1, num_updates + 1):
    # PHASE 1: ROLLOUT
    for step in range(NUM_STEPS):
        global_step += NUM_ENVS
        obs[step] = next_obs
        dones[step] = next_done

        with torch.inference_mode():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()

        actions[step] = action
        logprobs[step] = logprob

        # Step environment (action transfer happens while env computes)
        action_np = action.cpu().numpy()
        obs_np, reward_np, term, trunc, info = envs.step(action_np)

        # Use pre-allocated pinned tensors for fast async transfer
        reward_cpu.copy_(torch.from_numpy(reward_np))
        rewards[step] = reward_cpu.to(device, non_blocking=True)

        # Track episode rewards
        episodic_rewards += reward_np
        done_mask = np.logical_or(term, trunc)
        for i in np.where(done_mask)[0]:
            reward_history.append(episodic_rewards[i])
            episodic_rewards[i] = 0

        # Fast transfer using pinned memory
        next_done_cpu.copy_(torch.from_numpy(done_mask.astype(np.float32)))
        next_done = next_done_cpu.to(device, non_blocking=True)
        next_obs_cpu.copy_(torch.from_numpy(obs_np))
        next_obs = next_obs_cpu.to(device, non_blocking=True)

    # PHASE 2: GAE (Advantage Estimation)
    # Ensure all async transfers complete before computation
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.inference_mode():
        _, _, _, next_value = agent.get_action_and_value(next_obs)
        next_value = next_value.reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - next_done
                next_v = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_v = values[t + 1]
            delta = rewards[t] + GAMMA * next_v * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            )
        returns = advantages + values

    # PHASE 3: PPO UPDATE
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1, num_heads))
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    for epoch in range(PPO_EPOCHS):
        inds = np.arange(batch_size)
        np.random.shuffle(inds)
        for start in range(0, batch_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            mb_inds = inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )

            # PPO Ratio Calculation
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            # Advantage Normalization
            mb_adv = b_advantages[mb_inds]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            # Policy Loss
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value Loss
            v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()

            # Entropy Loss
            entropy_loss = entropy.mean()

            loss = pg_loss - ENTROPY_COEF * entropy_loss + v_loss

            optimizer.zero_grad(set_to_none=True)  # Faster than setting to zero
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

    avg_reward = np.mean(reward_history[-100:]) if reward_history else 0
    print(f"Update {update}/{num_updates} | Step: {global_step} | Avg Score: {avg_reward:.2f} | Loss: {loss.item():.2f}")

# Save model
torch.save(agent.state_dict(), "ssd_mamba2_vizdoom_ppo.pth")
print("Training complete. Model saved.")
