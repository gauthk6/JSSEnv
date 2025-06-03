#!/usr/bin/env python3

"""
Multi-Instance PPO Training for JSSP Generalization
Based on CleanRL's PPO implementation, modified for training on multiple ta instances.
"""

import argparse
import os
import random
import time
from distutils.util import strtobool
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import JSSEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="JSSEnv/JssEnv-v1",
        help="the id of the environment")
    parser.add_argument("--train-instances", type=str, nargs="+", 
        default=["ta01", "ta02", "ta03", "ta04", "ta05"],
        help="list of training instances")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=4096,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=20,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    # Environment configuration
    parser.add_argument("--max-episode-steps", type=int, default=None,
        help="maximum number of steps per episode (overrides default environment limit)")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def make_env_multi_instance(env_id, seed, idx, capture_video, run_name, train_instances, max_episode_steps=None):
    def thunk():
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Create a custom wrapper that changes instance on each reset
        class MultiInstanceWrapper(gym.Wrapper):
            def __init__(self, env_id, train_instances, project_root, max_episode_steps=None):
                self.env_id = env_id
                self.train_instances = train_instances
                self.project_root = project_root
                self.current_instance = None
                self.max_episode_steps = max_episode_steps
                
                # Initialize with first instance
                self._create_env()
                super().__init__(self.env)
                
            def _create_env(self):
                # Randomly select instance
                instance_name = random.choice(self.train_instances)
                self.current_instance = instance_name
                
                instance_path_full = os.path.join(self.project_root, "JSSEnv", "envs", "instances", instance_name)
                
                # Fallback for different directory structures
                if not os.path.exists(instance_path_full):
                    alt_instance_path = os.path.join("JSSEnv", "envs", "instances", instance_name)
                    if os.path.exists(alt_instance_path):
                        instance_path_full = alt_instance_path
                
                # Create env_config with max_episode_steps
                env_config = {"instance_path": instance_path_full}
                if self.max_episode_steps is not None:
                    env_config["max_episode_steps"] = self.max_episode_steps
                    if idx == 0:  # Only print once
                        print(f"   Setting max_episode_steps in env_config: {self.max_episode_steps}")
                
                self.env = gym.make(self.env_id, env_config=env_config)
                
            def reset(self, **kwargs):
                # Create new environment with random instance on each reset
                self._create_env()
                obs, info = self.env.reset(**kwargs)
                return obs, info
        
        # Create the multi-instance wrapper
        env = MultiInstanceWrapper(env_id, train_instances, project_root, max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        
        return env
    return thunk

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Access single_observation_space from the SyncVectorEnv
        real_obs_space = envs.single_observation_space["real_obs"]
        if isinstance(real_obs_space, gym.spaces.Box):
            real_obs_shape = real_obs_space.shape
        else:
            raise ValueError(f"Expected 'real_obs' to be a Box space, got {type(real_obs_space)}")

        if len(real_obs_shape) > 1: 
            input_features = int(np.prod(real_obs_shape)) 
        else: 
            input_features = real_obs_shape[0]

        self.critic = nn.Sequential(
            nn.Linear(input_features, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(input_features, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n), 
        )

    def _flatten_real_obs(self, real_obs_batch):
        return real_obs_batch.reshape(real_obs_batch.shape[0], -1)

    def get_value(self, x_dict):
        real_obs = x_dict["real_obs"] 
        real_obs_flat = self._flatten_real_obs(real_obs)
        return self.critic(real_obs_flat)

    def get_action_and_value(self, x_dict, action=None, action_mask=None):
        real_obs = x_dict["real_obs"]
        real_obs_flat = self._flatten_real_obs(real_obs)
        logits = self.actor(real_obs_flat)
        value = self.critic(real_obs_flat)

        if action_mask is not None:
            action_mask_bool = action_mask.bool() 
            if logits.device != action_mask_bool.device:
                action_mask_bool = action_mask_bool.to(logits.device)
            logits[~action_mask_bool] = -1e8 
        
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

if __name__ == "__main__":
    args = parse_args()
    
    # Create human-readable timestamp for easier parsing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_id.split('/')[-1]}__{args.exp_name}__{timestamp}"
    
    if args.track:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cpu") 
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (CUDA)")
    #elif torch.backends.mps.is_available():
      #  device = torch.device("mps")
     #   print(f"Using device: {device} (Apple Silicon GPU)")
    elif args.cuda:
        print(f"CUDA specified but not available. Using device: {device} (CPU)")
    else:
        print(f"Using device: {device} (CPU)")
        
    print(f"Training instances: {args.train_instances}")
    
    # Verify all training instances exist
    project_root = os.path.dirname(os.path.abspath(__file__))
    for instance in args.train_instances:
        instance_path = os.path.join(project_root, "JSSEnv", "envs", "instances", instance)
        if not os.path.exists(instance_path):
            alt_path = os.path.join("JSSEnv", "envs", "instances", instance)
            if not os.path.exists(alt_path):
                print(f"ERROR: Instance file '{instance}' not found!")
                exit(1)
            else:
                print(f"Found instance {instance} at: {alt_path}")
        else:
            print(f"Found instance {instance} at: {instance_path}")

    envs = gym.vector.SyncVectorEnv(
        [make_env_multi_instance(args.env_id, args.seed + i, i, False, run_name, args.train_instances, args.max_episode_steps) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    assert isinstance(envs.single_observation_space, gym.spaces.Dict)
    
    print(f"Single Action Space: {envs.single_action_space}")
    print(f"Single Observation Space: {envs.single_observation_space}")
    real_obs_buffer_shape = envs.single_observation_space["real_obs"].shape
    action_mask_buffer_shape = envs.single_observation_space["action_mask"].shape
    print(f"Real Obs Shape (from env): {real_obs_buffer_shape}")
    print(f"Action Mask Shape (from env): {action_mask_buffer_shape}")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_real_obs = torch.zeros((args.num_steps, args.num_envs) + real_obs_buffer_shape, dtype=torch.float32).to(device)
    obs_action_mask = torch.zeros((args.num_steps, args.num_envs) + action_mask_buffer_shape, dtype=torch.bool).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    
    # Episode tracking across rollouts
    episode_returns = []
    episode_lengths = []
    episode_makespans = []
    episode_all_jobs_completed = []
    instance_episode_counts = {instance: 0 for instance in args.train_instances}
    
    next_obs_tuple = envs.reset(seed=args.seed)
    next_obs_dict = next_obs_tuple[0]

    current_real_obs_np = next_obs_dict["real_obs"]
    current_action_mask_np = next_obs_dict["action_mask"]

    # Ensure correct batch dimension for num_envs=1
    if args.num_envs == 1:
        if current_real_obs_np.ndim == len(real_obs_buffer_shape):
            current_real_obs_np = np.expand_dims(current_real_obs_np, axis=0)
        if current_action_mask_np.ndim == len(action_mask_buffer_shape):
            current_action_mask_np = np.expand_dims(current_action_mask_np, axis=0)

    next_real_obs = torch.tensor(current_real_obs_np, dtype=torch.float32).to(device)
    next_action_mask = torch.tensor(current_action_mask_np, dtype=torch.bool).to(device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32).to(device)
    
    num_updates = args.total_timesteps // args.batch_size
    print(f"\nStarting OPTIMIZED multi-instance training for {args.total_timesteps} timesteps ({num_updates} PPO updates).")
    print(f"Optimized hyperparameters:")
    print(f"  Learning rate: {args.learning_rate} (with annealing)")
    print(f"  Rollout steps: {args.num_steps}")
    print(f"  SGD epochs per update: {args.update_epochs}")
    print(f"  Minibatches: {args.num_minibatches} (batch size: {args.minibatch_size})")
    print(f"Rollout buffer size (num_envs * num_steps): {args.batch_size}")
    print(f"Minibatch size for SGD: {args.minibatch_size}")

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step_idx in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs_real_obs[step_idx] = next_real_obs
            obs_action_mask[step_idx] = next_action_mask
            dones[step_idx] = next_done

            with torch.no_grad():
                current_obs_for_agent = {"real_obs": next_real_obs}
                action, logprob, _, value = agent.get_action_and_value(current_obs_for_agent, action_mask=next_action_mask)
                values[step_idx] = value.flatten()
            actions[step_idx] = action
            logprobs[step_idx] = logprob

            next_obs_tuple_step = envs.step(action.cpu().numpy())
            next_obs_dict_step, reward_step, terminated_step, truncated_step, info_tuple_step = next_obs_tuple_step
            
            done_step_bool_np = np.logical_or(terminated_step, truncated_step)
            
            rewards[step_idx] = torch.tensor(reward_step, dtype=torch.float32).to(device).view(-1)
            
            next_real_obs_np_step = next_obs_dict_step["real_obs"]
            next_action_mask_np_step = next_obs_dict_step["action_mask"]
            if args.num_envs == 1:
                if next_real_obs_np_step.ndim == len(real_obs_buffer_shape):
                    next_real_obs_np_step = np.expand_dims(next_real_obs_np_step, axis=0)
                if next_action_mask_np_step.ndim == len(action_mask_buffer_shape):
                    next_action_mask_np_step = np.expand_dims(next_action_mask_np_step, axis=0)

            next_real_obs = torch.tensor(next_real_obs_np_step, dtype=torch.float32).to(device)
            next_action_mask = torch.tensor(next_action_mask_np_step, dtype=torch.bool).to(device)
            next_done = torch.tensor(done_step_bool_np, dtype=torch.float32).to(device)

            # Episode completion logging
            if args.num_envs == 1 and done_step_bool_np[0]:
                # Get current instance name from the wrapper
                current_instance = 'unknown'
                env_to_check = envs.envs[0]
                
                # Navigate through the wrapper chain to find our MultiInstanceWrapper
                while hasattr(env_to_check, 'env'):
                    if hasattr(env_to_check, 'current_instance'):
                        current_instance = env_to_check.current_instance
                        break
                    env_to_check = env_to_check.env
                
                # If we still haven't found it, check the outermost wrapper
                if current_instance == 'unknown' and hasattr(envs.envs[0], 'current_instance'):
                    current_instance = envs.envs[0].current_instance
                
                instance_episode_counts[current_instance] = instance_episode_counts.get(current_instance, 0) + 1
                
                # Handle vectorized environment info structure
                if isinstance(info_tuple_step, dict) and "episode" in info_tuple_step:
                    episode_info = info_tuple_step['episode']
                    episodic_return = float(episode_info['r'][0])
                    episodic_length = int(episode_info['l'][0])
                    
                    # Accumulate episode data
                    episode_returns.append(episodic_return)
                    episode_lengths.append(episodic_length)
                    
                    # Get makespan from environment
                    makespan = None
                    all_jobs_completed = False
                    if hasattr(envs.envs[0], 'env'):
                        underlying_env = envs.envs[0].env
                        while hasattr(underlying_env, 'env'):
                            underlying_env = underlying_env.env
                        if hasattr(underlying_env, 'current_time_step'):
                            makespan = underlying_env.current_time_step
                        if hasattr(underlying_env, '_are_all_jobs_completed'):
                            all_jobs_completed = underlying_env._are_all_jobs_completed()
                    
                    if makespan is not None:
                        episode_makespans.append(makespan)
                        episode_all_jobs_completed.append(all_jobs_completed)
                        
                        print(f"global_step={global_step}, instance={current_instance}, episodic_return={episodic_return:.2f}, episodic_length={episodic_length}")
                        print(f"  makespan={makespan:.2f}, all_jobs_completed={all_jobs_completed}")
                        
                        # Log to TensorBoard immediately (like train_ppo.py)
                        if writer:
                            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                            writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                            writer.add_scalar("charts/makespan", makespan, global_step)
                            writer.add_scalar("charts/all_jobs_completed", float(all_jobs_completed), global_step)

        # PPO Update (same as original)
        with torch.no_grad():
            next_value = agent.get_value({"real_obs": next_real_obs}).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten for PPO update
        b_obs_real_obs = obs_real_obs.reshape((-1,) + real_obs_buffer_shape)
        b_obs_action_mask = obs_action_mask.reshape((-1,) + action_mask_buffer_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs_dict = {"real_obs": b_obs_real_obs[mb_inds]}
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs_dict, b_actions[mb_inds], b_obs_action_mask[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Logging
        if update % 20 == 0:
            current_sps = int(global_step / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
            print(f"\nUpdate: {update}/{num_updates}, Global Step: {global_step}, SPS: {current_sps}")
            
            if len(episode_returns) > 0:
                print(f"  Episodes completed: {len(episode_returns)}, Avg Return: {np.mean(episode_returns):.2f}, Avg Length: {np.mean(episode_lengths):.1f}")
                if len(episode_makespans) > 0:
                    print(f"  Avg Makespan: {np.mean(episode_makespans):.2f}")
                    success_rate = np.mean(episode_all_jobs_completed) * 100
                    print(f"  Success Rate: {success_rate:.2f}%")
                
                print(f"  Instance distribution: {instance_episode_counts}")

        if writer:
            current_sps = int(global_step / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0
            
            # Calculate explained variance
            y_pred = b_values.cpu().numpy()
            y_true = b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            # Core metrics
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs) if clipfracs else 0.0, global_step)
            writer.add_scalar("charts/SPS", current_sps, global_step)
            writer.add_scalar("explained_variance", explained_var, global_step)
            
            # Episode metrics (with exact names user expects)
            if len(episode_returns) > 10:
                recent_returns = episode_returns[-10:]
                recent_lengths = episode_lengths[-10:]
                
                # Use the exact metric names the user expects
                writer.add_scalar("mean_episodic_return", np.mean(recent_returns), global_step)
                writer.add_scalar("mean_episodic_length", np.mean(recent_lengths), global_step)
                
                # Also keep the original names for compatibility
                writer.add_scalar("charts/episodic_return", np.mean(recent_returns), global_step)
                writer.add_scalar("charts/episodic_length", np.mean(recent_lengths), global_step)
                
                if len(episode_makespans) > 10:
                    recent_makespans = episode_makespans[-10:]
                    recent_success = episode_all_jobs_completed[-10:]
                    
                    # Use the exact metric names the user expects
                    writer.add_scalar("mean_makespan", np.mean(recent_makespans), global_step)
                    writer.add_scalar("all_jobs_completed", np.mean(recent_success), global_step)
                    
                    # Also keep the original names for compatibility
                    writer.add_scalar("charts/episodic_makespan", np.mean(recent_makespans), global_step)
                    writer.add_scalar("charts/success_rate", np.mean(recent_success), global_step)

    # Save final model
    final_model_path = f"runs/{run_name}/multi_instance_optimized_agent_{timestamp}.pt"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(agent.state_dict(), final_model_path)
    print(f"Training finished.")
    print(f"Final optimized agent model saved to {final_model_path}")

    envs.close()
    if writer:
        writer.close() 