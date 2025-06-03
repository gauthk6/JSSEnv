#!/usr/bin/env python3

"""
Enhanced model evaluation script that supports size categories and comprehensive testing
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions.categorical import Categorical
from typing import Dict, List

import JSSEnv

# Define size categories and their test instances (matching scaling_study.py)
SIZE_CATEGORIES = {
    "small": {
        "name": "15x15_small",
        "test_instances": ["ta06", "ta07", "ta08", "ta09", "ta10"],
        "jobs": 15,
        "machines": 15
    },
    "medium": {
        "name": "20x20_medium", 
        "test_instances": ["ta26", "ta27", "ta28", "ta29", "ta30"],
        "jobs": 20,
        "machines": 20
    },
    "medium_large": {
        "name": "20x15_medium_large",
        "test_instances": ["ta46", "ta47", "ta48", "ta49", "ta50"], 
        "jobs": 20,
        "machines": 15
    },
    "large": {
        "name": "50x15_large",
        "test_instances": ["ta56", "ta57", "ta58", "ta59", "ta60"],
        "jobs": 50,
        "machines": 15
    },
    "very_large": {
        "name": "50x20_very_large",
        "test_instances": ["ta66", "ta67", "ta68", "ta69", "ta70"],
        "jobs": 50,
        "machines": 20
    }
}

class Agent(nn.Module):
    """Same agent architecture as in train_ppo_multi.py"""
    def __init__(self, envs):
        super().__init__()
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained model (.pt file)")
    
    # Support both individual instance and size category
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance", type=str,
                       help="Individual instance name to evaluate on (e.g., ta01)")
    group.add_argument("--size", type=str, choices=list(SIZE_CATEGORIES.keys()),
                       help="Size category to evaluate on (small, medium, medium_large, large, very_large)")
    
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of episodes to run per instance")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (cpu/cuda)")
    parser.add_argument("--max-episode-steps", type=int, default=3000,
                       help="Maximum steps per episode")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic policy (no exploration)")
    return parser.parse_args()

def make_env(env_id, instance_name, seed, max_episode_steps=None):
    """Create environment for a specific instance"""
    def thunk():
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Find instance file
        instance_path_full = os.path.join(project_root, "JSSEnv", "envs", "instances", instance_name)
        if not os.path.exists(instance_path_full):
            alt_instance_path = os.path.join("JSSEnv", "envs", "instances", instance_name)
            if os.path.exists(alt_instance_path):
                instance_path_full = alt_instance_path
            else:
                raise FileNotFoundError(f"Instance file '{instance_name}' not found!")
        
        env = gym.make(env_id, env_config={"instance_path": instance_path_full})
        
        # Override max_episode_steps if specified
        if max_episode_steps is not None:
            if hasattr(env.unwrapped, 'max_episode_steps'):
                original_steps = env.unwrapped.max_episode_steps
                env.unwrapped.max_episode_steps = max_episode_steps
                print(f"   Overriding max_episode_steps: {original_steps} -> {max_episode_steps}")
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        
        return env
    return thunk

def evaluate_single_instance(model_path: str, instance_name: str, episodes: int, 
                            seed: int, max_episode_steps: int, deterministic: bool, device):
    """Evaluate model on a single instance"""
    
    # Create environment
    env_id = "JSSEnv/JssEnv-v1"
    env = gym.vector.SyncVectorEnv([make_env(env_id, instance_name, seed, max_episode_steps)])
    
    # Create and load agent
    agent = Agent(env).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Evaluation metrics
    episode_returns = []
    episode_lengths = []
    episode_makespans = []
    episode_success = []
    episode_times = []
    inference_times = []
    
    print(f"   Evaluating {instance_name}...")
    
    for episode in range(episodes):
        episode_start_time = time.time()
        episode_inference_times = []
        
        obs_tuple = env.reset(seed=seed + episode)
        obs_dict = obs_tuple[0]
        
        # Ensure correct dimensions for vectorized env
        current_real_obs_np = obs_dict["real_obs"]
        current_action_mask_np = obs_dict["action_mask"]
        
        if current_real_obs_np.ndim == len(env.single_observation_space["real_obs"].shape):
            current_real_obs_np = np.expand_dims(current_real_obs_np, axis=0)
        if current_action_mask_np.ndim == len(env.single_observation_space["action_mask"].shape):
            current_action_mask_np = np.expand_dims(current_action_mask_np, axis=0)
        
        obs_real = torch.tensor(current_real_obs_np, dtype=torch.float32).to(device)
        obs_action_mask = torch.tensor(current_action_mask_np, dtype=torch.bool).to(device)
        
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # Time the inference
            inference_start = time.time()
            
            with torch.no_grad():
                current_obs_for_agent = {"real_obs": obs_real}
                
                if deterministic:
                    # Use deterministic policy (argmax)
                    logits = agent.actor(agent._flatten_real_obs(obs_real))
                    if obs_action_mask is not None:
                        logits[~obs_action_mask] = -1e8
                    action = torch.argmax(logits, dim=1)
                else:
                    # Use stochastic policy (sampling)
                    action, _, _, _ = agent.get_action_and_value(current_obs_for_agent, action_mask=obs_action_mask)
            
            inference_time = time.time() - inference_start
            episode_inference_times.append(inference_time)
            
            # Step environment
            step_result = env.step(action.cpu().numpy())
            next_obs_dict, reward, terminated, truncated, info = step_result
            
            done = terminated[0] or truncated[0]
            episode_reward += reward[0]
            step_count += 1
            
            if not done:
                # Update observations for next step
                next_real_obs_np = next_obs_dict["real_obs"]
                next_action_mask_np = next_obs_dict["action_mask"]
                
                if next_real_obs_np.ndim == len(env.single_observation_space["real_obs"].shape):
                    next_real_obs_np = np.expand_dims(next_real_obs_np, axis=0)
                if next_action_mask_np.ndim == len(env.single_observation_space["action_mask"].shape):
                    next_action_mask_np = np.expand_dims(next_action_mask_np, axis=0)
                
                obs_real = torch.tensor(next_real_obs_np, dtype=torch.float32).to(device)
                obs_action_mask = torch.tensor(next_action_mask_np, dtype=torch.bool).to(device)
        
        episode_time = time.time() - episode_start_time
        
        # Extract episode statistics
        episode_returns.append(episode_reward)
        episode_lengths.append(step_count)
        episode_times.append(episode_time)
        inference_times.extend(episode_inference_times)
        
        # Get makespan and success from environment
        makespan = None
        all_jobs_completed = False
        
        if hasattr(env.envs[0], 'env'):
            underlying_env = env.envs[0].env
            while hasattr(underlying_env, 'env'):
                underlying_env = underlying_env.env
            if hasattr(underlying_env, 'current_time_step'):
                makespan = underlying_env.current_time_step
            if hasattr(underlying_env, '_are_all_jobs_completed'):
                all_jobs_completed = underlying_env._are_all_jobs_completed()
        
        if makespan is not None:
            episode_makespans.append(makespan)
            episode_success.append(all_jobs_completed)
    
    env.close()
    
    # Compute statistics for this instance
    instance_results = {
        "instance": instance_name,
        "episodes": episodes,
        "stats": {
            "returns": {
                "mean": float(np.mean(episode_returns)),
                "std": float(np.std(episode_returns)),
                "min": float(np.min(episode_returns)),
                "max": float(np.max(episode_returns))
            },
            "lengths": {
                "mean": float(np.mean(episode_lengths)),
                "std": float(np.std(episode_lengths)),
                "min": int(np.min(episode_lengths)),
                "max": int(np.max(episode_lengths))
            },
            "inference_time": {
                "mean_ms": float(np.mean(inference_times) * 1000),
                "std_ms": float(np.std(inference_times) * 1000),
                "total_actions": len(inference_times)
            }
        }
    }
    
    if episode_makespans:
        instance_results["stats"]["makespans"] = {
            "mean": float(np.mean(episode_makespans)),
            "std": float(np.std(episode_makespans)),
            "min": float(np.min(episode_makespans)),
            "max": float(np.max(episode_makespans))
        }
        instance_results["stats"]["success_rate"] = float(np.mean(episode_success))
    
    return instance_results

def evaluate_size_category(model_path: str, size_category: str, episodes: int, 
                          seed: int, max_episode_steps: int, deterministic: bool, 
                          device, output_dir: str):
    """Evaluate model on all instances in a size category"""
    
    if size_category not in SIZE_CATEGORIES:
        raise ValueError(f"Unknown size category: {size_category}")
    
    category_info = SIZE_CATEGORIES[size_category]
    test_instances = category_info["test_instances"]
    
    print(f"\n🚀 Evaluating {category_info['name']} ({category_info['jobs']}x{category_info['machines']}) on {len(test_instances)} instances")
    print(f"   Test instances: {', '.join(test_instances)}")
    print(f"   Episodes per instance: {episodes}")
    
    all_results = {}
    aggregated_stats = {
        "returns": [],
        "makespans": [],
        "success_rates": [],
        "inference_times": [],
        "episode_lengths": []
    }
    
    start_time = time.time()
    
    for i, instance in enumerate(test_instances):
        print(f"\n   📊 Instance {i+1}/{len(test_instances)}: {instance}")
        instance_results = evaluate_single_instance(
            model_path, instance, episodes, seed + i * 1000, 
            max_episode_steps, deterministic, device
        )
        
        # Store individual results
        all_results[instance] = instance_results
        
        # Aggregate for overall statistics
        aggregated_stats["returns"].append(instance_results["stats"]["returns"]["mean"])
        aggregated_stats["episode_lengths"].append(instance_results["stats"]["lengths"]["mean"])
        aggregated_stats["inference_times"].append(instance_results["stats"]["inference_time"]["mean_ms"])
        
        if "makespans" in instance_results["stats"]:
            aggregated_stats["makespans"].append(instance_results["stats"]["makespans"]["mean"])
            aggregated_stats["success_rates"].append(instance_results["stats"]["success_rate"])
        
        # Print instance summary
        print(f"      Return: {instance_results['stats']['returns']['mean']:.2f}, "
              f"Length: {instance_results['stats']['lengths']['mean']:.1f}, "
              f"Inference: {instance_results['stats']['inference_time']['mean_ms']:.2f}ms", end="")
        
        if "makespans" in instance_results["stats"]:
            print(f", Makespan: {instance_results['stats']['makespans']['mean']:.1f}, "
                  f"Success: {instance_results['stats']['success_rate']*100:.1f}%")
        else:
            print()
    
    total_time = time.time() - start_time
    
    # Compute aggregated statistics across all instances
    final_results = {
        "size_category": size_category,
        "category_info": category_info,
        "model_path": model_path,
        "episodes_per_instance": episodes,
        "total_instances": len(test_instances),
        "total_episodes": episodes * len(test_instances),
        "evaluation_time": total_time,
        "individual_results": all_results,
        "aggregated_stats": {
            "returns": {
                "mean": float(np.mean(aggregated_stats["returns"])),
                "std": float(np.std(aggregated_stats["returns"])),
                "min": float(np.min(aggregated_stats["returns"])),
                "max": float(np.max(aggregated_stats["returns"]))
            },
            "lengths": {
                "mean": float(np.mean(aggregated_stats["episode_lengths"])),
                "std": float(np.std(aggregated_stats["episode_lengths"])),
                "min": float(np.min(aggregated_stats["episode_lengths"])),
                "max": float(np.max(aggregated_stats["episode_lengths"]))
            },
            "inference_time": {
                "mean_ms": float(np.mean(aggregated_stats["inference_times"])),
                "std_ms": float(np.std(aggregated_stats["inference_times"])),
                "min_ms": float(np.min(aggregated_stats["inference_times"])),
                "max_ms": float(np.max(aggregated_stats["inference_times"]))
            }
        }
    }
    
    if aggregated_stats["makespans"]:
        final_results["aggregated_stats"]["makespans"] = {
            "mean": float(np.mean(aggregated_stats["makespans"])),
            "std": float(np.std(aggregated_stats["makespans"])),
            "min": float(np.min(aggregated_stats["makespans"])),
            "max": float(np.max(aggregated_stats["makespans"]))
        }
        final_results["aggregated_stats"]["success_rate"] = {
            "mean": float(np.mean(aggregated_stats["success_rates"])),
            "std": float(np.std(aggregated_stats["success_rates"])),
            "min": float(np.min(aggregated_stats["success_rates"])),
            "max": float(np.max(aggregated_stats["success_rates"]))
        }
    
    # Print final summary
    print(f"\n📊 {size_category.upper()} CATEGORY SUMMARY:")
    print(f"   Average Return: {final_results['aggregated_stats']['returns']['mean']:.2f} ± {final_results['aggregated_stats']['returns']['std']:.2f}")
    print(f"   Average Length: {final_results['aggregated_stats']['lengths']['mean']:.1f} ± {final_results['aggregated_stats']['lengths']['std']:.1f}")
    if aggregated_stats["makespans"]:
        print(f"   Average Makespan: {final_results['aggregated_stats']['makespans']['mean']:.1f} ± {final_results['aggregated_stats']['makespans']['std']:.1f}")
        print(f"   Average Success Rate: {final_results['aggregated_stats']['success_rate']['mean']*100:.1f}% ± {final_results['aggregated_stats']['success_rate']['std']*100:.1f}%")
    print(f"   Average Inference Time: {final_results['aggregated_stats']['inference_time']['mean_ms']:.2f} ± {final_results['aggregated_stats']['inference_time']['std_ms']:.2f} ms/action")
    print(f"   Total Evaluation Time: {total_time:.1f}s")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/eval_{size_category}_{int(time.time())}.json"
    
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"💾 Results saved to {output_file}")
    
    return final_results

if __name__ == "__main__":
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = torch.device("cpu")
    
    print(f"📦 Loading model from {args.model_path}")
    
    try:
        if args.size:
            # Evaluate on size category
            results = evaluate_size_category(
                args.model_path, args.size, args.episodes, args.seed,
                args.max_episode_steps, args.deterministic, device, args.output_dir
            )
        else:
            # Evaluate on single instance
            print(f"\n🚀 Evaluating on single instance: {args.instance}")
            results = evaluate_single_instance(
                args.model_path, args.instance, args.episodes, args.seed,
                args.max_episode_steps, args.deterministic, device
            )
            
            # Print summary for single instance
            print(f"\n📊 Evaluation Summary for {args.instance}:")
            print(f"   Mean Return: {results['stats']['returns']['mean']:.2f} ± {results['stats']['returns']['std']:.2f}")
            print(f"   Mean Length: {results['stats']['lengths']['mean']:.1f} ± {results['stats']['lengths']['std']:.1f}")
            if "makespans" in results["stats"]:
                print(f"   Mean Makespan: {results['stats']['makespans']['mean']:.1f} ± {results['stats']['makespans']['std']:.1f}")
                print(f"   Success Rate: {results['stats']['success_rate']*100:.1f}%")
            print(f"   Mean Inference Time: {results['stats']['inference_time']['mean_ms']:.2f} ± {results['stats']['inference_time']['std_ms']:.2f} ms/action")
            
            # Save single instance results
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = f"{args.output_dir}/eval_{args.instance}_{int(time.time())}.json"
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to {output_file}")
        
        print("✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        exit(1) 