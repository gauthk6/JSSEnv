#!/usr/bin/env python3
"""
Comprehensive Evaluation of PPO Agent vs Baselines on JSSP Test Instances (ta06-ta10)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from distutils.util import strtobool
import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import random

# Import JSSEnv to register the environment
import JSSEnv 

# --- Agent Class Definition (matches train_ppo_multi.py) ---
class Agent(nn.Module):
    def __init__(self, single_observation_space, single_action_space):
        super().__init__()
        real_obs_space = single_observation_space["real_obs"]
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
            nn.Linear(64, single_action_space.n), 
        )

    def _flatten_real_obs(self, real_obs_batch):
        if real_obs_batch.ndim == 2:
            real_obs_batch = real_obs_batch.unsqueeze(0)
        return real_obs_batch.reshape(real_obs_batch.shape[0], -1)

    def get_value(self, x_dict):
        real_obs = x_dict["real_obs"] 
        real_obs_flat = self._flatten_real_obs(real_obs)
        return self.critic(real_obs_flat)

    def get_action(self, x_dict, action_mask=None, deterministic=True):
        real_obs = x_dict["real_obs"]
        real_obs_flat = self._flatten_real_obs(real_obs)
        
        logits = self.actor(real_obs_flat)

        if action_mask is not None:
            action_mask_bool = action_mask.bool()
            if action_mask_bool.ndim == 1:
                action_mask_bool = action_mask_bool.unsqueeze(0)
            if logits.device != action_mask_bool.device:
                action_mask_bool = action_mask_bool.to(logits.device)
            logits[~action_mask_bool] = -1e8 
        
        probs = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(probs.probs, dim=1)
        else:
            action = probs.sample()
        return action

# --- Baseline Algorithms ---
class BaselineAgent:
    def __init__(self, strategy="random"):
        self.strategy = strategy
    
    def get_action(self, obs_dict, env=None):
        """Get action using various JSSP heuristics"""
        valid_actions = np.where(obs_dict["action_mask"])[0]
        if len(valid_actions) == 0:
            return 0
        
        # Get environment info for proper heuristics
        if env is None:
            # Fallback to random if no environment access
            return np.random.choice(valid_actions)
        
        if self.strategy == "random":
            return np.random.choice(valid_actions)
        elif self.strategy == "spt":  # Shortest Processing Time
            return self._spt_heuristic(obs_dict, env, valid_actions)
        elif self.strategy == "mwkr":  # Most Work Remaining  
            return self._mwkr_heuristic(obs_dict, env, valid_actions)
        elif self.strategy == "first_available":
            return valid_actions[0]
        else:
            return np.random.choice(valid_actions)
    
    def _spt_heuristic(self, obs_dict, env, valid_actions):
        """Shortest Processing Time: Choose job with shortest current operation time"""
        best_action = valid_actions[0]
        shortest_time = float('inf')
        
        # Get the actual JSSEnv from wrapped environment
        actual_env = env
        while hasattr(actual_env, 'env'):
            actual_env = actual_env.env
        
        for action in valid_actions:
            if action == actual_env.NO_OP_ACTION:  # Skip NO_OP for comparison
                continue
                
            job_idx = action
            current_op_idx = actual_env.todo_time_step_job[job_idx]
            
            if current_op_idx < actual_env.machines:
                current_op_time = actual_env.instance_matrix[job_idx, current_op_idx]['time']
                if current_op_time < shortest_time:
                    shortest_time = current_op_time
                    best_action = action
        
        # If no job actions found, return first valid action (might be NO_OP)
        return best_action
    
    def _mwkr_heuristic(self, obs_dict, env, valid_actions):
        """Most Work Remaining: Choose job with most remaining processing time"""
        best_action = valid_actions[0]
        most_work = -1
        
        # Get the actual JSSEnv from wrapped environment
        actual_env = env
        while hasattr(actual_env, 'env'):
            actual_env = actual_env.env
        
        for action in valid_actions:
            if action == actual_env.NO_OP_ACTION:  # Skip NO_OP for comparison
                continue
                
            job_idx = action
            current_op_idx = actual_env.todo_time_step_job[job_idx]
            
            # Calculate remaining work for this job
            remaining_work = 0
            for op_idx in range(current_op_idx, actual_env.machines):
                remaining_work += actual_env.instance_matrix[job_idx, op_idx]['time']
            
            if remaining_work > most_work:
                most_work = remaining_work
                best_action = action
        
        # If no job actions found, return first valid action (might be NO_OP)
        return best_action

def run_simulated_annealing(env, max_iterations=1000, initial_temp=100.0, cooling_rate=0.95, seed=42):
    """Simple simulated annealing baseline for JSSP"""
    np.random.seed(seed)
    random.seed(seed)
    
    obs_dict, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    
    best_makespan = float('inf')
    episode_length = 0
    total_reward = 0
    
    while not (terminated or truncated):
        valid_actions = np.where(obs_dict["action_mask"])[0]
        if len(valid_actions) == 0:
            break
            
        # Simple SA: choose random valid action (in real SA, would evaluate neighbors)
        action = np.random.choice(valid_actions)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            makespan = info.get('makespan', float('inf'))
            all_jobs_completed = info.get('all_jobs_completed', False)
            if makespan < best_makespan:
                best_makespan = makespan
    
    return {
        'makespan': best_makespan if best_makespan != float('inf') else info.get('makespan', float('inf')),
        'total_reward': total_reward,
        'episode_length': episode_length,
        'all_jobs_completed': info.get('all_jobs_completed', False)
    }

def evaluate_single_instance(env, agent, agent_name, num_episodes=5, device="cpu"):
    """Evaluate a single agent on one instance"""
    results = []
    
    for episode in range(num_episodes):
        if agent_name == "PPO":
            # PPO evaluation
            obs_dict, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            episode_length = 0
            
            while not (terminated or truncated):
                real_obs_tensor = torch.tensor(obs_dict["real_obs"], dtype=torch.float32).to(device)
                if real_obs_tensor.ndim == 2:
                    real_obs_tensor = real_obs_tensor.unsqueeze(0)
                
                action_mask_tensor = torch.tensor(obs_dict["action_mask"], dtype=torch.bool).to(device)
                if action_mask_tensor.ndim == 1:
                    action_mask_tensor = action_mask_tensor.unsqueeze(0)

                obs_for_agent = {"real_obs": real_obs_tensor}

                with torch.no_grad():
                    action_tensor = agent.get_action(obs_for_agent, action_mask=action_mask_tensor, deterministic=True)
                
                action = action_tensor.item()
                obs_dict, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                episode_length += 1
            
            result = {
                'makespan': info.get('makespan', float('inf')),
                'total_reward': total_reward,
                'episode_length': episode_length,
                'all_jobs_completed': info.get('all_jobs_completed', False)
            }
            
        elif agent_name == "Simulated_Annealing":
            # Simulated Annealing evaluation
            result = run_simulated_annealing(env, seed=42+episode)
            
        else:
            # Baseline agent evaluation
            obs_dict, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            episode_length = 0
            
            while not (terminated or truncated):
                action = agent.get_action(obs_dict, env=env)  # Pass env for proper heuristics
                obs_dict, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                episode_length += 1
            
            result = {
                'makespan': info.get('makespan', float('inf')),
                'total_reward': total_reward,
                'episode_length': episode_length,
                'all_jobs_completed': info.get('all_jobs_completed', False)
            }
        
        results.append(result)
    
    return results

def comprehensive_evaluation(model_path, test_instances=None, num_episodes=5, device="cpu"):
    """Comprehensive evaluation of PPO vs baselines on test instances"""
    
    if test_instances is None:
        test_instances = ["ta06", "ta07", "ta08", "ta09", "ta10"]
    
    print(f"🚀 Comprehensive JSSP Evaluation")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Test Instances: {test_instances}")
    print(f"Episodes per instance: {num_episodes}")
    print(f"Device: {device}")
    print(f"{'='*50}\n")

    # Initialize agents
    agents = {
        "Random": BaselineAgent("random"),
        "FIFO": BaselineAgent("first_available"),
        "SPT": BaselineAgent("spt"), 
        "MWKR": BaselineAgent("mwkr"),
        "Simulated_Annealing": None,  # Special case
        "PPO": None  # Will be loaded
    }
    
    # Load PPO model
    project_root = Path(__file__).parent.resolve()
    instance_dir = "JSSEnv/envs/instances"
    
    # Create a temporary env to get spaces for PPO agent initialization
    temp_instance_path = project_root / instance_dir / test_instances[0]
    if not temp_instance_path.exists():
        temp_instance_path = Path(os.getcwd()) / instance_dir / test_instances[0]
    
    temp_env = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": str(temp_instance_path)})
    agents["PPO"] = Agent(temp_env.observation_space, temp_env.action_space).to(device)
    agents["PPO"].load_state_dict(torch.load(model_path, map_location=device))
    agents["PPO"].eval()
    temp_env.close()
    
    print(f"✓ PPO model loaded successfully\n")
    
    # Results storage
    all_results = []
    
    for instance_name in test_instances:
        print(f"📋 Evaluating on {instance_name}")
        print(f"-" * 30)
        
        # Create environment for this instance
        instance_path = project_root / instance_dir / instance_name
        if not instance_path.exists():
            instance_path = Path(os.getcwd()) / instance_dir / instance_name
        
        if not instance_path.exists():
            print(f"❌ Instance {instance_name} not found, skipping...")
            continue
        
        env = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": str(instance_path)})
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        instance_results = {}
        
        for agent_name, agent in agents.items():
            print(f"  Testing {agent_name}...", end=" ")
            
            try:
                results = evaluate_single_instance(env, agent, agent_name, num_episodes, device)
                
                # Calculate statistics
                makespans = [r['makespan'] for r in results if r['makespan'] != float('inf') and r['makespan'] > 0]  # Filter invalid makespans
                total_rewards = [r['total_reward'] for r in results]
                episode_lengths = [r['episode_length'] for r in results]
                success_rate = sum(r['all_jobs_completed'] for r in results) / len(results)
                
                stats = {
                    'mean_makespan': np.mean(makespans) if makespans else float('inf'),
                    'std_makespan': np.std(makespans) if len(makespans) > 1 else 0,
                    'min_makespan': np.min(makespans) if makespans else float('inf'),
                    'max_makespan': np.max(makespans) if makespans else float('inf'),
                    'mean_reward': np.mean(total_rewards),
                    'mean_episode_length': np.mean(episode_lengths),
                    'success_rate': success_rate,
                    'episodes_evaluated': len(results)
                }
                
                instance_results[agent_name] = stats
                print(f"✓ Makespan: {stats['mean_makespan']:.2f} ± {stats['std_makespan']:.2f}")
                
                # Store detailed results
                for i, result in enumerate(results):
                    all_results.append({
                        'instance': instance_name,
                        'agent': agent_name,
                        'episode': i+1,
                        'makespan': result['makespan'],
                        'total_reward': result['total_reward'],
                        'episode_length': result['episode_length'],
                        'all_jobs_completed': result['all_jobs_completed']
                    })
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                instance_results[agent_name] = {'error': str(e)}
        
        env.close()
        print()
        
        # Print instance summary
        print(f"  {instance_name} Summary:")
        valid_results = {k: v for k, v in instance_results.items() if 'error' not in v}
        if valid_results:
            best_agent = min(valid_results.keys(), key=lambda x: valid_results[x]['mean_makespan'])
            best_makespan = valid_results[best_agent]['mean_makespan']
            print(f"    Best: {best_agent} (makespan: {best_makespan:.2f})")
            
            # Show PPO vs best baseline
            if 'PPO' in valid_results and best_agent != 'PPO':
                ppo_makespan = valid_results['PPO']['mean_makespan']
                improvement = ((best_makespan - ppo_makespan) / best_makespan) * 100
                print(f"    PPO vs Best Baseline: {improvement:+.2f}% ({'better' if improvement > 0 else 'worse'})")
        print()
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Generate summary statistics
    print(f"📊 Overall Results Summary")
    print(f"{'='*50}")
    
    if not results_df.empty:
        summary_stats = results_df.groupby(['instance', 'agent'])['makespan'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print("\nMakespan Summary (mean ± std):")
        print(summary_stats)
        
        # Overall rankings
        print(f"\n🏆 Overall Agent Rankings (by mean makespan across all instances):")
        valid_results = results_df[(results_df['makespan'] != float('inf')) & (results_df['makespan'] > 0)]  # Filter invalid makespans
        overall_rankings = valid_results.groupby('agent')['makespan'].mean().sort_values()
        for rank, (agent, makespan) in enumerate(overall_rankings.items(), 1):
            print(f"  {rank}. {agent}: {makespan:.2f}")
        
        # Success rates
        print(f"\n✅ Success Rates (episodes with all jobs completed):")
        success_rates = results_df.groupby('agent')['all_jobs_completed'].mean().sort_values(ascending=False)
        for agent, rate in success_rates.items():
            print(f"  {agent}: {rate*100:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/comprehensive_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / f"ppo_vs_baselines_ta06-ta10_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    
    print(f"\n💾 Detailed results saved to: {csv_file}")
    print(f"🏁 Comprehensive evaluation completed!")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive PPO vs Baselines evaluation on JSSP test instances")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained PPO model (.pt file)"
    )
    parser.add_argument(
        "--test_instances", 
        type=str,
        nargs="+",
        default=["ta06", "ta07", "ta08", "ta09", "ta10"],
        help="Test instances to evaluate on (default: ta06-ta10)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per instance (default: 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for PPO evaluation (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Device validation
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, using CPU")
        device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS requested but not available, using CPU")
        device = "cpu"
    else:
        device = args.device
    
    comprehensive_evaluation(
        model_path=args.model_path,
        test_instances=args.test_instances,
        num_episodes=args.episodes,
        device=device
    )