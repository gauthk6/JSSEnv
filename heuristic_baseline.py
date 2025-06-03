#!/usr/bin/env python3

"""
Heuristic Baseline Computation for JSSP
Computes performance of common heuristics for comparison with RL methods.
"""

import argparse
import os
import json
import time
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple

import JSSEnv

# Size category mappings (same as eval_model.py)
SIZE_CATEGORIES = {
    "small": ["ta16", "ta17", "ta18", "ta19", "ta20"],          # 15x15
    "medium": ["ta21", "ta22", "ta23", "ta24", "ta25"],         # 20x20  
    "medium_large": ["ta46", "ta47", "ta48", "ta49", "ta50"],   # 20x15
    "large": ["ta56", "ta57", "ta58", "ta59", "ta60"],          # 50x15
    "very_large": ["ta66", "ta67", "ta68", "ta69", "ta70"]      # 50x20
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Make instance and size mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance", type=str,
                       help="Instance name to evaluate (e.g., ta01)")
    group.add_argument("--size", type=str, choices=list(SIZE_CATEGORIES.keys()),
                       help="Size category to evaluate")
    
    parser.add_argument("--heuristics", type=str, nargs="+", 
                       default=["SPT", "LPT", "FIFO", "Random"],
                       help="Heuristics to evaluate")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes per heuristic (for stochastic ones)")
    parser.add_argument("--output-dir", type=str, default="heuristic_results",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max-episode-steps", type=int, default=3000,
                       help="Maximum number of steps per episode")
    return parser.parse_args()

class HeuristicAgent:
    """Agent that uses heuristic policies"""
    
    def __init__(self, heuristic_type: str):
        self.heuristic_type = heuristic_type.upper()
        self.rng = np.random.RandomState(42)
    
    def get_action(self, obs_dict, env) -> int:
        """Get action based on heuristic policy"""
        action_mask = obs_dict["action_mask"]
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            return 0  # NO_OP
        
        # Get the NO_OP action (usually the last one)
        no_op_action = getattr(env, 'NO_OP_ACTION', len(action_mask) - 1)
        
        # Filter out NO_OP action for job selection heuristics
        job_actions = [a for a in valid_actions if a != no_op_action]
        
        if len(job_actions) == 0:
            return no_op_action
        
        if self.heuristic_type == "SPT":
            return self._shortest_processing_time(job_actions, env)
        elif self.heuristic_type == "LPT":
            return self._longest_processing_time(job_actions, env)
        elif self.heuristic_type == "FIFO":
            return self._first_in_first_out(job_actions, env)
        elif self.heuristic_type == "RANDOM":
            return self.rng.choice(valid_actions)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic_type}")
    
    def _shortest_processing_time(self, job_actions: List[int], env) -> int:
        """Select job with shortest processing time for next operation"""
        min_time = float('inf')
        best_action = job_actions[0]
        
        for action in job_actions:
            job_idx = action
            if hasattr(env, 'todo_time_step_job') and hasattr(env, 'instance_matrix'):
                current_op_idx = env.todo_time_step_job[job_idx]
                if current_op_idx < env.machines:
                    processing_time = env.instance_matrix[job_idx, current_op_idx]['time']
                    if processing_time < min_time:
                        min_time = processing_time
                        best_action = action
        
        return best_action
    
    def _longest_processing_time(self, job_actions: List[int], env) -> int:
        """Select job with longest processing time for next operation"""
        max_time = -1
        best_action = job_actions[0]
        
        for action in job_actions:
            job_idx = action
            if hasattr(env, 'todo_time_step_job') and hasattr(env, 'instance_matrix'):
                current_op_idx = env.todo_time_step_job[job_idx]
                if current_op_idx < env.machines:
                    processing_time = env.instance_matrix[job_idx, current_op_idx]['time']
                    if processing_time > max_time:
                        max_time = processing_time
                        best_action = action
        
        return best_action
    
    def _first_in_first_out(self, job_actions: List[int], env) -> int:
        """Select job that arrived first (lowest job index)"""
        return min(job_actions)

def make_env(env_id: str, instance_name: str, seed: int, max_episode_steps=None):
    """Create environment for specific instance"""
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
            # Only print once per instance evaluation
            if seed % 1000 == 42:  # Print for first episode (seed=42, 1042, etc.)
                print(f"   Overriding max_episode_steps: {original_steps} -> {max_episode_steps}")
    
    env.action_space.seed(seed)
    
    return env

def evaluate_heuristic(heuristic_name: str, instance_name: str, episodes: int, seed: int, max_episode_steps=None) -> Dict:
    """Evaluate a single heuristic on an instance"""
    print(f"  Evaluating {heuristic_name} heuristic...")
    
    env_id = "JSSEnv/JssEnv-v1"
    agent = HeuristicAgent(heuristic_name)
    
    episode_results = []
    all_inference_times = []
    
    for episode in range(episodes):
        env = make_env(env_id, instance_name, seed + episode, max_episode_steps)
        
        obs_dict, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        step_count = 0
        done = False
        episode_inference_times = []
        
        start_time = time.time()
        
        while not done:
            # Get underlying environment for heuristic access
            underlying_env = env
            while hasattr(underlying_env, 'env'):
                underlying_env = underlying_env.env
            
            # Time the heuristic inference
            inference_start = time.time()
            action = agent.get_action(obs_dict, underlying_env)
            inference_time = time.time() - inference_start
            episode_inference_times.append(inference_time)
            
            obs_dict, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        episode_time = time.time() - start_time
        all_inference_times.extend(episode_inference_times)
        
        # Get final makespan and success
        makespan = None
        all_jobs_completed = False
        
        if hasattr(underlying_env, 'current_time_step'):
            makespan = underlying_env.current_time_step
        if hasattr(underlying_env, '_are_all_jobs_completed'):
            all_jobs_completed = underlying_env._are_all_jobs_completed()
        
        episode_results.append({
            "episode": episode,
            "reward": episode_reward,
            "steps": step_count,
            "makespan": makespan,
            "success": all_jobs_completed,
            "time": episode_time,
            "avg_inference_time_ms": np.mean(episode_inference_times) * 1000
        })
        
        env.close()
    
    # Compute statistics
    makespans = [r["makespan"] for r in episode_results if r["makespan"] is not None]
    rewards = [r["reward"] for r in episode_results]
    steps = [r["steps"] for r in episode_results]
    successes = [r["success"] for r in episode_results]
    times = [r["time"] for r in episode_results]
    
    results = {
        "heuristic": heuristic_name,
        "instance": instance_name,
        "episodes": episodes,
        "stats": {
            "makespan": {
                "mean": float(np.mean(makespans)) if makespans else None,
                "std": float(np.std(makespans)) if makespans else None,
                "min": float(np.min(makespans)) if makespans else None,
                "max": float(np.max(makespans)) if makespans else None
            },
            "reward": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards))
            },
            "steps": {
                "mean": float(np.mean(steps)),
                "std": float(np.std(steps)),
                "min": int(np.min(steps)),
                "max": int(np.max(steps))
            },
            "success_rate": float(np.mean(successes)),
            "runtime": {
                "mean": float(np.mean(times)),
                "total": float(np.sum(times))
            },
            "inference_time": {
                "mean_ms": float(np.mean(all_inference_times) * 1000),
                "std_ms": float(np.std(all_inference_times) * 1000),
                "min_ms": float(np.min(all_inference_times) * 1000),
                "max_ms": float(np.max(all_inference_times) * 1000),
                "total_actions": len(all_inference_times)
            }
        },
        "episode_results": episode_results
    }
    
    if makespans:
        print(f"    Makespan: {results['stats']['makespan']['mean']:.1f} ± {results['stats']['makespan']['std']:.1f}")
        print(f"    Success rate: {results['stats']['success_rate']*100:.1f}%")
    print(f"    Inference time: {results['stats']['inference_time']['mean_ms']:.3f} ± {results['stats']['inference_time']['std_ms']:.3f} ms/action")
    
    return results

def evaluate_size_category(size_name: str, instances: List[str], heuristics: List[str], 
                          episodes: int, seed: int, max_episode_steps: int) -> Dict:
    """Evaluate all heuristics on a size category"""
    print(f"🧮 Evaluating {size_name.upper()} category heuristics")
    print(f"   Test instances: {', '.join(instances)}")
    print(f"   Heuristics: {', '.join(heuristics)}")
    print(f"   Episodes per instance: {episodes}")
    
    all_results = {}
    category_stats = {}
    
    # Initialize aggregated stats for each heuristic
    for heuristic in heuristics:
        category_stats[heuristic] = {
            "makespans": [],
            "rewards": [],
            "steps": [],
            "successes": [],
            "inference_times": []
        }
    
    # Evaluate each instance
    for i, instance in enumerate(instances):
        print(f"\n   📊 Instance {i+1}/{len(instances)}: {instance}")
        
        instance_results = {}
        for heuristic in heuristics:
            try:
                result = evaluate_heuristic(heuristic, instance, episodes, seed, max_episode_steps)
                instance_results[heuristic] = result
                
                # Aggregate stats for category summary
                if result["stats"]["makespan"]["mean"] is not None:
                    category_stats[heuristic]["makespans"].append(result["stats"]["makespan"]["mean"])
                category_stats[heuristic]["rewards"].append(result["stats"]["reward"]["mean"])
                category_stats[heuristic]["steps"].append(result["stats"]["steps"]["mean"])
                category_stats[heuristic]["successes"].append(result["stats"]["success_rate"])
                category_stats[heuristic]["inference_times"].append(result["stats"]["inference_time"]["mean_ms"])
                
            except Exception as e:
                print(f"     ❌ Failed to evaluate {heuristic}: {e}")
                instance_results[heuristic] = {"status": "failed", "error": str(e)}
        
        all_results[instance] = instance_results
    
    # Compute category-level statistics
    print(f"\n📊 {size_name.upper()} CATEGORY SUMMARY:")
    
    summary_stats = {}
    for heuristic in heuristics:
        stats = category_stats[heuristic]
        
        if stats["makespans"]:
            makespan_mean = np.mean(stats["makespans"])
            makespan_std = np.std(stats["makespans"])
            success_mean = np.mean(stats["successes"]) * 100
            success_std = np.std(stats["successes"]) * 100
            inference_mean = np.mean(stats["inference_times"])
            inference_std = np.std(stats["inference_times"])
            
            summary_stats[heuristic] = {
                "makespan_mean": makespan_mean,
                "makespan_std": makespan_std,
                "success_rate_mean": success_mean,
                "success_rate_std": success_std,
                "inference_time_mean": inference_mean,
                "inference_time_std": inference_std
            }
            
            print(f"   {heuristic:8s}: Makespan={makespan_mean:6.1f}±{makespan_std:4.1f}, "
                  f"Success={success_mean:5.1f}±{success_std:4.1f}%, "
                  f"Inference={inference_mean:5.2f}±{inference_std:4.2f}ms")
    
    return {
        "size_category": size_name,
        "instances": instances,
        "heuristics": heuristics,
        "episodes_per_instance": episodes,
        "instance_results": all_results,
        "category_summary": summary_stats
    }

def main():
    args = parse_args()
    
    print(f"🧮 Computing Heuristic Baselines")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.instance:
        # Single instance evaluation
        print(f"Instance: {args.instance}")
        print(f"Heuristics: {args.heuristics}")
        print(f"Episodes per heuristic: {args.episodes}")
        if args.max_episode_steps:
            print(f"Max episode steps: {args.max_episode_steps}")
        
        all_results = {}
        
        for heuristic in args.heuristics:
            try:
                result = evaluate_heuristic(heuristic, args.instance, args.episodes, args.seed, args.max_episode_steps)
                all_results[heuristic] = result
            except Exception as e:
                print(f"  ❌ Failed to evaluate {heuristic}: {e}")
                all_results[heuristic] = {"status": "failed", "error": str(e)}
        
        # Save results
        output_file = f"{args.output_dir}/heuristic_{args.instance}_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
        
        # Print summary
        print(f"\n📊 Summary for {args.instance}:")
        for heuristic, result in all_results.items():
            if "stats" in result and result["stats"]["makespan"]["mean"] is not None:
                makespan = result["stats"]["makespan"]["mean"]
                success = result["stats"]["success_rate"] * 100
                print(f"  {heuristic:8s}: Makespan={makespan:6.1f}, Success={success:5.1f}%")
    
    else:
        # Size category evaluation
        size_name = args.size
        instances = SIZE_CATEGORIES[size_name]
        
        results = evaluate_size_category(size_name, instances, args.heuristics, 
                                       args.episodes, args.seed, args.max_episode_steps)
        
        # Save results
        output_file = f"{args.output_dir}/heuristic_{size_name}_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
        print("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main() 