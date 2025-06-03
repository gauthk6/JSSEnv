#!/usr/bin/env python3
"""
Test script to check RecordEpisodeStatistics wrapper behavior
"""

import gymnasium as gym
import JSSEnv
import numpy as np

def test_episode_stats():
    print("=== Testing RecordEpisodeStatistics Wrapper ===")
    
    # Create environment with RecordEpisodeStatistics wrapper
    env = gym.make('JSSEnv/JssEnv-v1', env_config={"instance_path": "JSSEnv/envs/instances/ta01"})
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    obs, info = env.reset()
    print(f"Initial info: {info}")
    
    done = False
    truncated = False
    steps = 0
    
    while not (done or truncated):
        legal_actions = np.where(obs['action_mask'] == 1)[0]
        if len(legal_actions) == 0:
            print(f"Step {steps}: No legal actions available!")
            break
            
        action = np.random.choice(legal_actions)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        
        if done or truncated:
            print(f"\nEpisode completed at step {steps}!")
            print(f"Done: {done}, Truncated: {truncated}")
            print(f"Final info structure: {info}")
            print(f"Info type: {type(info)}")
            if isinstance(info, dict):
                print(f"Info keys: {list(info.keys())}")
                for key, value in info.items():
                    print(f"  {key}: {value} (type: {type(value)})")
            break
        
        if steps % 100 == 0:
            print(f"Step {steps}...")
    
    env.close()

if __name__ == "__main__":
    test_episode_stats() 