#!/usr/bin/env python3

"""
Test script to verify max_episode_steps adjustment works correctly
"""

import os
import gymnasium as gym
import JSSEnv

def calculate_max_steps(jobs, machines):
    """Calculate appropriate max_episode_steps based on problem size"""
    base_steps = jobs * machines * 2
    min_steps = 2000
    max_steps = 15000
    calculated_steps = max(min_steps, min(base_steps, max_steps))
    return int(calculated_steps * 1.2)

def test_instance_step_limits():
    """Test max_episode_steps for different instance sizes"""
    
    # Test instances from different size categories
    test_instances = {
        "ta01": {"jobs": 15, "machines": 15, "expected_makespan": "~1200-1300"},
        "ta21": {"jobs": 20, "machines": 20, "expected_makespan": "~1600-1700"},
        "ta51": {"jobs": 50, "machines": 15, "expected_makespan": "~2700-2800"},
        "ta61": {"jobs": 50, "machines": 20, "expected_makespan": "~2800-3000"},
        "ta71": {"jobs": 100, "machines": 20, "expected_makespan": "~5400-5600"}
    }
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print("🧪 Testing max_episode_steps adjustment for different instance sizes\n")
    
    for instance_name, info in test_instances.items():
        print(f"📐 Testing {instance_name} ({info['jobs']}x{info['machines']})")
        
        # Calculate appropriate step limit
        recommended_steps = calculate_max_steps(info['jobs'], info['machines'])
        print(f"   Recommended max_steps: {recommended_steps:,}")
        
        # Find instance file
        instance_path = os.path.join(project_root, "JSSEnv", "envs", "instances", instance_name)
        if not os.path.exists(instance_path):
            instance_path = os.path.join("JSSEnv", "envs", "instances", instance_name)
            if not os.path.exists(instance_path):
                print(f"   ❌ Instance file not found: {instance_name}")
                continue
        
        try:
            # Create environment with default settings
            env_default = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": instance_path})
            default_steps = getattr(env_default.unwrapped, 'max_episode_steps', 'Not set')
            print(f"   Default max_steps: {default_steps}")
            
            # Create environment with overridden settings
            env_override = gym.make("JSSEnv/JssEnv-v1", env_config={"instance_path": instance_path})
            env_override.unwrapped.max_episode_steps = recommended_steps
            new_steps = env_override.unwrapped.max_episode_steps
            print(f"   Overridden max_steps: {new_steps:,}")
            
            # Quick episode test to see typical episode lengths
            obs, _ = env_override.reset()
            step_count = 0
            done = False
            
            while not done and step_count < 100:  # Just test first 100 steps
                valid_actions = [i for i, valid in enumerate(obs["action_mask"]) if valid]
                if valid_actions:
                    action = valid_actions[0]  # Take first valid action
                    obs, reward, terminated, truncated, info = env_override.step(action)
                    done = terminated or truncated
                    step_count += 1
                else:
                    break
            
            print(f"   First 100 steps completed: {step_count}, Done: {done}")
            print(f"   Expected makespan range: {info['expected_makespan']}")
            
            env_default.close()
            env_override.close()
            
        except Exception as e:
            print(f"   ❌ Error testing {instance_name}: {e}")
        
        print()

def test_scaling_study_step_calculation():
    """Test the step calculation logic from scaling_study.py"""
    print("📊 Testing step calculation for scaling study size categories\n")
    
    from scaling_study import INSTANCE_GROUPS, calculate_max_steps
    
    for size_name, group in INSTANCE_GROUPS.items():
        jobs = group['jobs']
        machines = group['machines']
        max_steps = calculate_max_steps(jobs, machines)
        
        print(f"{size_name:20s} ({group['size']:6s}): {max_steps:5,} steps")
        print(f"   Jobs: {jobs:3d}, Machines: {machines:2d}, Search space: {group['search_space_approx']}")
        print(f"   Step limit ratio: {max_steps/(jobs*machines):4.1f}x problem size")
        print()

if __name__ == "__main__":
    print("🔬 Max Episode Steps Testing\n")
    
    test_scaling_study_step_calculation()
    test_instance_step_limits()
    
    print("✅ Testing completed!") 