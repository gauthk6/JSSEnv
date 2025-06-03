#!/usr/bin/env python3

"""
Scaling Study: RL Performance vs Problem Size
Train separate models for different JSSP sizes and evaluate how RL performance 
compares to heuristics as the search space grows.

Study Design:
1. Train separate models on different instance sizes
2. Evaluate each model on test instances of various sizes
3. Compare RL vs heuristic performance gap as size increases
4. Generate comprehensive performance analysis
"""

import argparse
import os
import json
import time
from datetime import datetime
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Instance size categories for the study
INSTANCE_GROUPS = {
    "15x15_small": {
        "train": ["ta01", "ta02", "ta03", "ta04", "ta05"],
        "test": ["ta06", "ta07", "ta08", "ta09", "ta10"],
        "size": "15x15",
        "jobs": 15,
        "machines": 15,
        "search_space_approx": "15^15"
    },
    "20x20_medium": {
        "train": ["ta21", "ta22", "ta23", "ta24", "ta25"], 
        "test": ["ta26", "ta27", "ta28", "ta29", "ta30"],
        "size": "20x20",
        "jobs": 20,
        "machines": 20,
        "search_space_approx": "20^20"
    },
    "20x15_medium_large": {
        "train": ["ta41", "ta42", "ta43", "ta44", "ta45"],
        "test": ["ta46", "ta47", "ta48", "ta49", "ta50"], 
        "size": "20x15",
        "jobs": 20,
        "machines": 15,
        "search_space_approx": "15^20"
    },
    "50x15_large": {
        "train": ["ta51", "ta52", "ta53", "ta54", "ta55"],
        "test": ["ta56", "ta57", "ta58", "ta59", "ta60"],
        "size": "50x15", 
        "jobs": 50,
        "machines": 15,
        "search_space_approx": "15^50"
    },
    "50x20_very_large": {
        "train": ["ta61", "ta62", "ta63", "ta64", "ta65"],
        "test": ["ta66", "ta67", "ta68", "ta69", "ta70"],
        "size": "50x20",
        "jobs": 50, 
        "machines": 20,
        "search_space_approx": "20^50"
    }
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval", "analyze", "full"], 
                       default="full", help="Run mode: train models, evaluate, analyze, or full pipeline")
    parser.add_argument("--study-name", type=str, default="scaling_study",
                       help="Name for this scaling study")
    parser.add_argument("--timesteps-per-size", type=int, default=3000000,
                       help="Training timesteps per size category")
    parser.add_argument("--eval-episodes", type=int, default=50,
                       help="Episodes per instance for evaluation")
    parser.add_argument("--include-heuristics", action="store_true", default=True,
                       help="Include heuristic baselines")
    parser.add_argument("--sizes", type=str, nargs="+", 
                       default=list(INSTANCE_GROUPS.keys()),
                       help="Which size categories to include")
    parser.add_argument("--parallel-jobs", type=int, default=1,
                       help="Number of parallel training jobs")
    return parser.parse_args()

def calculate_max_steps(jobs, machines):
    """
    Use a fixed max_episode_steps that works for all problem sizes.
    3000 steps should be more than enough for any JSSP instance.
    """
    return 3000

class ScalingStudyManager:
    def __init__(self, args):
        self.args = args
        self.study_dir = f"scaling_studies/{args.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.study_dir, exist_ok=True)
        
        # Save study configuration
        self.config = {
            "study_name": args.study_name,
            "timesteps_per_size": args.timesteps_per_size, 
            "eval_episodes": args.eval_episodes,
            "sizes_included": args.sizes,
            "instance_groups": {k: v for k, v in INSTANCE_GROUPS.items() if k in args.sizes},
            "timestamp": datetime.now().isoformat(),
            "max_steps_by_size": self._calculate_max_steps_for_all_sizes()
        }
        
        with open(f"{self.study_dir}/config.json", "w") as f:
            json.dump(self.config, f, indent=2)
            
        print(f"🚀 Scaling Study: {args.study_name}")
        print(f"📁 Study directory: {self.study_dir}")
        print(f"📊 Size categories: {args.sizes}")
        print(f"⏱️  Timesteps per size: {args.timesteps_per_size:,}")
        
        # Print step limits for each size
        print(f"\n📏 Calculated max_episode_steps by size:")
        for size_name in args.sizes:
            if size_name in INSTANCE_GROUPS:
                group = INSTANCE_GROUPS[size_name]
                max_steps = calculate_max_steps(group['jobs'], group['machines'])
                print(f"  {size_name} ({group['size']}): {max_steps:,} steps")
    
    def _calculate_max_steps_for_all_sizes(self):
        """Calculate max steps for all size categories"""
        max_steps_config = {}
        for size_name in self.args.sizes:
            if size_name in INSTANCE_GROUPS:
                group = INSTANCE_GROUPS[size_name]
                max_steps = calculate_max_steps(group['jobs'], group['machines'])
                max_steps_config[size_name] = max_steps
        return max_steps_config
    
    def train_model(self, size_name: str) -> str:
        """Train a model for a specific problem size"""
        print(f"\n🚀 Training model for {size_name}")
        
        group = INSTANCE_GROUPS[size_name]
        train_instances = group['train']
        
        # Calculate appropriate max_episode_steps for this problem size
        max_steps = calculate_max_steps(group['jobs'], group['machines'])
        print(f"📏 Using max_episode_steps: {max_steps:,} for {group['jobs']}x{group['machines']} problems")
        
        model_name = f"ppo_model_{size_name}"
        
        # Prepare training command with size-specific max steps
        cmd = [
            "python", "train_ppo_multi.py",
            "--exp-name", model_name,
            "--total-timesteps", str(self.config['timesteps_per_size']),
            "--num-envs", str(self.config['parallel_jobs']),
            "--num-steps", str(self.config['timesteps_per_size'] // max_steps),
            "--train-instances"
        ] + train_instances + [
            "--max-episode-steps", str(max_steps)  # Add the calculated step limit
        ]
        
        print(f"🎯 Command: {' '.join(cmd)}")
        
        # Execute training
        if not self.config['dry_run']:
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("✅ Training completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Training failed: {e}")
                print(f"Stderr: {e.stderr}")
                return None
        else:
            print("🏃 Dry run - skipping actual training")
        
        # Return model path
        model_path = f"models/{model_name}.pt"
        if os.path.exists(model_path):
            print(f"📦 Model saved: {model_path}")
            return model_path
        else:
            print(f"⚠️  Expected model file not found: {model_path}")
            return None
    
    def train_all_models(self):
        """Train separate models for each size category"""
        print(f"\n🏋️  Training Phase: {len(self.args.sizes)} size categories")
        
        training_results = {}
        
        for size_name in self.args.sizes:
            if size_name not in INSTANCE_GROUPS:
                print(f"⚠️  Unknown size category: {size_name}, skipping")
                continue
                
            group = INSTANCE_GROUPS[size_name]
            max_steps = calculate_max_steps(group['jobs'], group['machines'])
            
            print(f"\n📐 Training model for {size_name} ({group['size']})")
            print(f"   Training instances: {group['train']}")
            print(f"   Max episode steps: {max_steps:,}")
            
            # Create model directory
            model_dir = f"{self.study_dir}/models/{size_name}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Train using the working train_ppo_multi.py with adjusted step limit
            cmd = [
                "python", "train_ppo_multi.py",
                "--exp-name", f"scaling_study_{size_name}",
                "--train-instances"] + group['train'] + [
                "--total-timesteps", str(self.config['timesteps_per_size']),
                "--max-episode-steps", str(max_steps),
                "--track",  # Enable tensorboard logging
                "--seed", "42"
            ]
            
            print(f"   🔧 Command: {' '.join(cmd)}")
            
            start_time = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
                training_time = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"   ✅ Training completed in {training_time:.1f}s")
                    
                    # Find and move the trained model
                    runs_dir = "runs"
                    if os.path.exists(runs_dir):
                        # Find the most recent run for this experiment
                        run_dirs = [d for d in os.listdir(runs_dir) if f"scaling_study_{size_name}" in d]
                        if run_dirs:
                            latest_run = max(run_dirs, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))
                            model_files = [f for f in os.listdir(f"{runs_dir}/{latest_run}") if f.endswith('.pt')]
                            if model_files:
                                model_file = model_files[0]
                                old_path = f"{runs_dir}/{latest_run}/{model_file}"
                                new_path = f"{model_dir}/model.pt"
                                os.rename(old_path, new_path)
                                print(f"   📦 Model saved to {new_path}")
                    
                    training_results[size_name] = {
                        "status": "success",
                        "training_time": training_time,
                        "model_path": f"{model_dir}/model.pt",
                        "max_episode_steps": max_steps
                    }
                else:
                    print(f"   ❌ Training failed: {result.stderr}")
                    training_results[size_name] = {
                        "status": "failed", 
                        "error": result.stderr,
                        "max_episode_steps": max_steps
                    }
            except subprocess.TimeoutExpired:
                print(f"   ⏰ Training timed out after 2 hours")
                training_results[size_name] = {
                    "status": "timeout",
                    "max_episode_steps": max_steps
                }
            except Exception as e:
                print(f"   💥 Training error: {e}")
                training_results[size_name] = {
                    "status": "error",
                    "error": str(e),
                    "max_episode_steps": max_steps
                }
        
        # Save training results
        with open(f"{self.study_dir}/training_results.json", "w") as f:
            json.dump(training_results, f, indent=2)
        
        return training_results
    
    def evaluate_all_models(self):
        """Evaluate each trained model on test instances of all sizes"""
        print(f"\n🧪 Evaluation Phase")
        
        # Check which models are available
        available_models = {}
        for size_name in self.args.sizes:
            model_path = f"{self.study_dir}/models/{size_name}/model.pt"
            if os.path.exists(model_path):
                available_models[size_name] = model_path
            else:
                print(f"⚠️  Model not found for {size_name}: {model_path}")
        
        if not available_models:
            print("❌ No trained models found for evaluation")
            return {}
        
        print(f"🎯 Evaluating {len(available_models)} models on {len(self.args.sizes)} size categories")
        
        evaluation_results = {}
        
        # Evaluate each model on each test set
        for model_size, model_path in available_models.items():
            evaluation_results[model_size] = {}
            
            for test_size in self.args.sizes:
                if test_size not in INSTANCE_GROUPS:
                    continue
                    
                test_group = INSTANCE_GROUPS[test_size]
                test_max_steps = calculate_max_steps(test_group['jobs'], test_group['machines'])
                
                print(f"\n🔍 Evaluating {model_size} model on {test_size} instances")
                print(f"   Max episode steps for evaluation: {test_max_steps:,}")
                
                # Evaluate on each test instance
                instance_results = {}
                for instance in test_group['test']:
                    print(f"   Testing on {instance}...")
                    
                    # Create evaluation command with appropriate step limit
                    eval_cmd = [
                        "python", "eval_model.py",
                        "--model-path", model_path,
                        "--instance", instance,
                        "--episodes", str(self.config['eval_episodes']),
                        "--max-episode-steps", str(test_max_steps),
                        "--output-dir", f"{self.study_dir}/eval_results",
                        "--seed", "42"
                    ]
                    
                    try:
                        # Run evaluation
                        result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
                        
                        if result.returncode == 0:
                            # Parse results from the eval script output or find the JSON file
                            eval_results_dir = f"{self.study_dir}/eval_results"
                            if os.path.exists(eval_results_dir):
                                # Find the most recent eval file for this instance
                                eval_files = [f for f in os.listdir(eval_results_dir) 
                                            if f.startswith(f"eval_{instance}_") and f.endswith('.json')]
                                if eval_files:
                                    latest_eval_file = max(eval_files, 
                                                         key=lambda x: os.path.getctime(os.path.join(eval_results_dir, x)))
                                    
                                    with open(os.path.join(eval_results_dir, latest_eval_file), 'r') as f:
                                        eval_data = json.load(f)
                                    
                                    instance_results[instance] = {
                                        "mean_makespan": eval_data['stats']['makespans']['mean'] if 'makespans' in eval_data['stats'] else None,
                                        "std_makespan": eval_data['stats']['makespans']['std'] if 'makespans' in eval_data['stats'] else None,
                                        "success_rate": eval_data['stats']['success_rate'] if 'success_rate' in eval_data['stats'] else None,
                                        "mean_return": eval_data['stats']['returns']['mean'],
                                        "std_return": eval_data['stats']['returns']['std'],
                                        "episodes": eval_data['episodes'],
                                        "evaluation_time": eval_data['evaluation_time'],
                                        "mean_inference_time_ms": eval_data['stats']['inference_time']['mean_ms'] if 'inference_time' in eval_data['stats'] else None,
                                        "std_inference_time_ms": eval_data['stats']['inference_time']['std_ms'] if 'inference_time' in eval_data['stats'] else None,
                                        "max_episode_steps": test_max_steps
                                    }
                                    print(f"     ✅ Makespan: {instance_results[instance]['mean_makespan']:.1f}, "
                                          f"Inference: {instance_results[instance]['mean_inference_time_ms']:.2f}ms, "
                                          f"Steps: {test_max_steps:,}")
                                else:
                                    print(f"     ⚠️  No evaluation file found")
                                    instance_results[instance] = {"status": "no_file", "max_episode_steps": test_max_steps}
                            else:
                                print(f"     ⚠️  Eval results directory not found")
                                instance_results[instance] = {"status": "no_dir", "max_episode_steps": test_max_steps}
                        else:
                            print(f"     ❌ Evaluation failed: {result.stderr}")
                            instance_results[instance] = {"status": "failed", "error": result.stderr, "max_episode_steps": test_max_steps}
                    
                    except subprocess.TimeoutExpired:
                        print(f"     ⏰ Evaluation timed out")
                        instance_results[instance] = {"status": "timeout", "max_episode_steps": test_max_steps}
                    except Exception as e:
                        print(f"     💥 Evaluation error: {e}")
                        instance_results[instance] = {"status": "error", "error": str(e), "max_episode_steps": test_max_steps}
                
                evaluation_results[model_size][test_size] = instance_results
        
        # Save evaluation results
        with open(f"{self.study_dir}/evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    def compute_heuristic_baselines(self):
        """Compute heuristic baselines for comparison"""
        print(f"\n🧮 Computing Heuristic Baselines")
        
        heuristic_results = {}
        heuristics = ["SPT", "LPT", "FIFO", "Random"]
        
        for size_name in self.args.sizes:
            if size_name not in INSTANCE_GROUPS:
                continue
                
            group = INSTANCE_GROUPS[size_name]
            max_steps = calculate_max_steps(group['jobs'], group['machines'])
            heuristic_results[size_name] = {}
            
            print(f"\n📐 Computing baselines for {size_name} ({group['size']})")
            print(f"   Max episode steps: {max_steps:,}")
            
            for instance in group['test']:
                print(f"   Computing baselines for {instance}...")
                
                # Run heuristic baseline script with appropriate step limit
                baseline_cmd = [
                    "python", "heuristic_baseline.py",
                    "--instance", instance,
                    "--heuristics"] + heuristics + [
                    "--episodes", str(self.config['eval_episodes']),
                    "--max-episode-steps", str(max_steps),
                    "--output-dir", f"{self.study_dir}/heuristic_results",
                    "--seed", "42"
                ]
                
                try:
                    result = subprocess.run(baseline_cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
                    
                    if result.returncode == 0:
                        # Parse results from the baseline script output
                        baseline_results_dir = f"{self.study_dir}/heuristic_results"
                        if os.path.exists(baseline_results_dir):
                            # Find the most recent baseline file for this instance
                            baseline_files = [f for f in os.listdir(baseline_results_dir) 
                                            if f.startswith(f"heuristic_{instance}_") and f.endswith('.json')]
                            if baseline_files:
                                latest_baseline_file = max(baseline_files, 
                                                         key=lambda x: os.path.getctime(os.path.join(baseline_results_dir, x)))
                                
                                with open(os.path.join(baseline_results_dir, latest_baseline_file), 'r') as f:
                                    baseline_data = json.load(f)
                                
                                # Extract results for each heuristic
                                for heuristic in heuristics:
                                    if heuristic in baseline_data and "stats" in baseline_data[heuristic]:
                                        if size_name not in heuristic_results:
                                            heuristic_results[size_name] = {}
                                        if heuristic not in heuristic_results[size_name]:
                                            heuristic_results[size_name][heuristic] = {}
                                        
                                        stats = baseline_data[heuristic]["stats"]
                                        heuristic_results[size_name][heuristic][instance] = {
                                            "makespan": stats["makespan"]["mean"] if stats["makespan"]["mean"] is not None else 9999,
                                            "makespan_std": stats["makespan"]["std"] if stats["makespan"]["std"] is not None else 0,
                                            "success_rate": stats["success_rate"],
                                            "inference_time_ms": stats["inference_time"]["mean_ms"] if "inference_time" in stats else 0,
                                            "runtime": stats["runtime"]["mean"],
                                            "max_episode_steps": max_steps
                                        }
                                
                                # Print summary for this instance
                                spt_makespan = heuristic_results[size_name]["SPT"][instance]["makespan"] if "SPT" in heuristic_results[size_name] else None
                                spt_inference = heuristic_results[size_name]["SPT"][instance]["inference_time_ms"] if "SPT" in heuristic_results[size_name] else None
                                print(f"     ✅ SPT Makespan: {spt_makespan:.1f}, Inference: {spt_inference:.3f}ms, Steps: {max_steps:,}")
                            else:
                                print(f"     ⚠️  No baseline file found")
                        else:
                            print(f"     ❌ Baseline computation failed: {result.stderr}")
                
                except subprocess.TimeoutExpired:
                    print(f"     ⏰ Baseline computation timed out")
                except Exception as e:
                    print(f"     💥 Baseline computation error: {e}")
        
        # Save heuristic results
        with open(f"{self.study_dir}/heuristic_results.json", "w") as f:
            json.dump(heuristic_results, f, indent=2)
        
        return heuristic_results
    
    def analyze_results(self):
        """Perform comprehensive analysis of scaling study results"""
        print(f"\n📊 Analysis Phase")
        
        # Load results
        try:
            with open(f"{self.study_dir}/evaluation_results.json", "r") as f:
                eval_results = json.load(f)
        except FileNotFoundError:
            print("❌ Evaluation results not found")
            return
        
        if self.args.include_heuristics:
            try:
                with open(f"{self.study_dir}/heuristic_results.json", "r") as f:
                    heuristic_results = json.load(f)
            except FileNotFoundError:
                print("⚠️  Heuristic results not found, computing...")
                heuristic_results = self.compute_heuristic_baselines()
        else:
            heuristic_results = {}
        
        # Create analysis dataframe
        analysis_data = []
        
        for model_size in eval_results:
            for test_size in eval_results[model_size]:
                group = INSTANCE_GROUPS[test_size]
                
                # Get RL results
                rl_makespans = []
                rl_inference_times = []
                rl_success_rates = []
                for instance, result in eval_results[model_size][test_size].items():
                    if isinstance(result, dict) and 'mean_makespan' in result:
                        rl_makespans.append(result['mean_makespan'])
                        if result['mean_inference_time_ms'] is not None:
                            rl_inference_times.append(result['mean_inference_time_ms'])
                        if result['success_rate'] is not None:
                            rl_success_rates.append(result['success_rate'])
                
                rl_mean_makespan = np.mean(rl_makespans) if rl_makespans else None
                rl_mean_inference = np.mean(rl_inference_times) if rl_inference_times else None
                rl_mean_success = np.mean(rl_success_rates) if rl_success_rates else None
                
                # Get heuristic baselines
                spt_mean_makespan = None
                spt_mean_inference = None
                spt_mean_success = None
                if test_size in heuristic_results and "SPT" in heuristic_results[test_size]:
                    spt_makespans = [heuristic_results[test_size]["SPT"][inst]["makespan"] 
                                   for inst in group['test'] 
                                   if inst in heuristic_results[test_size]["SPT"]]
                    spt_inference_times = [heuristic_results[test_size]["SPT"][inst]["inference_time_ms"] 
                                         for inst in group['test'] 
                                         if inst in heuristic_results[test_size]["SPT"]]
                    spt_success_rates = [heuristic_results[test_size]["SPT"][inst]["success_rate"] 
                                       for inst in group['test'] 
                                       if inst in heuristic_results[test_size]["SPT"]]
                    spt_mean_makespan = np.mean(spt_makespans) if spt_makespans else None
                    spt_mean_inference = np.mean(spt_inference_times) if spt_inference_times else None
                    spt_mean_success = np.mean(spt_success_rates) if spt_success_rates else None
                
                analysis_data.append({
                    "model_size": model_size,
                    "test_size": test_size,
                    "jobs": group['jobs'],
                    "machines": group['machines'],
                    "search_space_factor": group['jobs'] * group['machines'],
                    "rl_makespan": rl_mean_makespan,
                    "spt_makespan": spt_mean_makespan,
                    "rl_vs_spt_ratio": rl_mean_makespan / spt_mean_makespan if (rl_mean_makespan and spt_mean_makespan) else None,
                    "rl_improvement": (spt_mean_makespan - rl_mean_makespan) / spt_mean_makespan * 100 if (rl_mean_makespan and spt_mean_makespan) else None,
                    "rl_inference_time_ms": rl_mean_inference,
                    "spt_inference_time_ms": spt_mean_inference,
                    "inference_speedup": spt_mean_inference / rl_mean_inference if (rl_mean_inference and spt_mean_inference) else None,
                    "inference_slowdown": rl_mean_inference / spt_mean_inference if (rl_mean_inference and spt_mean_inference) else None,
                    "rl_success_rate": rl_mean_success,
                    "spt_success_rate": spt_mean_success,
                    "success_rate_diff": rl_mean_success - spt_mean_success if (rl_mean_success is not None and spt_mean_success is not None) else None
                })
        
        df = pd.DataFrame(analysis_data)
        
        # Save analysis dataframe
        df.to_csv(f"{self.study_dir}/analysis_results.csv", index=False)
        
        # Generate plots
        self.generate_plots(df)
        
        # Generate summary report
        self.generate_report(df)
        
        return df
    
    def generate_plots(self, df):
        """Generate visualization plots for the scaling study"""
        plots_dir = f"{self.study_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # Main performance plot - now with 3x3 grid for more metrics
        plt.figure(figsize=(18, 15))
        
        # Filter to same-size training/testing for cleaner comparison
        same_size_df = df[df['model_size'] == df['test_size']].copy()
        
        # Plot 1: RL vs Heuristic Performance by Problem Size
        if not same_size_df.empty and 'rl_vs_spt_ratio' in same_size_df.columns:
            plt.subplot(3, 3, 1)
            plt.plot(same_size_df['search_space_factor'], same_size_df['rl_vs_spt_ratio'], 
                    'o-', linewidth=2, markersize=8, label='RL vs SPT Ratio')
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
            plt.xlabel('Search Space Factor (Jobs × Machines)')
            plt.ylabel('RL/SPT Makespan Ratio')
            plt.title('RL Performance vs Problem Size\n(Lower is Better)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 2: Improvement Percentage
        if not same_size_df.empty and 'rl_improvement' in same_size_df.columns:
            plt.subplot(3, 3, 2) 
            plt.plot(same_size_df['search_space_factor'], same_size_df['rl_improvement'],
                    's-', linewidth=2, markersize=8, color='green', label='RL Improvement %')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
            plt.xlabel('Search Space Factor (Jobs × Machines)')
            plt.ylabel('Improvement over SPT (%)')
            plt.title('RL Improvement vs Problem Size\n(Higher is Better)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 3: Success Rate Comparison
        if not same_size_df.empty and 'rl_success_rate' in same_size_df.columns:
            plt.subplot(3, 3, 3)
            x_pos = np.arange(len(same_size_df))
            width = 0.35
            
            rl_success = (same_size_df['rl_success_rate'].fillna(0) * 100)
            spt_success = (same_size_df['spt_success_rate'].fillna(0) * 100)
            
            plt.bar(x_pos - width/2, rl_success, width, label='RL', alpha=0.8)
            plt.bar(x_pos + width/2, spt_success, width, label='SPT', alpha=0.8)
            
            plt.xlabel('Problem Size')
            plt.ylabel('Success Rate (%)')
            plt.title('Success Rate Comparison\n(Higher is Better)')
            plt.xticks(x_pos, same_size_df['test_size'].str.replace('_', '\n'), rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 105)
        
        # Plot 4: Inference Time Comparison
        if not same_size_df.empty and 'rl_inference_time_ms' in same_size_df.columns:
            plt.subplot(3, 3, 4)
            x_pos = np.arange(len(same_size_df))
            width = 0.35
            
            rl_times = same_size_df['rl_inference_time_ms'].fillna(0)
            spt_times = same_size_df['spt_inference_time_ms'].fillna(0)
            
            plt.bar(x_pos - width/2, rl_times, width, label='RL', alpha=0.8)
            plt.bar(x_pos + width/2, spt_times, width, label='SPT', alpha=0.8)
            
            plt.xlabel('Problem Size')
            plt.ylabel('Inference Time (ms)')
            plt.title('Inference Time Comparison\n(Lower is Better)')
            plt.xticks(x_pos, same_size_df['test_size'].str.replace('_', '\n'), rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale since RL will likely be much slower
        
        # Plot 5: Performance vs Speed Trade-off
        if not same_size_df.empty and 'rl_inference_time_ms' in same_size_df.columns and 'rl_improvement' in same_size_df.columns:
            plt.subplot(3, 3, 5)
            valid_data = same_size_df.dropna(subset=['rl_inference_time_ms', 'rl_improvement'])
            if not valid_data.empty:
                scatter = plt.scatter(valid_data['rl_inference_time_ms'], valid_data['rl_improvement'], 
                                    s=100, alpha=0.7, c=valid_data['search_space_factor'], cmap='viridis')
                plt.colorbar(scatter, label='Search Space Factor')
                
                # Add problem size labels
                for i, row in valid_data.iterrows():
                    plt.annotate(row['test_size'].replace('_', '\n'), 
                               (row['rl_inference_time_ms'], row['rl_improvement']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.xlabel('RL Inference Time (ms)')
                plt.ylabel('RL Improvement over SPT (%)')
                plt.title('Performance vs Speed Trade-off')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                plt.grid(True, alpha=0.3)
        
        # Plot 6: Success Rate vs Problem Size
        if not same_size_df.empty and 'rl_success_rate' in same_size_df.columns:
            plt.subplot(3, 3, 6)
            valid_data = same_size_df.dropna(subset=['rl_success_rate', 'spt_success_rate'])
            if not valid_data.empty:
                plt.plot(valid_data['search_space_factor'], valid_data['rl_success_rate'] * 100, 
                        'o-', linewidth=2, markersize=8, label='RL', color='blue')
                plt.plot(valid_data['search_space_factor'], valid_data['spt_success_rate'] * 100, 
                        's-', linewidth=2, markersize=8, label='SPT', color='orange')
                
                plt.xlabel('Search Space Factor')
                plt.ylabel('Success Rate (%)')
                plt.title('Success Rate vs Problem Size')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 105)
        
        # Plot 7: Cross-size Generalization Heatmap
        if len(df) > 0:
            plt.subplot(3, 3, 7)
            pivot_df = df.pivot(index='model_size', columns='test_size', values='rl_makespan')
            sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='RdYlBu_r', 
                       cbar_kws={'label': 'Mean Makespan'})
            plt.title('Cross-Size Generalization\n(Model Size vs Test Size)')
            plt.ylabel('Trained Model Size')
            plt.xlabel('Test Problem Size')
        
        # Plot 8: Success Rate Heatmap
        if len(df) > 0 and 'rl_success_rate' in df.columns:
            plt.subplot(3, 3, 8)
            pivot_success_df = df.pivot(index='model_size', columns='test_size', values='rl_success_rate')
            sns.heatmap(pivot_success_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                       cbar_kws={'label': 'Success Rate'}, vmin=0, vmax=1)
            plt.title('Success Rate Cross-Size\n(Model Size vs Test Size)')
            plt.ylabel('Trained Model Size')
            plt.xlabel('Test Problem Size')
        
        # Plot 9: Inference Speed Scaling
        if not same_size_df.empty and 'inference_slowdown' in same_size_df.columns:
            plt.subplot(3, 3, 9)
            valid_slowdown = same_size_df.dropna(subset=['inference_slowdown'])
            if not valid_slowdown.empty:
                plt.plot(valid_slowdown['search_space_factor'], valid_slowdown['inference_slowdown'],
                        '^-', linewidth=2, markersize=8, color='orange', label='RL Slowdown Factor')
                plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Speed')
                plt.xlabel('Search Space Factor (Jobs × Machines)')
                plt.ylabel('RL/SPT Inference Time Ratio')
                plt.title('Inference Speed Scaling\n(Lower is Better)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/scaling_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plots saved to {plots_dir}/")
        
        # Generate additional detailed analysis plots
        if not same_size_df.empty:
            self._plot_inference_analysis(same_size_df, plots_dir)
            self._plot_success_analysis(same_size_df, plots_dir)
    
    def _plot_inference_analysis(self, df, plots_dir):
        """Generate detailed inference time analysis plot"""
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Absolute inference times
        plt.subplot(2, 2, 1)
        if 'rl_inference_time_ms' in df.columns and 'spt_inference_time_ms' in df.columns:
            x_labels = df['test_size'].str.replace('_', '\n')
            x_pos = np.arange(len(df))
            
            plt.semilogy(x_pos, df['rl_inference_time_ms'].fillna(1), 'o-', label='RL', linewidth=2, markersize=8)
            plt.semilogy(x_pos, df['spt_inference_time_ms'].fillna(0.001), 's-', label='SPT', linewidth=2, markersize=8)
            
            plt.xticks(x_pos, x_labels)
            plt.ylabel('Inference Time (ms, log scale)')
            plt.title('Absolute Inference Times')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Subplot 2: Inference slowdown factor
        plt.subplot(2, 2, 2)
        if 'inference_slowdown' in df.columns:
            valid_data = df.dropna(subset=['inference_slowdown'])
            if not valid_data.empty:
                plt.bar(range(len(valid_data)), valid_data['inference_slowdown'], alpha=0.7)
                plt.xticks(range(len(valid_data)), valid_data['test_size'].str.replace('_', '\n'), rotation=45)
                plt.ylabel('Slowdown Factor (RL/SPT)')
                plt.title('RL Inference Slowdown')
                plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
                plt.grid(True, alpha=0.3)
        
        # Subplot 3: Performance per unit time
        plt.subplot(2, 2, 3)
        if 'rl_improvement' in df.columns and 'rl_inference_time_ms' in df.columns:
            valid_data = df.dropna(subset=['rl_improvement', 'rl_inference_time_ms'])
            if not valid_data.empty:
                efficiency = valid_data['rl_improvement'] / valid_data['rl_inference_time_ms']
                plt.bar(range(len(valid_data)), efficiency, alpha=0.7, color='green')
                plt.xticks(range(len(valid_data)), valid_data['test_size'].str.replace('_', '\n'), rotation=45)
                plt.ylabel('Improvement % per ms')
                plt.title('Performance Efficiency\n(Improvement/Inference Time)')
                plt.grid(True, alpha=0.3)
        
        # Subplot 4: Size vs inference time scaling
        plt.subplot(2, 2, 4)
        if 'search_space_factor' in df.columns and 'rl_inference_time_ms' in df.columns:
            valid_data = df.dropna(subset=['search_space_factor', 'rl_inference_time_ms'])
            if not valid_data.empty:
                plt.loglog(valid_data['search_space_factor'], valid_data['rl_inference_time_ms'], 
                          'o-', linewidth=2, markersize=8, label='RL Inference Time')
                
                # Add trend line
                if len(valid_data) > 1:
                    z = np.polyfit(np.log(valid_data['search_space_factor']), 
                                 np.log(valid_data['rl_inference_time_ms']), 1)
                    trend_line = np.exp(z[1]) * valid_data['search_space_factor'] ** z[0]
                    plt.loglog(valid_data['search_space_factor'], trend_line, 
                             '--', alpha=0.7, label=f'Trend (slope={z[0]:.2f})')
                
                plt.xlabel('Search Space Factor')
                plt.ylabel('Inference Time (ms)')
                plt.title('Inference Time Scaling')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/inference_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"⚡ Inference analysis plots saved to {plots_dir}/inference_analysis.png")
    
    def _plot_success_analysis(self, df, plots_dir):
        """Generate detailed success rate analysis plot"""
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Success rate by problem size
        plt.subplot(2, 2, 1)
        if 'rl_success_rate' in df.columns and 'spt_success_rate' in df.columns:
            x_labels = df['test_size'].str.replace('_', '\n')
            x_pos = np.arange(len(df))
            
            plt.plot(x_pos, df['rl_success_rate'].fillna(0) * 100, 'o-', label='RL', linewidth=2, markersize=8)
            plt.plot(x_pos, df['spt_success_rate'].fillna(0) * 100, 's-', label='SPT', linewidth=2, markersize=8)
            
            plt.xticks(x_pos, x_labels, rotation=45)
            plt.ylabel('Success Rate (%)')
            plt.title('Success Rate by Problem Size')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 105)
        
        # Subplot 2: Success rate difference
        plt.subplot(2, 2, 2)
        if 'success_rate_diff' in df.columns:
            valid_data = df.dropna(subset=['success_rate_diff'])
            if not valid_data.empty:
                colors = ['green' if x >= 0 else 'red' for x in valid_data['success_rate_diff']]
                plt.bar(range(len(valid_data)), valid_data['success_rate_diff'] * 100, 
                       alpha=0.7, color=colors)
                plt.xticks(range(len(valid_data)), valid_data['test_size'].str.replace('_', '\n'), rotation=45)
                plt.ylabel('Success Rate Difference (%)')
                plt.title('RL vs SPT Success Rate Difference')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.7)
                plt.grid(True, alpha=0.3)
        
        # Subplot 3: Success rate vs performance correlation
        plt.subplot(2, 2, 3)
        if 'rl_success_rate' in df.columns and 'rl_improvement' in df.columns:
            valid_data = df.dropna(subset=['rl_success_rate', 'rl_improvement'])
            if not valid_data.empty:
                plt.scatter(valid_data['rl_success_rate'] * 100, valid_data['rl_improvement'], 
                           s=100, alpha=0.7, c=valid_data['search_space_factor'], cmap='viridis')
                plt.colorbar(label='Search Space Factor')
                
                # Add correlation coefficient
                if len(valid_data) > 1:
                    corr = np.corrcoef(valid_data['rl_success_rate'], valid_data['rl_improvement'])[0,1]
                    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.xlabel('RL Success Rate (%)')
                plt.ylabel('RL Improvement over SPT (%)')
                plt.title('Success Rate vs Performance')
                plt.grid(True, alpha=0.3)
        
        # Subplot 4: Success rate scaling
        plt.subplot(2, 2, 4)
        if 'search_space_factor' in df.columns and 'rl_success_rate' in df.columns:
            valid_data = df.dropna(subset=['search_space_factor', 'rl_success_rate', 'spt_success_rate'])
            if not valid_data.empty:
                plt.semilogx(valid_data['search_space_factor'], valid_data['rl_success_rate'] * 100, 
                           'o-', linewidth=2, markersize=8, label='RL')
                plt.semilogx(valid_data['search_space_factor'], valid_data['spt_success_rate'] * 100, 
                           's-', linewidth=2, markersize=8, label='SPT')
                
                plt.xlabel('Search Space Factor (log scale)')
                plt.ylabel('Success Rate (%)')
                plt.title('Success Rate Scaling')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/success_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Success rate analysis plots saved to {plots_dir}/success_analysis.png")
    
    def generate_report(self, df):
        """Generate a comprehensive text report"""
        report_path = f"{self.study_dir}/scaling_study_report.md"
        
        with open(report_path, "w") as f:
            f.write(f"# Scaling Study Report: {self.args.study_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Study Configuration\n")
            f.write(f"- **Training timesteps per size:** {self.args.timesteps_per_size:,}\n")
            f.write(f"- **Evaluation episodes:** {self.args.eval_episodes}\n")
            f.write(f"- **Size categories:** {', '.join(self.args.sizes)}\n\n")
            
            f.write("## Problem Sizes Tested\n")
            for size_name in self.args.sizes:
                if size_name in INSTANCE_GROUPS:
                    group = INSTANCE_GROUPS[size_name]
                    f.write(f"- **{size_name}:** {group['size']} ({group['jobs']} jobs, {group['machines']} machines)\n")
            f.write("\n")
            
            # Key findings
            same_size_df = df[df['model_size'] == df['test_size']].copy()
            if not same_size_df.empty:
                f.write("## Key Findings\n")
                
                # Performance findings
                if 'rl_improvement' in same_size_df.columns:
                    best_improvement = same_size_df.loc[same_size_df['rl_improvement'].idxmax()]
                    worst_improvement = same_size_df.loc[same_size_df['rl_improvement'].idxmin()]
                    
                    f.write(f"### Performance Analysis\n")
                    f.write(f"- **Best RL performance:** {best_improvement['test_size']} ")
                    f.write(f"({best_improvement['rl_improvement']:.1f}% improvement over SPT)\n")
                    f.write(f"- **Worst RL performance:** {worst_improvement['test_size']} ")
                    f.write(f"({worst_improvement['rl_improvement']:.1f}% vs SPT)\n")
                
                # Success rate findings
                if 'rl_success_rate' in same_size_df.columns:
                    f.write(f"\n### Success Rate Analysis\n")
                    valid_success = same_size_df.dropna(subset=['rl_success_rate', 'spt_success_rate'])
                    if not valid_success.empty:
                        best_rl_success = valid_success.loc[valid_success['rl_success_rate'].idxmax()]
                        worst_rl_success = valid_success.loc[valid_success['rl_success_rate'].idxmin()]
                        
                        f.write(f"- **Best RL success rate:** {best_rl_success['test_size']} ")
                        f.write(f"({best_rl_success['rl_success_rate']*100:.1f}%)\n")
                        f.write(f"- **Worst RL success rate:** {worst_rl_success['test_size']} ")
                        f.write(f"({worst_rl_success['rl_success_rate']*100:.1f}%)\n")
                        
                        # Average success rates
                        avg_rl_success = valid_success['rl_success_rate'].mean() * 100
                        avg_spt_success = valid_success['spt_success_rate'].mean() * 100
                        f.write(f"- **Average RL success rate:** {avg_rl_success:.1f}%\n")
                        f.write(f"- **Average SPT success rate:** {avg_spt_success:.1f}%\n")
                        
                        # Success rate correlation
                        if len(valid_success) > 1:
                            success_size_corr = np.corrcoef(valid_success['search_space_factor'], 
                                                          valid_success['rl_success_rate'])[0,1]
                            f.write(f"- **Success rate vs size correlation:** {success_size_corr:.3f}\n")
                
                # Inference time findings
                if 'rl_inference_time_ms' in same_size_df.columns:
                    f.write(f"\n### Inference Time Analysis\n")
                    valid_inference = same_size_df.dropna(subset=['rl_inference_time_ms', 'spt_inference_time_ms'])
                    if not valid_inference.empty:
                        fastest_rl = valid_inference.loc[valid_inference['rl_inference_time_ms'].idxmin()]
                        slowest_rl = valid_inference.loc[valid_inference['rl_inference_time_ms'].idxmax()]
                        
                        f.write(f"- **Fastest RL inference:** {fastest_rl['test_size']} ")
                        f.write(f"({fastest_rl['rl_inference_time_ms']:.2f}ms)\n")
                        f.write(f"- **Slowest RL inference:** {slowest_rl['test_size']} ")
                        f.write(f"({slowest_rl['rl_inference_time_ms']:.2f}ms)\n")
                        
                        # Average speedup/slowdown
                        avg_slowdown = valid_inference['inference_slowdown'].mean()
                        f.write(f"- **Average RL slowdown factor:** {avg_slowdown:.1f}x vs SPT\n")
                
                # Trend analysis
                if len(same_size_df) > 1:
                    f.write(f"\n### Scaling Trends\n")
                    
                    # Performance scaling
                    if 'rl_improvement' in same_size_df.columns:
                        correlation = np.corrcoef(same_size_df['search_space_factor'], 
                                                same_size_df['rl_improvement'])[0,1]
                        f.write(f"- **Performance scaling:** Correlation between problem size and RL improvement: {correlation:.3f}\n")
                        
                        if correlation > 0.3:
                            f.write("  - **Positive trend:** RL gets relatively better with larger problems ✅\n")
                        elif correlation < -0.3:
                            f.write("  - **Negative trend:** RL gets relatively worse with larger problems ❌\n")
                        else:
                            f.write("  - **No clear trend:** RL performance doesn't strongly correlate with size 🤷\n")
                    
                    # Success rate scaling
                    if 'rl_success_rate' in same_size_df.columns:
                        valid_success_trend = same_size_df.dropna(subset=['rl_success_rate'])
                        if len(valid_success_trend) > 1:
                            success_correlation = np.corrcoef(valid_success_trend['search_space_factor'], 
                                                            valid_success_trend['rl_success_rate'])[0,1]
                            f.write(f"- **Success rate scaling:** Correlation between problem size and success rate: {success_correlation:.3f}\n")
                            
                            if success_correlation < -0.3:
                                f.write("  - **Degrading trend:** Success rate decreases with larger problems ⚠️\n")
                            elif success_correlation > 0.3:
                                f.write("  - **Improving trend:** Success rate increases with larger problems 📈\n")
                            else:
                                f.write("  - **Stable trend:** Success rate remains consistent across sizes ✅\n")
                    
                    # Inference time scaling
                    if 'rl_inference_time_ms' in same_size_df.columns:
                        valid_inference_trend = same_size_df.dropna(subset=['rl_inference_time_ms'])
                        if len(valid_inference_trend) > 1:
                            inference_correlation = np.corrcoef(valid_inference_trend['search_space_factor'], 
                                                              valid_inference_trend['rl_inference_time_ms'])[0,1]
                            f.write(f"- **Inference time scaling:** Correlation between problem size and inference time: {inference_correlation:.3f}\n")
                            
                            if inference_correlation > 0.7:
                                f.write("  - **Strong scaling:** Inference time increases significantly with size ⚠️\n")
                            elif inference_correlation > 0.3:
                                f.write("  - **Moderate scaling:** Inference time increases with size 📊\n")
                            else:
                                f.write("  - **Weak scaling:** Inference time relatively constant across sizes ✅\n")
            
            f.write("\n## Detailed Results\n")
            f.write(df.to_string(index=False))
            
        print(f"📋 Report saved to {report_path}")
    
    def run_full_study(self):
        """Run the complete scaling study pipeline"""
        print(f"\n🎯 Running Full Scaling Study Pipeline")
        
        if self.args.mode in ["train", "full"]:
            training_results = self.train_all_models()
        
        if self.args.mode in ["eval", "full"]:
            evaluation_results = self.evaluate_all_models()
            
            if self.args.include_heuristics:
                heuristic_results = self.compute_heuristic_baselines()
        
        if self.args.mode in ["analyze", "full"]:
            analysis_df = self.analyze_results()
        
        print(f"\n✅ Scaling study complete!")
        print(f"📁 All results saved to: {self.study_dir}")

if __name__ == "__main__":
    args = parse_args()
    
    study_manager = ScalingStudyManager(args)
    
    if args.mode == "train":
        study_manager.train_all_models()
    elif args.mode == "eval":
        study_manager.evaluate_all_models()
    elif args.mode == "analyze":
        study_manager.analyze_results()
    else:  # full
        study_manager.run_full_study() 