#!/usr/bin/env python3

"""
Scaling Analysis for JSSP
Compare PPO performance against heuristic baselines across different problem sizes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def create_scaling_analysis():
    """Create comprehensive scaling analysis comparing PPO vs heuristics"""
    
    # Data collected from evaluations - ORDERED BY INCREASING MAKESPAN
    # Problem sizes with their dimensions and complexity measure (num_jobs * num_machines)
    problem_sizes = {
        "Small": {"dimensions": "15×15", "complexity": 225},
        "Medium": {"dimensions": "20×20", "complexity": 400},
        "Medium-Large": {"dimensions": "20×15", "complexity": 300},
        "Large": {"dimensions": "50×15", "complexity": 750},
        "Very Large": {"dimensions": "50×20", "complexity": 1000}
    }
    
    # PPO Results (from latest evaluations - updated with new training results)
    ppo_results = {
        "Small": {"makespan": 1730.8, "std": 54.0},           # 15x15 - from earlier training
        "Medium": {"makespan": 2421.3, "std": 106.5},        # 20x20 - from earlier training
        "Medium-Large": {"makespan": 3112.2, "std": 96.7},   # 20x15 - NEW: latest training
        "Large": {"makespan": 3926.5, "std": 122.5},         # 50x15 - NEW: latest training  
        "Very Large": {"makespan": 4493.3, "std": 132.2}     # 50x20 - NEW: latest training
    }
    
    # Heuristic Results (SPT, LPT, FIFO - no Random)
    heuristic_results = {
        "Small": {
            "SPT": {"makespan": 1760.0, "std": 112.1},
            "LPT": {"makespan": 1885.0, "std": 98.5},
            "FIFO": {"makespan": 1950.0, "std": 89.2}
        },
        "Medium": {
            "SPT": {"makespan": 2050.6, "std": 89.4},
            "LPT": {"makespan": 2298.8, "std": 115.7},
            "FIFO": {"makespan": 2156.2, "std": 97.3}
        },
        "Medium-Large": {
            "SPT": {"makespan": 2680.4, "std": 156.8},
            "LPT": {"makespan": 2949.6, "std": 187.2},
            "FIFO": {"makespan": 2789.2, "std": 142.5}
        },
        "Large": {
            "SPT": {"makespan": 2714.6, "std": 198.3},
            "LPT": {"makespan": 3156.8, "std": 234.1},
            "FIFO": {"makespan": 2889.4, "std": 201.7}
        },
        "Very Large": {
            "SPT": {"makespan": 2742.8, "std": 189.2},
            "LPT": {"makespan": 3198.4, "std": 245.6},
            "FIFO": {"makespan": 2912.6, "std": 201.3}
        }
    }
    
    # Order by increasing makespan: Small, Medium, Medium-Large, Large, Very Large
    size_order = ["Small", "Medium", "Medium-Large", "Large", "Very Large"]
    
    # Extract data for plotting
    sizes = size_order
    ppo_makespans = [ppo_results[size]["makespan"] for size in sizes]
    ppo_stds = [ppo_results[size]["std"] for size in sizes]
    
    spt_makespans = [heuristic_results[size]["SPT"]["makespan"] for size in sizes]
    spt_stds = [heuristic_results[size]["SPT"]["std"] for size in sizes]
    
    lpt_makespans = [heuristic_results[size]["LPT"]["makespan"] for size in sizes]
    lpt_stds = [heuristic_results[size]["LPT"]["std"] for size in sizes]
    
    fifo_makespans = [heuristic_results[size]["FIFO"]["makespan"] for size in sizes]
    fifo_stds = [heuristic_results[size]["FIFO"]["std"] for size in sizes]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(sizes))
    width = 0.2
    
    # Plot bars with error bars
    bars1 = plt.bar(x - 1.5*width, ppo_makespans, width, yerr=ppo_stds, 
                   label='PPO (Ours)', color='#2E86AB', capsize=5, alpha=0.8)
    bars2 = plt.bar(x - 0.5*width, spt_makespans, width, yerr=spt_stds,
                   label='SPT', color='#A23B72', capsize=5, alpha=0.8)
    bars3 = plt.bar(x + 0.5*width, lpt_makespans, width, yerr=lpt_stds,
                   label='LPT', color='#F18F01', capsize=5, alpha=0.8)
    bars4 = plt.bar(x + 1.5*width, fifo_makespans, width, yerr=fifo_stds,
                   label='FIFO', color='#C73E1D', capsize=5, alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Problem Size Category', fontsize=12, fontweight='bold')
    plt.ylabel('Average Makespan', fontsize=12, fontweight='bold')
    plt.title('JSSP Scaling Study: PPO vs Heuristic Baselines\nPerformance Across Problem Sizes', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels with dimensions
    x_labels = [f"{size}\n({problem_sizes[size]['dimensions']})" for size in sizes]
    plt.xticks(x, x_labels)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend
    plt.legend(loc='upper left', fontsize=11)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("eval_results_comprehensive")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "scaling_study_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "scaling_study_comprehensive.pdf", bbox_inches='tight')
    plt.show()  # Add this to display the plot
    
    print("📊 COMPREHENSIVE SCALING STUDY RESULTS")
    print("=" * 60)
    print()
    
    for size in sizes:
        ppo_makespan = ppo_results[size]["makespan"]
        spt_makespan = heuristic_results[size]["SPT"]["makespan"]
        lpt_makespan = heuristic_results[size]["LPT"]["makespan"]
        fifo_makespan = heuristic_results[size]["FIFO"]["makespan"]
        
        # Find best heuristic
        best_heuristic = min(spt_makespan, lpt_makespan, fifo_makespan)
        
        # Calculate gaps
        spt_gap = ((ppo_makespan - spt_makespan) / spt_makespan) * 100
        best_gap = ((ppo_makespan - best_heuristic) / best_heuristic) * 100
        
        print(f"🔍 {size} ({problem_sizes[size]['dimensions']}):")
        print(f"   PPO:  {ppo_makespan:.1f} ± {ppo_results[size]['std']:.1f}")
        print(f"   SPT:  {spt_makespan:.1f} ± {heuristic_results[size]['SPT']['std']:.1f}")
        print(f"   LPT:  {lpt_makespan:.1f} ± {heuristic_results[size]['LPT']['std']:.1f}")
        print(f"   FIFO: {fifo_makespan:.1f} ± {heuristic_results[size]['FIFO']['std']:.1f}")
        print(f"   Gap vs SPT: {spt_gap:+.1f}%")
        print(f"   Gap vs Best: {best_gap:+.1f}%")
        print()
    
    print("📈 KEY INSIGHTS:")
    print("=" * 60)
    
    # Calculate overall trends
    small_gap = ((ppo_results["Small"]["makespan"] - heuristic_results["Small"]["SPT"]["makespan"]) / 
                heuristic_results["Small"]["SPT"]["makespan"]) * 100
    very_large_gap = ((ppo_results["Very Large"]["makespan"] - heuristic_results["Very Large"]["SPT"]["makespan"]) / 
                     heuristic_results["Very Large"]["SPT"]["makespan"]) * 100
    
    print(f"• PPO vs SPT gap changes from {small_gap:+.1f}% (Small) to {very_large_gap:+.1f}% (Very Large)")
    print("• PPO shows competitive performance on smaller problems")
    print("• Performance gap increases significantly with problem complexity")
    print("• SPT consistently outperforms other heuristics")
    print("• All methods maintain 100% success rate")
    print("• PPO struggles more on larger, more complex instances")
    
    print(f"\n📁 Plots saved to: {output_dir}/scaling_study_comprehensive.png")
    print(f"📁 PDF version: {output_dir}/scaling_study_comprehensive.pdf")

if __name__ == "__main__":
    create_scaling_analysis() 