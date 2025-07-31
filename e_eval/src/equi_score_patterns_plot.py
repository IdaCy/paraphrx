"""
python3 e_eval/src/equi_score_patterns_plot.py a_data/alpaca/equi_scores/paraphrases_500_phrx_scores_histograms.json a_data/alpaca/equi_scores/paraphrases_500_phrx_scores.png --top 10

# multiple instruction types as side-by-side bars
python3 e_eval/src/equi_score_patterns_plot.py \
    a_data/alpaca/equi_scores/paraphrases_500_phrx_scores_histograms.json \
    a_data/alpaca/equi_scores/paraphrases_500_phrx_scores_multi_hist.png \
    instruct_casual instruct_formal

# top instruction types
python3 e_eval/src/equi_score_patterns_plot.py \
    a_data/alpaca/equi_scores/paraphrases_500_phrx_scores_histograms.json \
    a_data/alpaca/equi_scores/paraphrases_500_phrx_scores_top30_hist.png \
    --top 30

# Median score distribution
python3 e_eval/src/equi_score_patterns_plot.py \
    histograms.json median_hist.png --median median_scores.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
MAX_LABEL_LEN = 20 

if len(sys.argv) < 4:
    print("Usage: python plot_histograms.py histograms.json output.png instruct_type1 [instruct_type2 ...]")
    print("   OR: python plot_histograms.py histograms.json output.png --top N")
    print("   OR: python plot_histograms.py histograms.json output.png --median median.json")
    print("Example: python plot_histograms.py histograms.json plot.png instruct_casual instruct_formal")
    print("Example: python plot_histograms.py histograms.json plot.png --top 10")
    print("Example: python plot_histograms.py histograms.json plot.png --median median_scores.json")
    sys.exit(1)

hist_file = sys.argv[1]
output_file = sys.argv[2]
args = sys.argv[3:]

score_labels = [0, 1, 2, 3, 4, 5]
n_scores = len(score_labels)

# Check if using --median mode
if args[0] == "--median":
    if len(args) != 2:
        print("Error: --median requires exactly one median file path")
        sys.exit(1)
    
    median_file = args[1]
    with open(median_file, "r") as f:
        median_data = json.load(f)
    
    # Create histogram of median scores
    plt.figure(figsize=(10, 6))
    median_scores = list(median_data.values())
    plt.hist(median_scores, bins=range(0, 7, 1), alpha=0.7, edgecolor='black', align='left')
    plt.xlabel("Median Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Median Scores Across All Instruction Types")
    plt.xticks(range(6))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Median histogram saved to {output_file}")
    plt.close()
    
else:
    # Load histogram data for non-median modes
    with open(hist_file, "r") as f:
        hist = json.load(f)
    
    # Check if using --top mode
    if args[0] == "--top":
        if len(args) != 2:
            print("Error: --top requires exactly one number")
            sys.exit(1)
        try:
            n_top = int(args[1])
        except ValueError:
            print("Error: --top requires a valid number")
            sys.exit(1)
        
        # Sort by total count (as a simple proxy for mean score)
        sorted_types = sorted(hist.items(), key=lambda x: sum(x[1]), reverse=True)
        instruct_types = [item[0] for item in sorted_types[:n_top]]
        print(f"Plotting top {n_top} instruction types by total count")
    else:
        instruct_types = args

    plt.figure(figsize=(12, 8))
    
    # Set up colours for different instruction types
    colors = plt.cm.Set3(np.linspace(0, 1, len(instruct_types)))
    
    # Calculate bar width and positions
    bar_width = 0.8 / len(instruct_types)
    
    for i, instruct_type in enumerate(instruct_types):
        if instruct_type not in hist:
            print(f"Warning: {instruct_type} not found in histogram data")
            continue
        
        counts = np.array(hist[instruct_type])
        if counts.sum() == 0:
            print(f"Warning: {instruct_type} has no data")
            continue
        
        # Remove 'instruct_' prefix for cleaner labels

        label = instruct_type.replace('instruct_', '')
        if len(label) > MAX_LABEL_LEN:
            label = label[:MAX_LABEL_LEN - 3] + '...'
                
        # Calculate bar positions (offset for each instruction type)
        x_pos = np.arange(n_scores) + i * bar_width
        
        # Create histogram bars
        plt.bar(x_pos, counts, bar_width, label=label, alpha=0.7, color=colors[i])

    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score Distributions by Instruction Type")
    plt.legend(fontsize='small', ncol=2)
    plt.xticks(np.arange(n_scores) + bar_width * (len(instruct_types) - 1) / 2, score_labels)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram plot saved to {output_file}")
    plt.close()
