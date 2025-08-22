#!/usr/bin/env python3
"""
python3 b_tests/graphics/patching.py \
    -o b_tests/graphics/patching.png
"""
import matplotlib.pyplot as plt
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Generate AlpacaEval win rate bar chart')
    parser.add_argument('-o', '--output', type=str, help='Output file path (e.g., chart.png, chart.pdf)')
    parser.add_argument('--show', action='store_true', help='Display the chart in a window')
    
    args = parser.parse_args()
    
    # If no arguments provided, default to showing the chart
    if not args.output and not args.show:
        args.show = True

    scores = {
        "Base": 23.23,
        "Attn-only": 24.41,
        "MLP-only": 38.17,
        "Full FT": 39.86,
    }

    colors = {
        "Base": "#27ae60",
        "Attn-only": "#16a085",
        "MLP-only": "#1abc9c",
        "Full FT": "#8e44ad",
    }


    colors = {
        "Base": "#16a085",
        "Attn-only": "#2c3e50",
        "MLP-only": "#7f8c8d",
        "Full FT": "#34495e",
    }

    colors = {
        "Base": "#2e7d32",
        "Attn-only": "#34495e",
        "MLP-only": "#7f8c8d",
        "Full FT": "#2c3e50",
    }


    fig, ax = plt.subplots(figsize=(6,4))

    bars = ax.bar(scores.keys(), scores.values(),
                  color=[colors[k] for k in scores.keys()],
                  edgecolor="black")

    ax.set_ylabel("AlpacaEval win rate (%)")
    ax.set_title("Causal Patching at Layer 6")
    ax.set_ylim(0, 50)

    plt.tight_layout()
    
    # Save or show based on arguments
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {args.output}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
